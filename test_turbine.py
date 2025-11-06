import streamlit as st
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import altair as alt

st.set_page_config(layout="wide")
st.title("Turbine Power & Efficiency Predictor ")

# Sidebar turbine selection
turbine_type = st.sidebar.selectbox("Select Turbine Type", [
    "Wind Turbine",
    "Hydraulic Turbine",
    "Steam Turbine",
    "Gas Turbine"
])

st.write("---")

# File upload (optional)
file = st.file_uploader("Upload CSV (optional). If CSV contains a measured 'Power_kW' column we will train an ML model.", type=["csv"])

# Manual input section (no efficiency input)-----------------
st.subheader("Or enter parameters manually (efficiency will be computed from predicted mechanical output)")
manual = {}

if turbine_type == "Wind Turbine":
    V = st.number_input("Wind Speed V (m/s)", 8.0, step=0.1)
    r = st.number_input("Blade Radius r (m)", 30.0, step=0.1)
    Cp = st.number_input("Power Coefficient Cp (0-1)", 0.42, step=0.01)
    rho = st.number_input("Air Density rho (kg/m^3)", 1.225, step=0.001)
    manual = {"V": V, "r": r, "Cp": Cp, "rho": rho}

elif turbine_type == "Hydraulic Turbine":
    Q = st.number_input("Flow Rate Q (m^3/s)", 20.0, step=0.1)
    H = st.number_input("Head H (m)", 50.0, step=0.1)
    manual = {"Q": Q, "H": H}

elif turbine_type == "Steam Turbine":
    m = st.number_input("Mass Flow m (kg/s)", 10.0, step=0.1)
    h1 = st.number_input("Inlet Enthalpy h1 (kJ/kg)", 3000.0, step=1.0)
    h2 = st.number_input("Exit Enthalpy h2 (kJ/kg)", 2500.0, step=1.0)
    manual = {"m": m, "h1": h1, "h2": h2}

elif turbine_type == "Gas Turbine":
    m = st.number_input("Mass Flow m (kg/s)", 10.0, step=0.1)
    cp = st.number_input("Specific Heat cp (kJ/kg·K)", 1.0, step=0.01)
    T3 = st.number_input("Inlet Temp T3 (K)", 1400.0, step=1.0)
    T4 = st.number_input("Exit Temp T4 (K)", 800.0, step=1.0)
    manual = {"m": m, "cp": cp, "T3": T3, "T4": T4}

st.write("---")
run_button = st.button("Start Prediction")

# Helper functions ------------------------------------------------
def compute_available_input_power_kW(row, turbine):
    # returns available input power in kW
    if turbine == "Wind Turbine":
        A = math.pi * (row['r'] ** 2)
        P_available = 0.5 * row['rho'] * A * (row['V'] ** 3) / 1000.0
        return P_available
    if turbine == "Hydraulic Turbine":
        rho, g = 1000.0, 9.81
        return rho * g * row['Q'] * row['H'] / 1000.0
    if turbine == "Steam Turbine":
        # m (kg/s) * (h1 - h2) (kJ/kg) --> kW (since kJ/s = kW)
        return row['m'] * (row['h1'] - row['h2'])
    if turbine == "Gas Turbine":
        # m * cp (kJ/kgK) * (T3-T4) (K) --> kW
        return row['m'] * row['cp'] * (row['T3'] - row['T4'])
    return 0.0


def predict_power_with_model(model, feat_df):
    return model.predict(feat_df)


def physics_predict_power(row, turbine):
    # Provide fallback physics-based estimate for mechanical output (uses typical extraction factors)
    if turbine == "Wind Turbine":
        # assume a realistic extraction fraction of available power using Cp (provided)
        A = math.pi * (row['r'] ** 2)
        return 0.5 * row['rho'] * A * (row['V'] ** 3) * row.get('Cp', 0.4) / 1000.0
    if turbine == "Hydraulic Turbine":
        rho, g = 1000.0, 9.81
        # assume turbine extracts 90% by default
        return 0.9 * rho * g * row['Q'] * row['H'] / 1000.0
    if turbine == "Steam Turbine":
        # assume 35% mechanical conversion if no data
        return 0.35 * row['m'] * (row['h1'] - row['h2'])
    if turbine == "Gas Turbine":
        # assume 30% mechanical conversion by default
        return 0.30 * row['m'] * row['cp'] * (row['T3'] - row['T4'])
    return 0.0

# Main execution ----------------------------------------------------
if run_button:
    # Prepare dataframe for processing
    if file is not None:
        df = pd.read_csv(file)
        st.write("### Uploaded CSV preview")
        st.dataframe(df.head())
    else:
        df = pd.DataFrame([manual])

    # Ensure required columns exist for available power calculation
    # For manual single-row df it's fine; for CSV, we expect appropriate columns present

    # Check if CSV contains a measured power column we can train on
    target_col = None
    for possible in ['Power_kW', 'Measured_Power_kW', 'MeasuredPower_kW']:
        if file is not None and possible in df.columns:
            target_col = possible
            break

    # Build features dataframe (numeric columns excluding target)
    features = df.select_dtypes(include=[np.number]).copy()
    if target_col and target_col in features.columns:
        features = features.drop(columns=[target_col])

    # Compute available input power for each row
    available = []
    for idx, row in df.iterrows():
        # row may be a Series; convert to dict for safe key access
        rdict = row.to_dict()
        available.append(compute_available_input_power_kW(rdict, turbine_type))
    df['AvailableInput_kW'] = available

    # If we can train ML (target exists), train model to predict mechanical output
    model = None
    if target_col is not None:
        st.write("### Training RandomForest model from provided measured power column:", target_col)
        # prepare X and y
        X = features.fillna(0.0)
        y = df[target_col].fillna(0.0)
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            st.success("Model trained successfully — now using ML predictions for mechanical output.")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            model = None

    # Predict mechanical output (Power_mech_kW)
    predicted = []
    for idx, row in df.iterrows():
        rdict = row.to_dict()
        if model is not None:
            # use model features; align columns
            X_row = pd.DataFrame([ {c: rdict.get(c, 0.0) for c in X.columns} ])
            try:
                p = float(model.predict(X_row)[0])
            except Exception:
                p = physics_predict_power(rdict, turbine_type)
        else:
            p = physics_predict_power(rdict, turbine_type)
        predicted.append(p)

    df['PredictedPower_kW'] = predicted

    # Efficiency = PredictedPower / AvailableInput
    eff = []
    for idx, row in df.iterrows():
        avail = row['AvailableInput_kW']
        p = row['PredictedPower_kW']
        if avail is None or avail == 0:
            eff.append(np.nan)
        else:
            eff.append(100.0 * (p / avail))
    df['Efficiency_%'] = eff

    # For wind turbine, also compute blade area for plotting
    if turbine_type == 'Wind Turbine':
        df['BladeArea_m2'] = df.apply(lambda r: math.pi * (r['r'] ** 2), axis=1)
    else:
        df['BladeArea_m2'] = np.nan

    st.write("### Results")
    st.dataframe(df[['PredictedPower_kW', 'AvailableInput_kW', 'Efficiency_%', 'BladeArea_m2']])

    # Create x-axis labels combining power and blade area (as requested)
    labels = []
    for idx, r in df.iterrows():
        p = r['PredictedPower_kW']
        a = r['BladeArea_m2']
        if pd.isna(a):
            a_str = "-"
        else:
            a_str = f"{a:.1f} m2"
        labels.append(f"P={p:.1f}kW, A={a_str}")

    df['label'] = labels

    # Combined chart: Efficiency (bar) + Predicted Power (line)
    chart_df = df[['label', 'Efficiency_%', 'PredictedPower_kW']].copy().reset_index(drop=True)

    # Aesthetic theme
    color_eff = '#6A5ACD'  # soft violet
    color_pow = '#FF8C00'  # warm orange

    bar = alt.Chart(chart_df).mark_bar(opacity=0.75, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X('label:N', sort=None, title=''),
        y=alt.Y('Efficiency_%:Q', title='Efficiency (%)'),
        color=alt.value(color_eff),
        tooltip=['label', 'PredictedPower_kW', 'Efficiency_%']
    )

    line = alt.Chart(chart_df).mark_line(point=alt.OverlayMarkDef(size=60, filled=True, fill=color_pow)).encode(
        x='label:N',
        y=alt.Y('PredictedPower_kW:Q', title='Predicted Power (kW)'),
        color=alt.value(color_pow)
    )

    combined_chart = alt.layer(bar, line).resolve_scale(y='independent').properties(
        width=850,
        height=420,
        background='white'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        labelColor='black',
        titleColor='black'
    ).configure_view(
        stroke='transparent'
    )

    st.write("### 🎨 Aesthetic Combined Power & Efficiency Chart")
    st.altair_chart(combined_chart, use_container_width=True)

    # Download results
    st.download_button("⬇️ Download Results CSV", df.to_csv(index=False), file_name="turbine_results.csv")
    st.success("Prediction complete — efficiency computed from predicted mechanical output.")

else:
    st.info("Upload a CSV or enter manual parameters, then click 'Start Prediction'.")
