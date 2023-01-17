import pandas as pd
address = "../input/nab/realKnownCause/realKnownCause/ambient_temperature_system_failure.csv"
df = pd.read_csv(address)
df
import plotly.graph_objects as go
fig = go.Figure([go.Scatter(x=df['timestamp'], y=df['value'])])
fig.show()