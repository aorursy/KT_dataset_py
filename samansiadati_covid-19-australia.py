import pandas as pd
address = '../input/covid19-australia/Covid-19 Australia.csv'
df = pd.read_csv(address)
df
import plotly.express as px
fig = px.line(df, x='date', y='case')
fig.show()
fig = px.line(df, x='date', y='death')
fig.show()
fig = px.line(df, x='date', y=df.columns)
fig.show()