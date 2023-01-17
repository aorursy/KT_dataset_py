!pip install yahoofinance > /dev/null
import yahoofinance as yf

import pandas as pd

import plotly

import plotly.express as px
q = yf.HistoricalPrices('CBA.AX', '1900-01-01', '2019-11-15')

df = q.to_dfs()['Historical Prices']
df = df.dropna()

df
fig = px.line(df.reset_index(), x='Date', y='Adj Close')

fig.show()