import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import datetime
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

from pandas_datareader import data
now = datetime.datetime.now()
ticker = ["AAPL", "AMZN"]
start_date = '2010-01-01'
end_date = now.strftime("%Y-%m-%d")
panel_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
panel_data.reset_index(inplace = True)
panel_data
py.init_notebook_mode(connected = True)
trace0 = go.Scatter(x = panel_data["Date"], y = panel_data["Close"]["AAPL"], name = "AAPL", line = dict(color = "lime"))
trace1 = go.Scatter(x = panel_data["Date"], y = panel_data["Close"]["AMZN"], name = "AMZN", line = dict(color = 'gray'))
data = [trace0, trace1]
layout = dict(title = "Stock Prices of Apple vs Amazon",
             xaxis = dict(rangeselector = dict(buttons = list([
                 dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
                 dict(count = 3, label = "3m", step = "month", stepmode = "backward"),
                 dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
                 dict(count = 12, label = "1y", step = "month", stepmode = "backward"),
                 dict(count = 60, label = "5y", step = "month", stepmode = "backward")])),
                rangeslider = dict(visible = True), type = "date"),
             yaxis = dict(title = "price ($)"))
fig = dict(data = data, layout = layout)
py.iplot(fig)
trace0 = go.Box(y = panel_data["Close"]["AAPL"],
                name = 'AAPL Close',
                marker = dict(color = 'red'))
trace1 = go.Box(y = panel_data["Close"]["AMZN"],
                name = 'AMZN Close',
                marker = dict(color = 'navy'))
data = [trace0, trace1]
layout = dict(title='Stock Prices of APPLE & AMAZON')
fig = dict(data=data, layout=layout)
py.iplot(fig)
data = panel_data[[["Open"]["AAPL"], ["Close"]["AAPL"], ["Volume"]["AAPL"]]]
data["index"] = np.arange(len(data))
fig = ff.create_scatterplotmatrix(data, diag = "box", index = "index",
                                 size = 3, height = 700, colormap = "RdBu")
py.iplot(fig)