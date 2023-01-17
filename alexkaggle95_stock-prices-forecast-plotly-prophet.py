import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns



import plotly

import plotly.plotly as py

import cufflinks as cf

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()



from fbprophet import Prophet



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/repsol.csv")

df.head(10)
df.info()


trace_open = go.Scatter(

    x = df["Date"],

    y = df["Open"],

    mode = 'lines',

    name="Open"

)



trace_high = go.Scatter(

    x = df["Date"],

    y = df["High"],

    mode = 'lines',

    name="High"

)



trace_low = go.Scatter(

    x = df["Date"],

    y = df["Low"],

    mode = 'lines',

    name="Low"

)



trace_close = go.Scatter(

    x = df["Date"],

    y = df["Close"],

    mode = 'lines',

    name="Close"

)







data = [trace_open,trace_high,trace_low,trace_close]



layout = go.Layout(title="Repsol Stock Price",xaxis_rangeslider_visible=True)



fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
trace_volume = go.Scatter(

    x = df["Date"],

    y = df["Volume"],

    mode = 'lines',

    name="Volume"

)



data_volume = [trace_volume]



layout_volume = go.Layout(title="Volume",xaxis_rangeslider_visible=True)



fig_volume = go.Figure(data=data_volume,layout=layout_volume)



plotly.offline.iplot(fig_volume)
df.shape
df_fc = df[935:3140] # 2010-10-21 date

df_fc.head()

model = Prophet()





df_prophet = df_fc.drop(['Open', 'High', 'Low','Volume'], axis=1)

df_prophet.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)



df_prophet.head(10)
model.fit(df_prophet)


future_prices = model.make_future_dataframe(periods=365)

forecast = model.predict(future_prices)

df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

df_forecast.head()
trace_open = go.Scatter(

    x = df_forecast["ds"],

    y = df_forecast["yhat"],

    mode = 'lines',

    name="Forecast"

)



trace_high = go.Scatter(

    x = df_forecast["ds"],

    y = df_forecast["yhat_upper"],

    mode = 'lines',

    fill = "tonexty", 

    line = {"color": "#57b8ff"}, 

    name="Higher uncertainty interval"

)



trace_low = go.Scatter(

    x = df_forecast["ds"],

    y = df_forecast["yhat_lower"],

    mode = 'lines',

    fill = "tonexty", 

    line = {"color": "#57b8ff"}, 

    name="Lower uncertainty interval"

)



trace_close = go.Scatter(

    x = df_prophet["ds"],

    y = df_prophet["y"],

    name="Data values"

)







data = [trace_open,trace_high,trace_low,trace_close]



layout = go.Layout(title="Repsol Stock Price Forecast",xaxis_rangeslider_visible=True)



fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)






trace = go.Candlestick(x = df_fc['Date'].values.tolist(),

                       open = df_fc['Open'].values.tolist(),

                       high = df_fc['High'].values.tolist(),

                       low = df_fc['Low'].values.tolist(),

                       close = df_fc['Close'].values.tolist(),

                      increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = go.Layout(title="Repsol Historical Price",

                   xaxis_rangeslider_visible=False,

                   yaxis= {'title': 'Stock Price'}

                  )









data = [trace]



fig = go.Figure(data=data,layout=layout)



plotly.offline.iplot(fig)
