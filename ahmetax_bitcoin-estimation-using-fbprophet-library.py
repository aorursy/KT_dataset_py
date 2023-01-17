## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import fbprophet
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import plotly.graph_objs as go
import plotly as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/btcusd.csv")
# important: For fbprophet, dataframe column names must be 'ds' and 'y'. Their types must be datetime and float.
df['ds'] = pd.to_datetime(df['ds'])
df['y']=df['y'].astype(float)
df.info()
df.describe()
# Actually there is no need to draw this graph. Actual data is included the estimation chart.
# plt.plot(df['ds'],df['y'])
# plt.title("BTC Prices in USD")
# plt.ylabel('Price (USD)')
# plt.xlabel('Dates')
# plt.savefig('btc01.png')
# plt.show()
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
df_prophet.fit(df)

# Forecast for 4 months
fcast_time=123   # 4 months
df_forecast = df_prophet.make_future_dataframe(periods= fcast_time, freq='D')

# Do forecasting
df_forecast = df_prophet.predict(df_forecast)
df_prophet.plot(df_forecast, xlabel = 'Dates', ylabel = 'BTC Price in USD')
plt.savefig('btc02.png')
plt.show()
trace = go.Scatter(
        name = 'Actual price',
       mode = 'markers',
       x = list(df_forecast['ds']),
       y = list(df['y']),
       marker=dict(
              color='#FFBAD2',
              line=dict(width=1)
       )
)

trace1 = go.Scatter(
    name = 'trend',
       mode = 'lines',
       x = list(df_forecast['ds']),
       y = list(df_forecast['yhat']),
       marker=dict(
              color='red',
              line=dict(width=1)
       )
)

upper_band = go.Scatter(
    name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
)

lower_band = go.Scatter(
    name= 'lower band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
)

data = [trace, trace1, lower_band, upper_band]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=False))
figure=dict(data=data,layout=layout)
plt.savefig('btc03.png')
py.offline.iplot(figure)