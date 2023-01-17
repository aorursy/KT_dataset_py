!pip3 uninstall --yes fbprophet

!pip3 install fbprophet --no-cache-dir --no-binary :all:
import os

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



%matplotlib inline

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from scipy.stats import norm

import datetime as dt

from fbprophet import Prophet
all_stocks = pd.read_csv('../input/all_stocks_5yr.csv', index_col = 'date', parse_dates=['date'])

all_stocks.head()
apple = all_stocks[all_stocks['Name'] == 'AAPL']

apple.head()
apple.info()
apple.describe()
apple['2014':'2017'].plot(subplots=True, figsize=(10, 12))

plt.title('Apple stock attributes from 2014 to 2017')
amazon = all_stocks[all_stocks['Name'] == 'AMZN']

amazon.head()
apple.high.plot()

amazon.high.plot()

plt.legend(['Apple', 'Amazon'])
norm_apple = apple.high.div(apple.high.iloc[0]).mul(100)

norm_amazon = amazon.high.div(amazon.high.iloc[0]).mul(100)

norm_apple.plot()

norm_amazon.plot()

plt.legend(['Apple', 'Amazon'])
trace = go.Ohlc(x = apple.index,

               open = apple['open'],

               high = apple['high'],

               low = apple['low'],

               close = apple['close'])



layout = {

    'xaxis': {

        'title':'Date',

        'rangeslider':{'visible':False}

    },

    'yaxis': {

        'title':'Price in US Dollars'

    },

    'shapes': [{

        'x0': '2016-11-08', 'x1': '2016-11-08',

        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',

        'line': {'color': 'rgb(30,30,30)', 'width': 1}

    }],

    'annotations': [{

        'x':'2016-11-08', 'y':0.05, 'xref':'x', 'yref':'paper', 'showarrow':False, 'text': 'US Presidential Election'

    }]

}



data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
apple['close_std'] = apple['close'].rolling(5).std()

apple['close_mean'] = apple['close'].rolling(5).mean()



fig, (std, avg) = plt.subplots(1, 2, figsize=(16, 8))

std.plot(apple.index, apple['close_std'], label='5 day Standard Deviation')

std.legend()



avg.plot(apple.index, apple['close_mean'], label='5 Day Moving Average', color='green')

avg.plot(apple.index, apple['close'].rolling(20).mean(), label='20 Day Moving Average', color='red')

avg.legend()
apple_two = apple['2016':'2018']

apple_two = apple_two.reset_index()

apple_two.head()
apple_monthly = apple_two.groupby(['date'])
agg = apple_monthly.aggregate({'open':np.mean, 'high':np.mean, 'low':np.mean, 'close':np.mean})

agg = agg.reset_index()

agg.head()
trace = go.Candlestick(x = agg['date'],

                      open = agg['open'].values.tolist(),

                      high = agg['high'].values.tolist(),

                      low = agg['low'].values.tolist(),

                      close = agg['close'].values.tolist()

                      )

layout = {

    'title':'Apple Stock from 2016 to 2018',

    'xaxis': {'title':'Date',

             'rangeslider':{'visible':False}},

    'yaxis':{'title':'Price in US Dollars'}

}





data = [trace]





fig_candle = go.Figure(data, layout)

iplot(fig_candle)
apple_2018 = apple['2018']

apple_2018 = apple_2018.reset_index()

apple_2018.head()
apple_m = apple_2018.groupby(['date'])

agg_m = apple_m.aggregate({'open':np.mean, 'close':np.mean, 'high':np.mean, 'low':np.mean})

agg_m = agg_m.reset_index()



trace = go.Candlestick(x = agg_m['date'],

                       open = agg_m['open'],

                       high = agg_m['high'],

                       low = agg_m['low'],

                       close = agg_m['close']

                      )



data = [trace]



layout = {

    'title':'Closer look at Apple Stock for downward trend',

    'xaxis': {'title':'Date',

             'rangeslider':{'visible':False}},

    'yaxis':{'title':'Price in US Dollars'}

}



fig_candle1 = go.Figure(data=data, layout=layout)

iplot(fig_candle1)
apple.head()
model = Prophet()

ph_apple = apple.drop(['open', 'high', 'low', 'volume', 'Name', 'close_std', 'close_mean'], axis=1)



ph_apple = ph_apple.reset_index()
ph_apple.head()
ph_apple = ph_apple.rename(columns = {'close':'y', 'date':'ds'})
model.fit(ph_apple)
future = model.make_future_dataframe(periods=365)

future.tail()
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].tail()
fig1 = model.plot(forecast)
from fbprophet.plot import add_changepoints_to_plot

fig1 = model.plot(forecast)

a = add_changepoints_to_plot(fig1.gca(), model, forecast)
fig2 = model.plot_components(forecast)
model = Prophet(changepoint_prior_scale=0.05).fit(ph_apple)

future_m = model.make_future_dataframe(periods=12, freq='M')

m_forecast = model.predict(future_m)

fig1 = model.plot(m_forecast)

plt.title('Monthly Predictions (1 year timeframe)')
fig2 = model.plot_components(m_forecast)