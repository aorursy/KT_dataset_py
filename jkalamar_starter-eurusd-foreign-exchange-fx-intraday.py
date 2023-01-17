import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import plotly.figure_factory
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/eurusd.csv',parse_dates=[0],infer_datetime_format=True)

data.set_index('Unnamed: 0', inplace=True)

data.index.name = 'Date'
data.tail(10)
day = data.loc['2016-06-23':'2016-06-24',:] # Brexit Referendum Date

fig = plotly.figure_factory.create_candlestick(day.Open, day.High, day.Low, day.Close, dates=day.index)

fig.update_layout(title_text='EUR/USD 1-minute candlestick chart during UK\'s Brexit referendum')

fig.show()
# Here we resample to 15-minute intervals

data_15min = data.resample('15Min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})

data_15min.tail(10)
day = data_15min.loc['2016-06-23':'2016-06-24',:] # Brexit Referendum Date

fig = plotly.figure_factory.create_candlestick(day.Open, day.High, day.Low, day.Close, dates=day.index)

fig.update_layout(title_text='EUR/USD 15-minute candlestick chart during UK\'s Brexit referendum')

fig.show()