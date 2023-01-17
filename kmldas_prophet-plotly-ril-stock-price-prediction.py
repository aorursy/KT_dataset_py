

import math

import pandas_datareader as web

import numpy as np

import pandas as pd

from datetime import datetime

import math



from fbprophet import Prophet

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')





ril_price= pd.read_csv("../input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv")

#Show the data 

ril_price
ril_price=ril_price.dropna()

ril_price
ril_price.info()
ril_price["ds"]=pd.to_datetime(ril_price["Date"], format="%d-%m-%Y")

ril_price["y"]=ril_price["Close"]

ril_price=ril_price[['ds', 'y']]

ril_price
model = Prophet()

model.fit(ril_price)
future = model.make_future_dataframe(periods=365)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(model, forecast)  # This returns a plotly Figure

py.iplot(fig)