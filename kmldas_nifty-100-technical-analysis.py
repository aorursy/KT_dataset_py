import math

import pandas_datareader as web

import numpy as np

import pandas as pd

from datetime import datetime

import math







from subprocess import check_output



from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
N100_price= pd.read_excel("../input/nse-top-100-stocks/Index_NIFTY 100.xlsx")

#Show the data 

N100_price
N100_price.set_index('Date', inplace=True)

N100_price.info()
N100_price.describe()
#Visualize the closing price history

plt.figure(figsize=(16,8))

plt.title('Nifty 100 Close Price History')

plt.plot(N100_price['Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Close Price',fontsize=18)

plt.show()
m_a=50 #50 day moving average



ax = N100_price['Close'].plot(title="Nifty 100  Price with 50 Day Moving average", fontsize=12)

moving_average = N100_price['Close'].rolling(m_a).mean() 

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()
m_a=200 #200 day moving average



ax = N100_price['Close'].plot(title="Nifty 100  Price with 200 Day Moving average", fontsize=12)

moving_average = N100_price['Close'].rolling(m_a).mean() 

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()
ax = N100_price['Close'].plot(title="Nifty 100 Index Price with 50 & 200 Day Moving average", fontsize=12)



N100_price['Close'].rolling(50).mean().plot(label='Rolling mean', ax=ax)

N100_price['Close'].rolling(200).mean().plot(label='Rolling mean', ax=ax)

plt.show()
rolling_mean = N100_price['Close'].rolling(20).mean()



rolling_std = N100_price['Close'].rolling(20).std()



bollinger_upper_band = rolling_mean + rolling_std * 2



bollinger_lower_band = rolling_mean - rolling_std * 2


#Visualize the closing price history

plt.figure(figsize=(18,10))

plt.title('Nifty 100 Index Bollinger Bands')

ax = N100_price['Close'].plot(title="Bollinger Bands", fontsize=12)

rolling_average = rolling_mean

rolling_average.plot(label='Rolling mean', ax=ax)

bollinger_upper_band.plot(label='Bollinger upper band', ax=ax)

bollinger_lower_band.plot(label='Bollinger lower band', ax=ax)

plt.show()