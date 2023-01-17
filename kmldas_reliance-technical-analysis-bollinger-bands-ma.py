

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





ril_price= pd.read_csv("../input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv")

#Show the data 

ril_price
ril_price=ril_price.dropna()

ril_price
ril_price.info()
ril_price["Date"]=pd.to_datetime(ril_price["Date"], format="%d-%m-%Y")





ril_price["Date"]



ril_price.set_index('Date', inplace=True)

ril_price.info()
ril_price.describe()
#Visualize the closing price history

plt.figure(figsize=(16,8))

plt.title('Reliance Industries Close Price History')

plt.plot(ril_price['Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Close Price INR',fontsize=18)

plt.show()
m_a=50 #50 day moving average



ax = ril_price['Close'].plot(title="RIL Stock Price with 50 Day Moving average", fontsize=12)

moving_average = ril_price['Close'].rolling(m_a).mean() 

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()
m_a=200 #200 day moving average



ax = ril_price['Close'].plot(title="RIL Stock Price with 200 Day Moving average", fontsize=12)

moving_average = ril_price['Close'].rolling(m_a).mean() 

moving_average.plot(label='Rolling mean', ax=ax)

plt.show()
ax = ril_price['Close'].plot(title="RIL Stock Price with 50 & 200 Day Moving average", fontsize=12)



ril_price['Close'].rolling(50).mean().plot(label='Rolling mean', ax=ax)

ril_price['Close'].rolling(200).mean().plot(label='Rolling mean', ax=ax)

plt.show()
rolling_mean = ril_price['Close'].rolling(20).mean()



rolling_std = ril_price['Close'].rolling(20).std()



bollinger_upper_band = rolling_mean + rolling_std * 2



bollinger_lower_band = rolling_mean - rolling_std * 2


#Visualize the closing price history

plt.figure(figsize=(18,10))

plt.title('Reliance Industries Bollinger Bands')

ax = ril_price['Close'].plot(title="Bollinger Bands", fontsize=12)

rolling_average = rolling_mean

rolling_average.plot(label='Rolling mean', ax=ax)

bollinger_upper_band.plot(label='Bollinger upper band', ax=ax)

bollinger_lower_band.plot(label='Bollinger lower band', ax=ax)

plt.show()