import datetime as dt 

import pandas as pd 

import pandas_datareader.data as web

import numpy as np 

import matplotlib.pyplot as plt 

from matplotlib import style

style.use('ggplot')
start = dt.datetime(2000, 1, 1)

end = dt.datetime(2016, 12, 31)
# df = web.DataReader("TSLA", 'yahoo', start, end)

# df.shape
#df.to_csv('Tesla.csv', index )
tesla_data = pd.read_csv('../input/Tesla.csv', parse_dates = True, index_col = 'Date')

tesla_data.shape
tesla_data.head()
tesla_data.plot()
tesla_data['High'].plot()
tesla_data[['Open', 'Close']].plot()
tesla_data['100ma'] = tesla_data['Adj Close'].rolling(window = 100).mean()
tesla_data.head()
tesla_data['100ma'] = tesla_data['Adj Close'].rolling(window = 100, min_periods = 0).mean()
tesla_data.head()
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan = 5, colspan = 1)

ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan = 1, colspan = 1, sharex = ax1)



# Plotting the data

ax1.plot(tesla_data.index, tesla_data['Adj Close'], label = "Adj Close")

ax1.plot(tesla_data.index, tesla_data['100ma'], label = "100 Moving Average")

ax2.plot(tesla_data.index, tesla_data['Volume'])



ax1.set_ylabel('Price')

ax1.set_xlabel('Time Period')

plt.legend()
# We can collect data for every 10 days as an OHLC as well using the OHLC function 

tesla_ohlc = tesla_data['Adj Close'].resample('10D').ohlc()

tesla_ohlc
#Resampling the data

tesla_data_10days  = tesla_data['Adj Close'].resample('10D')

type(tesla_data_10days)
mean_10days = tesla_data_10days.mean()

mean_10days
# Sum of 10 days data of volume 

volume_10days = tesla_data['Volume'].resample('10D').sum()

(volume_10days)
import matplotlib.dates as mdates

#from mpl_finance import candlestick_ohlc
tesla_ohlc.reset_index(inplace = True)

tesla_ohlc.head()
tesla_ohlc['Date'] = tesla_ohlc['Date'].map(mdates.date2num)
tesla_ohlc.head()
ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan = 5, colspan = 1)

ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan = 1, colspan = 1, sharex = ax1)



#Plotting the data

ax1.xaxis_date()



candlestick_ohlc(ax1, tesla_ohlc)