import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web


style.use('ggplot')

start = dt.datetime(2003,1,1)
end = dt.datetime(2019,12,31)

df = web.DataReader('GOOGL', 'yahoo', start, end)
df.head(10)


#Stock's Highest Value over time
df['High'].plot()
plt.plot()
#Volume of shares sold
df['Volume'].plot()
plt.plot()
#Moving Average and Data Manipulation
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
df.dropna(inplace=True)

print(df.tail())
#Comparison of Adj Close, 100ma, and Volume
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()


