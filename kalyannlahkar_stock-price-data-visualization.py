#Basic Setup and some customization for matplotlib

import numpy as np

import pandas as pd 

import matplotlib as mpl

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.style.use('seaborn')

mpl.rc('axes', titlesize=14, titleweight='semibold')

#Getting the Data

aapl = pd.read_csv("../input/stock-time-series-20050101-to-20171231/AAPL_2006-01-01_to_2018-01-01.csv", \

                   parse_dates=["Date"])

aapl.head()
#Adding Returns and Cumulative returns to the aapl dataframe

aapl['returns'] = aapl['Close'].pct_change(1)

aapl['CumulativeReturn'] = (1+aapl['returns']).cumprod()

aapl.head()
#Moving Average Indicators

aapl['MA20'] = aapl['Close'].rolling(20).mean()

aapl['STD20'] = aapl['Close'].rolling(20).std()

aapl['MA150'] = aapl['Close'].rolling(150).mean()

aapl.head()
#Getting rid of Na values and setting the Date from 2012 onwards

aapl.dropna(inplace=True)

aapl = aapl.loc[aapl['Date'] >= '2012-01-01']

aapl.head()
#plotting the distrbution of returns of apple stock over the years

plt.figure(figsize=(16,10))

plt.hist(aapl['returns'], density=True, bins=200)

plt.title("Distribution of Returns")

plt.xlabel("Returns")

plt.ylabel("Frequency")

plt.xlim(-0.20, 0.20)

plt.show();
trading_days = aapl['Date']

fig, ax = plt.subplots(figsize=(16,10))

ax.plot(trading_days, aapl['Close'],label="Closing Price", color="#3498db")

ax.plot(trading_days, aapl['MA20'], label="20 Days SMA", color="#e67e22")

ax.plot(trading_days, aapl['MA150'], label="150 Days SMA", color="#95a5a6")

ax.set_title("Apple Stock Prices with Indicators", fontsize="14", fontweight="semibold")

ax.set_xlim([trading_days.min(), trading_days.max()])

plt.legend();
fig, priceplot = plt.subplots(figsize=(16, 10))

priceplot.plot(trading_days, aapl[["Close"]], color="#3498db", label="Closing Price")

priceplot.plot(trading_days, aapl[["MA20"]], color="#e67e22", label="20 Days SMA")

priceplot.plot(trading_days, aapl[["MA150"]], color="#95a5a6", label="150 Days SMA")

priceplot.set_xlim([trading_days.min(), trading_days.max()])

priceplot.set_title("Apple Stock Prices with Indicators and Volume", fontsize="14", fontweight="semibold")

plt.legend()



volplot = priceplot.twinx()

volplot.set_ylim([0,aapl['Volume'].max()*3])

volplot.plot(trading_days, aapl[["Volume"]], color="#2ecc71", label="volume traded", alpha=0.5)

volplot.grid(False)

plt.legend();

#Bollinger Bands

aapl['lower_band'] = aapl['MA20'] - 2*aapl['STD20']

aapl['upper_band'] = aapl['MA20'] + 2*aapl['STD20']

aapl.head()
fig, ax = plt.subplots(figsize=(16,10))

ax.plot(trading_days, aapl['Close'],label="Closing Price", color="#3498db")

ax.fill_between(trading_days, aapl['upper_band'], aapl['lower_band'], color='#e67e22', alpha=0.6, label="Bollinger Band")

ax.set_title('Apple Stock Prices with Bollinger Bands')

ax.set_xlim(trading_days.min(), trading_days.max())

plt.legend();