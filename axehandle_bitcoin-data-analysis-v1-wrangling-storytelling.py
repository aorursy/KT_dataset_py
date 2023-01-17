# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # making plots and charts

import requests # getting data through APIs



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the data from CSV file and save it to a dataframe



bitstamp = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
# Inspect the first a few rows of the data to see what potential cleaning is needed



bitstamp.head()
# Check the number of rows, columns and the datatypes of each column



bitstamp.info()
# Quickly check the statistics of all the data in each column to see if they make sense

# Based on my common sense historically the prices of Bitcoins went from 0 to an all 

# time high of around $20,000 per coin



bitstamp.describe()
# Inspect the shape of the dataset



bitstamp.shape
# There are around 1.2 million rows of data with missing values



bitstamp['Open'].value_counts(dropna = False)
# Do a quick histogram plot here to see the distribution of prices



bitstamp['Open'].plot('hist')
# Do a quick box plot here to see the distribution of prices



bitstamp.boxplot(column=['Open', 'High', 'Low', 'Close'])
# Set the index of the dataset to be the time of each observation in YYYY-MM-DD HH-MM-SS



bitstamp.set_index(pd.to_datetime(bitstamp['Timestamp'], unit='s'), inplace=True, drop=True)
# Inspect the dataset again



bitstamp.head()
# Fill the missing values using the forward fill method. 

# This method is appropriate here since the missing values in the original dataset were caused

# by the fact that there were not trading actions during those time periods, so it is safe to assume

# that the prices remained constant when there were no trading. Forward fill method takes the latest

# price data up to that point and filled it forward in time.



bitstamp.fillna(method = 'ffill', inplace = True)
# Inspect again



bitstamp.head()
# Also check the latest data. These values seem to make sense as compared to the actual prices in August this year



bitstamp.tail()
# Plot a histogram again to check the price distributions



bitstamp['Close'].plot('hist')
# Save the useful columns from the original dataset into a new and clean dataset called bitstamp_clean



bitstamp_clean = bitstamp.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]
# Inspect the information of the clean dataset



bitstamp_clean.info()
# Plot the time series price data



bitstamp_clean.plot(y='Close')
# Create a log-log plot of the closing prices for the past 7 years



bitstamp_clean.plot(y='Close', logx=True, logy=True)
# Obtain Bitcoin wallet data from Quandl 

# (which is a dataset of number of wallets hosts using My Wallet Service on each day from 2009 to 2019. )



wallet = pd.read_csv('../input/bitcoin-my-wallet-number-of-users/BCHAIN-MWNUS.csv')
# Inspect the first 5 rows to see the latest wallet data



wallet.head()
# Inspect the last 5 rows to see the oldest data from 2009



wallet.tail()
# Convert the date column to datetime format for easier processing later

# Also rename the columns while we are here



wallet['Date'] = pd.to_datetime(wallet['Date'])

wallet.rename(columns = {'Date': 'Date', 'Value': 'Wallets'}, inplace = True)
# Group our Bitcoin price data by day so that it could be plotted on the same scale

# against the daily wallet data



bitstamp_clean_day = bitstamp_clean.resample('D').mean()
# Create a date column in the bitstamp_clean_day dataframe



bitstamp_clean_day['Date'] = bitstamp_clean_day.index
# Inspect the first 5 rows to confirm that the timestamps are indeed grouped by days



bitstamp_clean_day.head()
# Join the two dataframes (bitstamp_clean_day and wallet) by matching their dates columns



df = pd.merge(bitstamp_clean_day, wallet, how='inner', on='Date')
# Inspect the first a few rows to confirm the data looks good to go



df.head()
# Plot both daily prices and daily number of wallets for Bitcoin on the same graph



plt.plot(df['Date'], df['Close'], 'r', df['Date'], df['Wallets']/10000, 'b')

plt.yscale('log')

plt.xlabel('Year')

plt.ylabel ('Price and Number of Wallets')

plt.title('Bitcoin Price compared to the Number of Wallets')

plt.legend(labels = ['Price', 'Wallets'])

plt.show()
# Import the Bitcoin difficulty dataset



diff = pd.read_csv('../input/bitcoin-difficulty/BCHAIN-DIFF.csv')
# Rename the columns for easier processing

# Also change the data format of the "Date" column while we are here



diff.rename(columns = {'Date': 'Date', 'Value': 'Difficulty'}, inplace = True)

diff['Date'] = pd.to_datetime(diff['Date'])
# Inspect the first a few rows of the dataset



diff.head()
# Merge these data with Bitcoin price dataframe for comparison later



df2 = pd.merge(bitstamp_clean_day, diff, how='inner', on='Date')
# Inspect the first a few rows of df2



df2.head()
# Plot both daily prices and level of difficulty for Bitcoin mining on the same graph



plt.plot(df2['Date'], df2['Close'], 'r', df2['Date'], df2['Difficulty']/100000, 'b')

plt.yscale('log')

plt.xlabel('Year')

plt.ylabel ('Price and Level of Difficulty')

plt.title('Bitcoin Price compared to the Level of Difficulty')

plt.legend(labels = ['Price', 'Difficulty'])

plt.show()