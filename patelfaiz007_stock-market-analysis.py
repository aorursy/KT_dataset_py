# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
import pandas_datareader
import datetime
import pandas_datareader.data as web
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2019, 1, 1)
#tesla = web.DataReader("TSLA", 'yahoo', start, end)
#ford = web.DataReader("TSLA", 'yahoo', start, end)
#gm = web.DataReader("TSLA", 'yahoo', start, end)

# For some reason DataReader doesn't worked for me, I tried Google as well as Yahoo. So created csv files manually.
tesla = pd.read_csv('../input/Tesla_Stock.csv')
ford = pd.read_csv('../input/Ford_Stock.csv')
gm = pd.read_csv('../input/GM_Stock.csv')
tesla['Date'] = pd.to_datetime(tesla['Date'])
ford['Date'] = pd.to_datetime(ford['Date'])
gm['Date'] = pd.to_datetime(gm['Date'])
tesla.set_index('Date', inplace = True)
ford.set_index('Date', inplace = True)
gm.set_index('Date', inplace = True)
tesla.head()
ford.head()
gm.head()
# Visualizing the Data
tesla['Open'].plot(label='Tesla',figsize=(16,8),title='Open Price')
gm['Open'].plot(label='GM')
ford['Open'].plot(label='Ford')
plt.legend()
plt.show()
# Volume of Stock traded every day

tesla['Volume'].plot(label='Tesla',figsize=(16,8),title='Volume Traded')
gm['Volume'].plot(label='gm')
ford['Volume'].plot(label='ford')
plt.legend()
plt.show()
#Interesting, looks like Ford had a really big spike somewhere in late 2013.

#What happened that day?
ford['Volume'].argmax()
# What happened:
# http://money.cnn.com/2013/12/18/news/companies/ford-profit/
# https://www.usatoday.com/story/money/cars/2013/12/18/ford-2014-profit-warning/4110015/
# https://media.ford.com/content/dam/fordmedia/North%20America/US/2014/01/28/4QFinancials.pdf
tesla['Total Traded'] = tesla['Open']*tesla['Volume']
ford['Total Traded'] = ford['Open']*ford['Volume']
gm['Total Traded'] = gm['Open']*gm['Volume']
tesla['Total Traded'].plot(label='Tesla',figsize=(16,8))
gm['Total Traded'].plot(label='GM')
ford['Total Traded'].plot(label='Ford')
plt.legend()
plt.ylabel('Total Traded')
plt.show()
#Interesting, looks like there was huge amount of money traded for Tesla somewhere in early 2014 and recent years. What date was that and what happened?
tesla['Total Traded'].argmax()
# I found this:
# https://www.cnbc.com/2018/08/07/tesla-says-no-final-decision-has-been-made-to-take-company-private.html
# Let's plot some Moving Averages

gm['MA50'] = gm['Open'].rolling(50).mean()
gm['MA200'] = gm['Open'].rolling(200).mean()
gm[['Open','MA50','MA200']].plot(label='gm',figsize=(16,8));
# Let's check if there is a relationship between these Stocks!
from pandas.plotting import scatter_matrix
car_comp = pd.concat([tesla['Open'],gm['Open'],ford['Open']],axis=1)
car_comp.columns = ['Tesla Open','GM Open','Ford Open']
scatter_matrix(car_comp,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
# Looks like Ford and GM had some relationships in the past, but it seems to be changing in the recent years!
# Daily Percentage Change
#First we will begin by calculating the daily percentage change. Daily percentage change is defined by the following formula:
# rt = (pt/(pt−1)) − 1
# This defines r_t (return at time t) as equal to the price at time t divided by the price at time t-1 (the previous day) minus 1.
#Basically this just informs you of your percent gain (or loss) if you bought the stock on day and then sold it the next day.
#While this isn't necessarily helpful for attempting to predict future values of the stock, its very helpful in analyzing the volatility of the stock.
#If daily returns have a wide distribution, the stock is more volatile from one day to the next.
#Let's calculate the percent returns and then plot them with a histogram, and decide which stock is the most stable!
tesla['returns'] = (tesla['Close'] / tesla['Close'].shift(1) ) - 1
tesla.head()
# The below code does the same thing
tesla['returns'] = tesla['Close'].pct_change(1)
tesla.head()
# Now repeat for the other dataframes
ford['returns'] = ford['Close'].pct_change(1)
gm['returns'] = gm['Close'].pct_change(1)
ford.head()
gm.head()
ford['returns'].hist(bins=50)
gm['returns'].hist(bins=50)
tesla['returns'].hist(bins=50)
tesla['returns'].hist(bins=100,label='Tesla',figsize=(10,8),alpha=0.5)
gm['returns'].hist(bins=100,label='GM',alpha=0.5)
ford['returns'].hist(bins=100,label='Ford',alpha=0.5)
plt.legend();
tesla['returns'].plot(kind='kde',label='Tesla',figsize=(12,6))
gm['returns'].plot(kind='kde',label='GM')
ford['returns'].plot(kind='kde',label='Ford')
plt.legend()
box_df = pd.concat([tesla['returns'],gm['returns'],ford['returns']],axis=1)
box_df.columns = ['Tesla Returns',' GM Returns','Ford Returns']
box_df.plot(kind='box',figsize=(8,11),colormap='jet')
# Comparing Daily Returns Between Stocks
scatter_matrix(box_df,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
# It looks like Ford and GM do have some sort of possible relationship, let's plot just these two against eachother in scatter plot to view this more closely!
box_df.plot(kind='scatter',x=' GM Returns',y='Ford Returns',alpha=0.4,figsize=(10,8))
# Cumulative Daily Returns
tesla['Cumulative Return'] = (1 + tesla['returns']).cumprod()
tesla.head()
ford['Cumulative Return'] = (1 + ford['returns']).cumprod()
gm['Cumulative Return'] = (1 + gm['returns']).cumprod()
tesla['Cumulative Return'].plot(label='Tesla',figsize=(16,8),title='Cumulative Return')
ford['Cumulative Return'].plot(label='Ford')
gm['Cumulative Return'].plot(label='GM')
plt.legend();
# Tesla gave the highest Returns as compared to other two companies!