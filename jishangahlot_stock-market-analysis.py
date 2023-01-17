import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import pandas_datareader

import datetime
import pandas_datareader.data as web
start = datetime.datetime(2017,1,1)

end = datetime.datetime(2020,1,1)
tesla = web.DataReader("TSLA","yahoo",start,end)
tesla.head()
ford = web.DataReader("F","yahoo",start,end)

gm = web.DataReader("GM","yahoo",start,end)
ford.head()
gm.head()
tesla["Open"].plot(label="Tesla", figsize=(16,8), title = "Opening Price")

ford["Open"].plot(label="Ford")

gm["Open"].plot(label="Genral Motor")

plt.legend()
tesla["Volume"].plot(label="Tesla", figsize=(16,8), title = "Volume Traded")

ford["Volume"].plot(label="Ford")

gm["Volume"].plot(label="Genral Motor")

plt.legend()
tesla['Total Trade'] = tesla['Open']*tesla['Volume']

ford['Total Trade'] = ford['Open']*ford['Volume']

gm['Total Trade'] = gm['Open']*gm['Volume']
tesla['Total Trade'].plot(label='Tesla',figsize=(16,8))

gm['Total Trade'].plot(label='GM')

ford['Total Trade'].plot(label='Ford')

plt.legend()

plt.ylabel('Total Trade')
gm['MA50'] = gm['Open'].rolling(50).mean()

gm['MA200'] = gm['Open'].rolling(200).mean()

gm[['Open','MA50','MA200']].plot(label='gm',figsize=(16,8))
from pandas.plotting import scatter_matrix
car_comp = pd.concat([tesla['Open'],gm['Open'],ford['Open']],axis=1)
car_comp.columns = ['Tesla Open','GM Open','Ford Open']
scatter_matrix(car_comp,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
# Method 1: Using shift

tesla['returns'] = (tesla['Close'] / tesla['Close'].shift(1) ) - 1
tesla.head()
# Method 2

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

plt.legend()
box_df = pd.concat([tesla['returns'],gm['returns'],ford['returns']],axis=1)

box_df.columns = ['Tesla Returns',' GM Returns','Ford Returns']

box_df.plot(kind='box',figsize=(8,11),colormap='jet')
scatter_matrix(box_df,figsize=(8,8),alpha=0.2,hist_kwds={'bins':50});
box_df.plot(kind='scatter',x=' GM Returns',y='Ford Returns',alpha=0.4,figsize=(10,8))
tesla['Cumulative Return'] = (1 + tesla['returns']).cumprod()
tesla.head()
ford['Cumulative Return'] = (1 + ford['returns']).cumprod()

gm['Cumulative Return'] = (1 + gm['returns']).cumprod()
tesla['Cumulative Return'].plot(label='Tesla',figsize=(16,8),title='Cumulative Return')

ford['Cumulative Return'].plot(label='Ford')

gm['Cumulative Return'].plot(label='GM')

plt.legend()