#importing all of the neccesary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import pandas_datareader

import datetime



import pandas_datareader.data as web
#creating a start and an end dates:

start = datetime.datetime(2016,1,1)

end = datetime.datetime.today()
#Shopify

shop = web.DataReader("SHOP", "yahoo", start, end)

shop.head()
#JD.com

jd = web.DataReader('JD', 'yahoo', start, end)

jd.head()
#Alibaba

alibaba = web.DataReader('BABA', 'yahoo', start, end)

alibaba.head()
shop['Open'].plot(label = 'Shopify', figsize = (16,10), title = 'Opening Prices')

jd['Open'].plot(label = 'JD.com')

alibaba['Open'].plot(label = 'Alibaba')

plt.legend(loc = 'best');
shop['Volume'].plot(label = 'Shopify', figsize = (16,10), title = 'Volume Traded')

jd['Volume'].plot(label = 'JD.com')

alibaba['Volume'].plot(label = 'Alibaba')

plt.legend();
shop['Total Traded'] = shop['Open']*shop['Volume']

jd['Total Traded'] = jd['Open']*jd['Volume']

alibaba['Total Traded'] = alibaba['Open']*alibaba['Volume']



shop['Total Traded'].plot(figsize = (16,8), label = 'Shopify')

jd['Total Traded'].plot(figsize = (16,8), label = 'JD.com')

alibaba['Total Traded'].plot(figsize = (16,8), label = 'Alibaba')

plt.legend(loc = 'best');
shop['Avg'] = shop[['High', 'Low']].mean(axis=1)

jd['Avg'] = jd[['High', 'Low']].mean(axis=1)

alibaba['Avg'] = alibaba[['High', 'Low']].mean(axis=1)



shop['Total Traded New'] = shop['Avg']*shop['Volume']

jd['Total Traded New'] = jd['Avg']*jd['Volume']

alibaba['Total Traded New'] = alibaba['Avg']*alibaba['Volume']



shop['Total Traded New'].plot(figsize = (16,8), label = 'Shopify')

jd['Total Traded New'].plot(figsize = (16,8), label = 'JD.com')

alibaba['Total Traded New'].plot(figsize = (16,8), label = 'Alibaba')

plt.legend(loc = 'best');
shop['Total Traded'].plot(figsize = (16,8), label = 'Shopify')

shop['Total Traded New'].plot(figsize = (16,8), label = 'Shopify of Avg')

plt.legend(loc = 'best');
shop['MA50'] = shop['Open'].rolling(50).mean()

shop['MA200'] = shop['Open'].rolling(200).mean()

shop[['Open','MA50','MA200']].plot(figsize = (16,10))



jd['MA50'] = jd['Open'].rolling(50).mean()

jd['MA200'] = jd['Open'].rolling(200).mean()



alibaba['MA50'] = alibaba['Open'].rolling(50).mean()

alibaba['MA200'] = alibaba['Open'].rolling(200).mean()
from pandas.plotting import scatter_matrix 



ret_comp = pd.concat([shop['Open'],jd['Open'],alibaba['Open']], axis = 1)

ret_comp.columns = ['Shopify Open', 'JD Open', 'Alibaba Open']

scatter_matrix(ret_comp, figsize =(8,8), alpha = 0.2, hist_kwds={'bins':50});
shop['Returns'] = shop['Close'].pct_change(1)

jd['Returns'] = jd['Close'].pct_change(1)

alibaba['Returns'] = alibaba['Close'].pct_change(1)

shop['Returns'].hist(bins = 50);
jd['Returns'].hist(bins = 50);
alibaba['Returns'].hist(bins = 50);
shop['Returns'].hist(bins = 100, figsize=(16,8), label = 'Shopify', alpha=0.4)

jd['Returns'].hist(bins = 100, figsize=(16,8), label = 'JD.com', alpha=0.4)

alibaba['Returns'].hist(bins = 100, figsize=(16,8), label = 'Alibaba', alpha=0.4)

plt.legend();
shop['Returns'].plot(kind='kde', label = 'Shopify', figsize=(10,8))

jd['Returns'].plot(kind='kde', label = 'JD.com',figsize=(10,8))

alibaba['Returns'].plot(kind='kde', label = 'Alibaba', figsize=(10,8))

plt.legend();
box_df = pd.concat([shop['Returns'], jd['Returns'], alibaba['Returns']], axis = 1)

box_df.columns = ['Shopify Ret', 'JD Ret', 'Alibaba Ret']

box_df.plot(kind='box', figsize = (8,10));
scatter_matrix(box_df, figsize=(8,8), alpha = 0.2, hist_kwds={'bins':100});
box_df.plot(kind='scatter', x='JD Ret', y='Alibaba Ret', alpha = 0.5, figsize = (10,8));