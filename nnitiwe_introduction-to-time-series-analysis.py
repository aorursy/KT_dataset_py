import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



btc_usd=pd.read_csv('../input/Bitstamp_BTCUSD.csv')

btc_usd.head()
#lets see what our dataset looks like

print(btc_usd.dtypes)

btc_usd.describe()
btc_usd['Date'] =btc_usd['Date'].astype(str)+ ' 00:00:00'



btc_usd.Date = pd.to_datetime(btc_usd.Date,format='%m/%d/%Y %H:%M:%S').dt.strftime('%d-%m-%Y %H:%M:%S')

btc_usd.Date = btc_usd.Date.astype('datetime64[ns]')

print(btc_usd.dtypes)

btc_usd.head()
btc_usd.set_index('Date', inplace=True)

btc_usd.head()
btc_usd['2019-05-01':'2019-05-31'].head()
weekly_btcusd=btc_usd.resample('W').mean()

weekly_btcusd.fillna(method='ffill',inplace=True) #handle missing records 

print('Weekly BTC/USD')



weekly_btcusd.head()
weekly_btcusd['Volume BTC'].plot()
weekly_btcusd['Datetime']=weekly_btcusd.index



weekly_btcusd = weekly_btcusd.sort_values('Datetime')

weekly_btcusd['Weeks'] = pd.factorize(weekly_btcusd['Datetime'])[0] + 1

mapping = dict(zip(weekly_btcusd['Weeks'], weekly_btcusd['Datetime'].dt.date))



ax = sns.regplot(x='Weeks',y='Volume BTC',data=weekly_btcusd,scatter_kws={'alpha':0.5},fit_reg=False)

labels = pd.Series(ax.get_xticks()).map(mapping).fillna('')





monthly_btcusd=btc_usd.resample('M').mean()

monthly_btcusd.fillna(method='ffill',inplace=True)

print('Monthly BTC/USD')



monthly_btcusd.head()
monthly_btcusd['Volume BTC'].plot()

monthly_btcusd['Datetime']=monthly_btcusd.index



monthly_btcusd = monthly_btcusd.sort_values('Datetime')

monthly_btcusd['Months'] = pd.factorize(monthly_btcusd['Datetime'])[0] + 1

mapping = dict(zip(monthly_btcusd['Months'], monthly_btcusd['Datetime'].dt.date))



ax = sns.regplot(x='Months',y='Volume BTC',data=monthly_btcusd,scatter_kws={'alpha':0.5},fit_reg=False)

labels = pd.Series(ax.get_xticks()).map(mapping).fillna('')
yearly_btcusd=btc_usd.resample('AS-JAN').mean()

yearly_btcusd.fillna(method='ffill',inplace=True)

print('Yearly BTC/USD')



yearly_btcusd.head()
yearly_btcusd['Volume BTC'].plot()
#2015

btc_usd['Volume BTC']['2015-01-01':'2015-12-31'].plot.hist(alpha=0.5)

#2016

btc_usd['Volume BTC']['2016-01-01':'2016-12-31'].plot.hist(alpha=0.5)

#2017

btc_usd['Volume BTC']['2017-01-01':'2017-12-31'].plot.hist(alpha=0.5)
timeseries=pd.DataFrame(index=monthly_btcusd.index)

timeseries['Volume BTC']=monthly_btcusd['Volume BTC']

timeseries['Datetime']=monthly_btcusd.index

timeseries['Datetime']=timeseries['Datetime'].dt.month





sns.set(style="darkgrid")

mapping=timeseries['2014-01-01':'2014-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='red',label='2014')

timeseries['2015-01-01':'2015-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='green',label='2015',ax=mapping)

timeseries['2016-01-01':'2016-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='blue',label='2017',ax=mapping)

timeseries['2018-01-01':'2018-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='orange', label='2018', ax=mapping)

timeseries['2019-01-01':'2019-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='purple', label='2019', ax=mapping)

mapping.set_xlabel('Months')

mapping.set_ylabel('Volume BTC')

mapping.set_title('BTC/USD Trade 2014 - 2019')

mapping=plt.gcf()

mapping.set_size_inches(10,6)
sns.violinplot(btc_usd['Volume BTC'])
from statsmodels.tsa.stattools import adfuller



score = adfuller(btc_usd['Volume BTC'])



print('Augmented Dickey-Fuller Statistic: %f' % score[0])

print('p-value: %f'%score[1])

print('Critical Values:')

for item, point in score[4].items():

    print('Value at %s = %.2f' % (item, point))