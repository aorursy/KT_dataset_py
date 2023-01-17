import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os



sc = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

sc.head()
sgsc = sc.loc[(sc.country=='Singapore') & (sc.age=='15-24 years'), :]



plt.plot('year','suicides/100k pop', data=sgsc.loc[sgsc.sex=='male',:], c='blue')

plt.plot('year','suicides/100k pop', data=sgsc.loc[sgsc.sex=='female',:], c='red')

plt.legend(('Male','Female'))

plt.show()
sns.lineplot(x='year', y='suicides/100k pop', data=sgsc, hue='sex'); # ';' is to avoid extra message before plot
plt.figure(figsize=(10,5)) # Figure size

sns.lineplot(x='year', y='suicides/100k pop', data=sgsc, hue='sex', marker='o') # Specify markers with marker argument

plt.title('Suicide Rate in Singapore Aged 15-24') # Title

plt.xticks(sgsc.year.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()
# print(os.listdir("../input/cryptocurrencypricehistory"))

eth = pd.read_csv('../input/cryptocurrencypricehistory/ethereum_price.csv', parse_dates=['Date'])

eth.set_index('Date', drop=True, inplace=True)

eth.sort_index(inplace=True)

eth.head()
plt.plot(eth.Close);
plt.figure(figsize=(8,6))

plt.plot(eth.asfreq('M').Close, marker='.')

plt.title('Ethereum Monthly Price')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(10,5))

plt.plot(eth['2017':].asfreq('W').Close, marker='.') # eth['2017':] returns a subset of eth since 2017

plt.title('Ethereum Weekly Price Since 2017')

plt.xticks(rotation=90)

plt.show()
kiva = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv', parse_dates=['posted_time'])

kiva_v = kiva.loc[kiva.country=='Vietnam',['posted_time','loan_amount','sector','lender_count']]

kiva_v.set_index('posted_time', inplace=True)

kiva_v.head()
plt.figure(figsize=(9,6))

plt.plot(kiva_v.resample('M').sum()['loan_amount'])

plt.title('Kiva Loan Amount in Vietnam')

plt.xticks(rotation=45) # Rotate 45 degrees

plt.show()
eth.loc[:,'pct_change'] = eth.Close.pct_change()*100

eth.loc['2018':,'pct_change'].plot(kind='bar', color='b')

plt.xticks([])

plt.show()
# Loading bitcoin data

btc = pd.read_csv('../input/cryptocurrencypricehistory/bitcoin_price.csv', parse_dates=['Date'])

btc.set_index('Date', drop=True, inplace=True)

btc.sort_index(inplace=True)

btc.tail()
eth_return = eth['2016-12-31':].Close.pct_change()+1

btc_return = btc['2016-12-31':].Close.pct_change()+1

eth_cum = eth_return.cumprod()

btc_cum = btc_return.cumprod()

plt.figure(figsize=(9,6))

btc_cum.plot(c='blue')

eth_cum.plot(c='cyan')

plt.title('Cumulative Return in Cryptocurrency since 2017')

plt.legend(('Bitcoin','Ethereum'))

plt.yscale('log')

plt.show()