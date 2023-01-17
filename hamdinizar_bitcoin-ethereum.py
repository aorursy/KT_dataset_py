import pandas as pd

bitcoin_price_1week_Test_Test = pd.read_csv("../input/bitcoin-price-prediction/bitcoin_price_1week_Test - Test.csv")

bitcoin_price_Training_Training = pd.read_csv("../input/bitcoin-price-prediction/bitcoin_price_Training - Training.csv")
import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd
bitcoin = pd.read_csv("../input/bitcoin/BTC-EUR.csv")

bitcoin.head()
bitcoin['Close'].plot(figsize=(12,8))

plt.show()
bitcoin.index
bitcoin = pd.read_csv("../input/bitcoin/BTC-EUR.csv",index_col='Date',parse_dates=True)

bitcoin.head()
bitcoin['Close'].plot(figsize=(12,8))

plt.show()
bitcoin.index
bitcoin['2019']['Close'].plot(figsize=(12,8))
bitcoin['2019/09']['Close'].plot(figsize=(12,8))
bitcoin['2017':'2019']['Close'].plot(figsize=(12,8))
bitcoin.loc['2018':'2019']['Close'].resample('M').plot(figsize=(12,8))

plt.show()
bitcoin.loc['2018':'2019']['Close'].resample('M').mean().plot(figsize=(12,8))

plt.show()
bitcoin.loc['2018':'2019']['Close'].resample('M').mean().plot.bar(figsize=(12,8))

plt.show()
bitcoin.loc['2018':'2019']['Close'].resample('3W').std().plot(figsize=(12,8))

plt.show()
plt.figure(figsize=(12,8))

bitcoin.loc['2019','Close'].plot()

bitcoin.loc['2019','Close'].resample('M').mean().plot(label='moyenne par mois',lw=3,ls='--',alpha=0.8)

bitcoin.loc['2019','Close'].resample('W').mean().plot(label='moyenne par semaine',lw=2,ls=':',alpha=0.8)

plt.legend()

plt.show()
bitcoin.loc['2019','Close'].resample('M').agg(['mean','std','min','max'])
m = bitcoin.loc['2019','Close'].resample('M').agg(['mean','std','min','max'])



plt.figure(figsize=(12,8))

m['mean']['2019'].plot(label='moyenne par mois')



plt.fill_between(m.index,m['min']['2019'],m['max']['2019'],alpha=0.2,label='max-min par mois')

plt.legend()

plt.show()
# Exponential weigthed function : ewf 

plt.figure(figsize=(12,8))

bitcoin['2019/09']['Close'].plot()



for i in np.arange(0.2,1,0.2):

    bitcoin['2019/09']['Close'].ewm(alpha=i).mean().plot(label=f"ewm pour alpha = {i}",ls = ':')

plt.legend()

plt.show()
etherium = pd.read_csv("../input/etherium/ETH-EUR.csv",index_col='Date',parse_dates=True)

etherium.head()
pd.merge(bitcoin,etherium,on='Date',how = 'inner',suffixes = ('_btc','_eth'))
btc_eth = pd.merge(bitcoin,etherium,on='Date',how = 'inner',suffixes = ('_btc','_eth'))
btc_eth[['Close_btc','Close_eth']].plot(subplots=True,figsize=(12,8))

plt.show()
btc_eth[['Close_btc','Close_eth']].corr()