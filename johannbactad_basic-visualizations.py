from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
%matplotlib inline
from pandas import *
from numpy.random import randn
import os
print(os.listdir('../input'))
phstocks = pd.read_csv('../input/conso12282018.csv')
phstocks2 = phstocks
tickersall = ['AEV', 'AP', 'AGI', 'AC', 'ALI', 'BPI', 'BDO', 'DMC', 'FGEN', 'GLO', 'GTCAP', 'ICT', 'JGS', 'JFC', 'LTG', 'MER', 'MEG', 'MPI', 'MBT', 'PCOR', 'TEL', 'PGOLD', 'RLC', 'RRHI', 'SMC', 'SECB', 'SM', 'SMPH', 'SCC', 'URC']
tickers= ['BDO', 'MBT', 'BPI','PNB','SECB','CHIB']
dfBDO = phstocks[phstocks.TICKER == 'BDO']
dfMBT = phstocks[phstocks.TICKER == 'MBT']
dfBPI = phstocks[phstocks.TICKER == 'BPI']
dfPNB = phstocks[phstocks.TICKER == 'PNB']
dfSECB = phstocks[phstocks.TICKER == 'SECB']
dfCHIB = phstocks[phstocks.TICKER == 'CHIB']
dfBDO = pd.pivot_table(dfBDO, index='DATE', columns='TICKER').swaplevel(axis=1)
dfMBT = pd.pivot_table(dfMBT, index='DATE', columns='TICKER').swaplevel(axis=1)
dfBPI = pd.pivot_table(dfBPI, index='DATE', columns='TICKER').swaplevel(axis=1)
dfPNB = pd.pivot_table(dfPNB, index='DATE', columns='TICKER').swaplevel(axis=1)
dfSECB = pd.pivot_table(dfSECB, index='DATE', columns='TICKER').swaplevel(axis=1)
dfCHIB = pd.pivot_table(dfCHIB, index='DATE', columns='TICKER').swaplevel(axis=1)
bank_stocks= pd.concat([dfBDO, dfMBT, dfBPI, dfPNB, dfSECB, dfCHIB], axis=1)
bank_stocks.head(3)
bank_stocks.columns.names= ['Bank Ticker', 'Stock Info']
bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').max()
returns = pd.DataFrame()
for tick in tickers:
    returns[tick+ ' Return']= bank_stocks[tick]['CLOSE'].pct_change()
import seaborn as sns
sns.pairplot(returns[1:])
returns.idxmin()
returns.idxmax()
returns.std()
returns.head(3)
returns.tail(3)
sns.distplot(returns['01/04/2018':'12/28/2018']['BDO Return'], color='green')

sns.distplot(returns['01/04/2018':'12/28/2018']['MBT Return'], color='red')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import plotly
import cufflinks as cf
cf.go_offline()
bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').plot()
bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').iplot()
dfAEV2.head(3)
plt.figure(figsize=(12, 4))
dfBDO['BDO']['CLOSE'].ix['01/04/2018':'12/28/2018'].rolling(window=30).mean().plot(label='30 day avg')
dfMBT['MBT']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfPNB['PNB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfSECB['SECB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
dfCHIB['CHIB']['CLOSE'].ix['01/04/2018':'12/28/2018'].plot(label='AEV')
plt.legend()
sns.heatmap(bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr())
sns.clustermap(bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr())
close_corr= bank_stocks.xs(key='CLOSE', axis=1, level='Stock Info').corr()
close_corr.iplot(kind='heatmap', colorscale='rdylbu')
dfBDO.head(3)
dfBDO['BDO'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')
dfMBT['MBT'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')
dfBPI['BPI'][['OPEN', 'HIGH', 'LOW', 'CLOSE']].iplot(kind='candle')
dfBDO['BDO']['CLOSE'].ta_plot(study='sma', periods=[13, 21, 55])
dfBDO['BDO']['CLOSE'].ta_plot(study='boll')
