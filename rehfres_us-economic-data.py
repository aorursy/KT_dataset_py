import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dfFedRate = pd.read_csv('../input/economic-data-s/Fed Funds Rate.csv', skiprows=2).drop([0, 1])

dfFedRate = dfFedRate[['DATE', 'TARGET RATE/RANGE\n(PERCENT)']]

dfFedRate = dfFedRate.dropna()

dfFedRate = dfFedRate.rename(columns={'TARGET RATE/RANGE\n(PERCENT)': 'Fed Interest Rate'})

dfFedRate['Fed Interest Rate'] = dfFedRate['Fed Interest Rate'].str.split('-').str[-1].astype('float64')
dfFedRate.head(10)
dfGold = pd.read_csv('../input/economic-data-s/Gold Futures Historical Data.csv')

pd.to_datetime(dfGold['Date'])

dfGold['Price'] = dfGold['Price'].str.replace(',','').astype('float64')
dfGold.head(10)
dfBanking = pd.read_csv('../input/economic-data-s/KBW Regional Banking Historical Data.csv')

pd.to_datetime(dfBanking['Date'])
dfBanking.head(10)
dfDow = pd.read_csv('../input/economic-data-s/Dow Jones Industrial Average Historical Data.csv')

pd.to_datetime(dfDow['Date'])

dfDow['Price'] = dfDow['Price'].str.replace(',','').astype('float64')
dfDow.head(10)
dfUSDIndex = pd.read_csv('../input/economic-data-s/US Dollar Index Futures Historical Data.csv')

pd.to_datetime(dfUSDIndex['Date'])
dfDow.head(10)
from sklearn.preprocessing import MinMaxScaler 



scaler = MinMaxScaler() 

dfFedRate['Fed Interest Rate'] = scaler.fit_transform(dfFedRate['Fed Interest Rate'].values.reshape(-1, 1))

dfGold['Price'] = scaler.fit_transform(dfGold['Price'].values.reshape(-1, 1))

dfBanking['Price'] = scaler.fit_transform(dfBanking['Price'].values.reshape(-1, 1))

dfDow['Price'] = scaler.fit_transform(dfDow['Price'].values.reshape(-1, 1))

dfUSDIndex['Price'] = scaler.fit_transform(dfUSDIndex['Price'].values.reshape(-1, 1))



fig, ax = plt.subplots()

ax.invert_xaxis()



dfFedRate.plot(ax=ax, x="DATE", y="Fed Interest Rate", figsize=(20,5), label='Fed Interest Rate')

dfGold.plot(ax=ax, x="Date", y="Price", figsize=(20,5), label='Gold Price')

dfBanking.plot(ax=ax, x="Date", y="Price", figsize=(20,5), label='Banking Index')

dfDow.plot(ax=ax, x="Date", y="Price", figsize=(20,5), label='Dow Index')

dfUSDIndex.plot(ax=ax, x="Date", y="Price", figsize=(20,5), label='USD Index')