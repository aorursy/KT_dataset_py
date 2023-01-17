import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/cryptocurrency-financial-data/consolidated_coin_data.csv',index_col='Date')

df.index=pd.to_datetime(df.index)

df.dropna()

df= df.drop(['Currency', 'Open', 'High', 'Low', 'Volume', 'Market Cap'],axis=1)

df= df.iloc[1:1000]

df['Close']= pd.to_numeric(df['Close'])

df['Close'].plot(figsize=(12,6))
from statsmodels.tsa.filters.hp_filter import hpfilter

close_cycle, close_trend= hpfilter(df['Close'],lamb=1600)
df['tend']=close_trend

df[['tend','Close']].plot(figsize=(12,6))
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Close'],model='multiplicative')

from pylab import rcParams

rcParams['figure.figsize']=12,6

result.plot();
df['Close_24_month_SMA']=df['Close'].rolling(window=24).mean() #Every 2 years take the mean

df['Close_12_month_SMA']=df['Close'].rolling(window=12).mean() #Every year take the mean

df[['Close_24_month_SMA','Close_12_month_SMA','Close']].plot(figsize=(12,6))
df['Close_24_month_EWMA']=df['Close'].ewm(span=24).mean()

df[['Close_24_month_EWMA','Close']].plot(figsize=(12,6))
idx= df.index

idx.freq='-1D' #Set the frequency to day 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 24

alpha= 2/(span+1)

df['EWMA_24']=df['Close'].ewm(alpha=alpha,adjust=False).mean()
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df['DES_mul_24']= ExponentialSmoothing(df['Close'],trend='mul').fit().fittedvalues.shift(-1)
df[['Close','DES_mul_24','EWMA_24']].plot(figsize=(12,6))
df['TES_mul_12']= ExponentialSmoothing(df['Close'],trend='mul',seasonal='mul',seasonal_periods=24).fit().fittedvalues
df[['Close','DES_mul_24','TES_mul_12']].plot(figsize=(12,6))
df[['Close','DES_mul_24','TES_mul_12']].iloc[:24].plot(figsize=(12,6))
df[['Close','DES_mul_24','TES_mul_12']].iloc[-24:].plot(figsize=(12,6))