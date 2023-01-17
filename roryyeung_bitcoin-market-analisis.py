#Start with importing packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt #Plotting

import seaborn as sns #Plotting but better looking
#Read the data

df = pd.read_csv('../input/bitcoin-data-from-2014-to-2020/BTC-USD.csv')

#Convert date to timeseries

df['DateTime'] =  pd.to_datetime(df['Date'])

#Display head

df.head()
#Initial time data plot

plt.pyplot.figure(figsize=(15, 6))

sns.lineplot(data=df, x='DateTime', y='Close')

plt.pyplot.xlabel('Year')

plt.pyplot.ylabel('Bitcoin Closing Price (USD)')