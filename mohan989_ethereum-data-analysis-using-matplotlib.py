import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from matplotlib import style

style.use('fivethirtyeight')

%matplotlib inline
data = pd.read_csv('../input/all_data.csv')

data.info()

print('-'*90)

print(data.head())
data = data.drop(['total_eth_growth'],axis=1)

dateconv = np.vectorize(dt.datetime.fromtimestamp)

data['timestamp'] = dateconv(data['timestamp'])

print(data.head())
plt.plot(data['timestamp'],data['price_USD'])

plt.xlabel('Date')

plt.ylabel('Price in USD')

plt.show()
plt.plot(data['timestamp'],data['blocksize'])

plt.xlabel('Date')

plt.ylabel('blocksize')

plt.show()
plt.plot(data['timestamp'],data['hashrate'])

plt.xlabel('Date')

plt.ylabel('hashrate')

plt.show()
plt.plot(data['timestamp'],data['total_addresses'])

plt.xlabel('Date')

plt.ylabel('total_addresses')

plt.show()
plt.plot(data['timestamp'],data['transactions'])

plt.xlabel('Date')

plt.ylabel('transactions')

plt.show()
ax1 = plt.subplot2grid((3,1),(0,0),rowspan=1,colspan=1)

ax2 = plt.subplot2grid((3,1),(1,0),rowspan=1,colspan=1)

ax3 = plt.subplot2grid((3,1),(2,0),rowspan=1,colspan=1)
fig= plt.figure()

ax1= fig.add_subplot(311)

ax2= fig.add_subplot(312)

ax3= fig.add_subplot(313)
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex= True) 
ax1.plot(data['timestamp'],data['price_USD'])

ax2.plot(data['timestamp'],data['total_addresses'])

ax3.plot(data['timestamp'],data['blocksize'])

plt.show()
fig2,(ax1,ax2,ax3) = plt.subplots(3,1,sharex= True) 

ax1.plot(data['timestamp'],data['hashrate'])

ax2.plot(data['timestamp'],data['transactions'])

ax3.plot(data['timestamp'],data['market-cap-value'])

plt.show()