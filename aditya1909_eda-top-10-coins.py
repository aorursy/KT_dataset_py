import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="white", color_codes=True)

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



print(__version__)

import cufflinks as cf

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
crypto = pd.read_csv('CryptocoinsHistoricalPrices.csv',index_col=0)

crypto.head()
#Deleting the NA values at the end



crypto.dropna(axis=0, inplace = True)
#extracting month

month = crypto['Date'].values

month = [my_str.split("-")[1] for my_str in month]

crypto["Month"] = month
#extracting year

year = crypto['Date'].values

year = [my_str.split("-")[0] for my_str in year]

crypto["Year"] = year
crypto.head()
crypto.tail()
cleaned_crypto = crypto[(crypto['Market.Cap']!='-') & (crypto['Volume']!='-')]
cleaned_crypto.head()
#Removing Commas from Market.Cap

cleaned_crypto['Market.Cap'] = cleaned_crypto['Market.Cap'].str.replace(',','')
#Removing Commas from Volume

cleaned_crypto['Volume'] = cleaned_crypto['Volume'].str.replace(',','')
cleaned_crypto.head()
#Volume and Market Cap are in Str, we need to change it to float

cleaned_crypto['Market.Cap'] = cleaned_crypto['Market.Cap'].astype('float')

cleaned_crypto['Volume'] = cleaned_crypto['Volume'].astype('float')
# Checking the type again

type(cleaned_crypto['Volume'][1])
# Checking number of records of each coin

cleaned_crypto['coin'].value_counts()
# We are taking data of only 2017 as cryptocurrency was hyped in that year

# Chunking the data according to the year 2017



cleaned_crypto2017 = cleaned_crypto[cleaned_crypto['Year']=='2017']

cleaned_crypto2017.info()
#Checking the record of each coin in 2017

cleaned_crypto2017['coin'].value_counts()
# Checking top Crypto Currency in the year 2017

#cleaned_crypto2017.groupby('coin').sort_values(by='Market.Cap',ascending=True)

#cleaned_crypto2017.sort_values(by='Market.Cap', ascending=True).groupby('coin').head(1000)#

cleaned_crypto2017.groupby('coin').count().sort_values(by='Market.Cap', ascending=False).head(1000)

#Sorting values by Market Cap

# We took 10% of the top market values to see the boosted currencies

top25000 = cleaned_crypto2017.sort_values('Market.Cap', ascending = False).head(25000)

top25000.groupby('coin').count().sort_values('Market.Cap', ascending=False).head(10)
# Checking top currency

top10 = top25000.groupby('coin').count().sort_values('Market.Cap', ascending=False).head(10)
#Checking Index of top 10 coins

top_coins = top10.index

top_coins
cryptoData = top25000[top25000['coin'].isin(top_coins)]
cryptoData.head()
AverageVolume =  pd.DataFrame(cryptoData.groupby('coin').mean()['Volume'])

AverageVolume



# Changing the index to column



AverageVolume = AverageVolume.reset_index

AverageVolume
plt.figure(figsize=(10,6))

sns.barplot(x='coin',y='Volume',data=cryptoData,estimator=np.mean)
# This plot shows that people are moving to BTC, XRP(Ripple), ETH(Ethereum), LTC( Lite Coin) and ETC( Ethereum Classic)as well
#Creating a Dataset for BTC 

cryptoData_BTC = cryptoData[cryptoData['coin']=='BTC']
import plotly.graph_objs as go

data = [go.Scatter(x=cryptoData_BTC['Month'], y=cryptoData_BTC['Open'], name='Open')]

cryptoData_BTC.iplot(data=data)
cryptoData_BTC_Open = pd.concat([cryptoData_BTC['Open'], cryptoData_BTC['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_BTC_Open = pd.DataFrame(cryptoData_BTC['Open'], cryptoData_BTC['Date'])
cryptoData_BTC_Open.head()
cryptoData_BTC_Open.sort_values('Date', ascending=True, inplace=True)
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



print(__version__)

import cufflinks as cf

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
cryptoData_BTC_Open.head()
data = [go.Scatter(x=cryptoData_BTC_Open['Date'], y=cryptoData_BTC_Open['Open'], name='Open')]

cryptoData_BTC_Open.iplot(data=data)
cryptoData_BTC_Open.plot(x='Date', y='Open', kind='Area')
#Creating a Dataset for ETH

cryptoData_ETH = cryptoData[cryptoData['coin']=='ETH']

cryptoData_ETH_Open = pd.concat([cryptoData_ETH['Open'], cryptoData_ETH['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_BTC_Open = pd.DataFrame(cryptoData_BTC['Open'], cryptoData_BTC['Date'])

cryptoData_ETH_Open.sort_values('Date', ascending=True, inplace=True)

cryptoData_ETH_Open.plot(x='Date', y='Open', kind='Area')
#Creating a Dataset for XRP

cryptoData_XRP = cryptoData[cryptoData['coin']=='XRP']

cryptoData_XRP_Open = pd.concat([cryptoData_XRP['Open'], cryptoData_XRP['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_XRP_Open = pd.DataFrame(cryptoData_XRP['Open'], cryptoData_XRP['Date'])

cryptoData_XRP_Open.sort_values('Date', ascending=True, inplace=True)

cryptoData_XRP_Open.plot(x='Date', y='Open', kind='Area')
#Creating a Dataset for LTC

cryptoData_LTC = cryptoData[cryptoData['coin']=='LTC']

cryptoData_LTC_Open = pd.concat([cryptoData_LTC['Open'], cryptoData_LTC['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_LTC_Open = pd.DataFrame(cryptoData_LTC['Open'], cryptoData_LTC['Date'])

cryptoData_LTC_Open.sort_values('Date', ascending=True, inplace=True)

cryptoData_LTC_Open.plot(x='Date', y='Open', kind='Area')
#Creating a Dataset for ETC

cryptoData_ETC = cryptoData[cryptoData['coin']=='ETC']

cryptoData_ETC_Open = pd.concat([cryptoData_ETC['Open'], cryptoData_ETC['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_ETC_Open = pd.DataFrame(cryptoData_ETC['Open'], cryptoData_ETC['Date'])

cryptoData_ETC_Open.sort_values('Date', ascending=True, inplace=True)

cryptoData_ETC_Open.plot(x='Date', y='Open', kind='Area')
#Creating a Dataset for DASH

cryptoData_DASH = cryptoData[cryptoData['coin']=='DASH']

cryptoData_DASH_Open = pd.concat([cryptoData_DASH['Open'], cryptoData_DASH['Date']], axis=1, keys=['Open', 'Date'])

#cryptoData_DASH_Open = pd.DataFrame(cryptoData_DASH['Open'], cryptoData_DASH['Date'])

cryptoData_DASH_Open.sort_values('Date', ascending=True, inplace=True)

cryptoData_DASH_Open.plot(x='Date', y='Open', kind='Area')
cryptoData_BTC_Delta = pd.concat([cryptoData_BTC['Delta'], cryptoData_BTC['Date']], axis=1, keys=['Delta', 'Date'])

#cryptoData_DASH_Open = pd.DataFrame(cryptoData_DASH['Open'], cryptoData_DASH['Date'])

cryptoData_BTC_Delta.sort_values('Date', ascending=True, inplace=True)

cryptoData_BTC_Delta.plot(x='Date', y='Delta')
cryptoData_ETH = cryptoData[cryptoData['coin']=='ETH']

cryptoData_ETH_Delta = pd.concat([cryptoData_ETH['Delta'], cryptoData_ETH['Date']], axis=1, keys=['Delta', 'Date'])

cryptoData_ETH_Delta.sort_values('Date', ascending=True, inplace=True)

cryptoData_ETH_Delta.plot(x='Date', y='Delta')
cryptoData_XRP = cryptoData[cryptoData['coin']=='XRP']

cryptoData_XRP_Delta = pd.concat([cryptoData_XRP['Delta'], cryptoData_XRP['Date']], axis=1, keys=['Delta', 'Date'])

cryptoData_XRP_Delta.sort_values('Date', ascending=True, inplace=True)

cryptoData_XRP_Delta.plot(x='Date', y='Delta')
cryptoData_LTC = cryptoData[cryptoData['coin']=='LTC']

cryptoData_LTC_Delta = pd.concat([cryptoData_LTC['Delta'], cryptoData_LTC['Date']], axis=1, keys=['Delta', 'Date'])

cryptoData_LTC_Delta.sort_values('Date', ascending=True, inplace=True)

cryptoData_LTC_Delta.plot(x='Date', y='Delta')
cryptoData_ETC = cryptoData[cryptoData['coin']=='ETC']

cryptoData_ETC_Delta = pd.concat([cryptoData_ETC['Delta'], cryptoData_ETC['Date']], axis=1, keys=['Delta', 'Date'])

cryptoData_ETC_Delta.sort_values('Date', ascending=True, inplace=True)

cryptoData_ETC_Delta.plot(x='Date', y='Delta')
# From the above Visualizations, it is clear that in the last year BTC, XRP, ETH, LTC have burgeoned. The transformation was a 

#sudden one occurend in the second half of the year
#From the Delta Plotting, It is understood that BTC has fluctuated the most and is not in a stable condition to invest.

#Ripple went high for the one period of time, but has not been fluctuating much after that. Ripple seems to be a promising cryptocurrency to 

#invest in
