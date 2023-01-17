import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv('../input/crypto-markets.csv', index_col='date', parse_dates=True)
df.head()
bitcoin_price = df[df['name']=='Bitcoin']
bitcoin_price[['open','high','low','close']].plot(figsize=(15,8))
bitcoin = df[df['name']=='Bitcoin'].loc['2017':]
ethereum = df[df['name']=='Ethereum'].loc['2017':]
ripple = df[df['name']=='Ripple'].loc['2017':]
bitcoin_cash = df[df['name']=='Bitcoin Cash'].loc['2017':]
cardano = df[df['name']=='Cardano'].loc['2017':]

plt.figure(figsize=(15,8))
(bitcoin['market']/1000000).plot(color='darkorange', label='Bitcoin')
(ethereum['market']/1000000).plot(color='grey', label='Ethereum')
(ripple['market']/1000000).plot(color='blue', label='Ripple')
(bitcoin_cash['market']/1000000).plot(color='yellow', label='Bitcoin Cash')
(cardano['market']/1000000).plot(color='cyan', label='Cardano')
plt.legend()
plt.title('Top5 Cryptocurrency Market Cap (Million USD)')
plt.show()
plt.figure(figsize=(15,8))
(bitcoin['volume']/1000000).plot(color='darkorange', label='Bitcoin')
(ethereum['volume']/1000000).plot(color='grey', label='Ethereum')
(ripple['volume']/1000000).plot(color='blue', label='Ripple')
(bitcoin_cash['volume']/1000000).plot(color='yellow', label='Bitcoin Cash')
(cardano['volume']/1000000).plot(color='cyan', label='Cardano')
plt.legend()
plt.title('Top5 Cryptocurrency Transactions Volume (Million Units)')
plt.show()
btc = bitcoin[['close']]
btc.columns = ['BTC']

eth = ethereum[['close']]
eth.columns = ['ETH']

xrp = ripple[['close']]
xrp.columns = ['XRP']

bch = bitcoin_cash[['close']]
bch.columns = ['BCH']

ada = cardano[['close']]
ada.columns = ['ADA']

close = pd.concat([btc,eth,xrp,bch,ada], axis=1, join='inner')
close.head()
close = close['10-2017':]
close.plot(figsize=(12,6))
plt.ylabel('price in USD')
plt.title('Historical closing price of top 5 Crypto since Oct 2017')
plt.figure(figsize=(12,6))
sns.heatmap(close.corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Bitcoin and other top 4 Crypto')
returns = close.apply(lambda x: x/x[0])
returns.plot(figsize=(12,6))
plt.ylabel('Return ration')
plt.title('Return of each Cryptocurrencies')
coins = (5000 / close.iloc[0]).round(3)
coins
f = plt.figure(figsize=(12,6))
ax = sns.barplot(['Bitcoin','Ethereum','Ripple','Bitcoin Cash','Cardano'], coins.values)
plt.title('Total Coins you get with $5000 within 2017-10-01')
plt.ylabel('Total Coins')

rects = ax.patches
labels = coins.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
#sell coins on 2018-02-05
earnings = coins * close.tail(1)
earnings = earnings.stack()
earnings.index = earnings.index.droplevel(0)
earnings
f = plt.figure(figsize=(12,6))
ax = sns.barplot(['Bitcoin','Ethereum','Ripple','Bitcoin Cash','Cardano'], earnings.values)
plt.title('Total Earnings you get with $5000 within 2018-02-05')
plt.ylabel('Total Earnings')

rects = ax.patches
labels = earnings.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha='center', va='bottom')
    
plt.show()
