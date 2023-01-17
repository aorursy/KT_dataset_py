import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df = pd.read_csv("/kaggle/input/AAPL.csv")
df
df=df.set_index(pd.DatetimeIndex(df['Date'].values))
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(19,8))
plt.plot(df['Close'],label='Close',linewidth=2)
plt.xticks(rotation=45)
plt.title('Apple Close Price (252 days)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()
shortEMA=df.Close.ewm(span=12, adjust=False).mean()
longEMA=df.Close.ewm(span=26, adjust=False).mean()
MACD=shortEMA-longEMA
signal=MACD.ewm(span=9,adjust=False).mean()
shortEMA
longEMA
MACD
df.index
plt.figure(figsize=(19,8))
plt.plot(df.index, MACD, label='AAPL MACD',color='red',linewidth=2)
plt.plot(df.index, signal, label='Signal Line',color='blue',linewidth=2)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.title('Date vs MACD-Signal Line')
plt.xlabel('Date')
plt.ylabel('MACD-Signal Line')
plt.show()
df['MACD']=MACD
df['Signal']=signal
df
def buy_sell(signal):
  buy=[]
  sell=[]
  flag=-1

  for i in range(0,len(signal)):
    if signal['MACD'][i]>signal['Signal'][i]:
      sell.append(np.nan)
      if flag!=1:
        buy.append(signal['Close'][i])
        flag=1
      else:
        buy.append(np.nan)

    elif signal['MACD'][i]<signal['Signal'][i]:
      buy.append(np.nan)
      if flag!=0:
        sell.append(signal['Close'][i])
        flag=0
      else:
        sell.append(np.nan)

    else:
      buy.append(np.nan)
      sell.append(np.nan)

  return (buy, sell)
a=buy_sell(df)
df['Buy Signal Price']=a[0]
df['Sell Signal Price']=a[1]
df
plt.figure(figsize=(19,8))
plt.scatter(df.index,df['Buy Signal Price'], color='green', label='Buy', marker='^', alpha=1)
plt.scatter(df.index,df['Sell Signal Price'], color='red', label='Sell', marker='v', alpha=1)
plt.plot(df.Close, label='Close Price', alpha=0.35)
plt.title('Buy vs Sell Signals')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.xticks(rotation=45)
plt.legend(loc='upper left')

plt.show()
plt.figure(figsize=(19,8))
plt.plot(df.index, MACD, label='AAPL MACD',color='red',linewidth=2)
plt.plot(df.index, signal, label='Signal Line',color='blue',linewidth=2)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.title('Date vs MACD-Signal Line')
plt.xlabel('Date')
plt.ylabel('MACD-Signal Line')
plt.show()
