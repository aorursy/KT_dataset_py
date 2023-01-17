# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
AAPL = pd.read_csv("/kaggle/input/appl-stock/AAPL.csv")
AAPL
plt.figure(figsize=(19,8))
plt.plot(AAPL['Adj Close'],label='AAPL',linewidth=2)
plt.xticks(rotation=45)
plt.title('Apple Adj Close Price (252 days)')
plt.xlabel('Date')
plt.ylabel('Adj Close Price ($)')
plt.legend(loc='upper left')
plt.show()
SMA30=pd.DataFrame()
SMA30['Adj Close']=AAPL['Adj Close'].rolling(window=30).mean()
SMA30
SMA100=pd.DataFrame()
SMA100['Adj Close']=AAPL['Adj Close'].rolling(window=100).mean()
SMA100
plt.figure(figsize=(19,8))
plt.plot(AAPL['Adj Close'],label='AAPL',linewidth=2)
plt.plot(SMA30['Adj Close'],label='SMA30',linewidth=2)
plt.plot(SMA100['Adj Close'],label='SMA100',linewidth=2)
plt.xticks(rotation=45)
plt.title('Apple Adj Close Price (252 days)')
plt.xlabel('Date')
plt.ylabel('Adj Close Price ($)')
plt.legend(loc='upper left')
plt.show()
data=pd.DataFrame()
data['AAPL']=AAPL['Adj Close']
data['SMA30']=SMA30['Adj Close']
data['SMA100']=SMA100['Adj Close']
data
def buy_sell(data):
    buy=[]
    sell=[]
    flag=-1

    for i in range(len(data)):
        if data['SMA30'][i]>data['SMA100'][i]:
            if flag!=1:
                buy.append(data['AAPL'][i])
                sell.append(np.nan)
                flag=1
            else:
                buy.append(np.nan)
                sell.append(np.nan)

        elif data['SMA30'][i]<data['SMA100'][i]:
            if flag!=0:
                buy.append(np.nan)
                sell.append(data['AAPL'][i])
                flag=0
            else:
                buy.append(np.nan)
                sell.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)
            
    return (buy, sell)  
a=buy_sell(data)
data['Buy Signal Price']=a[0]
data['Sell Signal Price']=a[1]
data
plt.figure(figsize=(19,8))
plt.plot(data['AAPL'],label='AAPL',linewidth=2, alpha=0.35)
plt.plot(data['SMA30'],label='SMA30',linewidth=2, alpha=0.35)
plt.plot(data['SMA100'],label='SMA100',linewidth=2, alpha=0.35)
plt.scatter(data.index, data['Buy Signal Price'], s=100, label='Buy', marker='^', color='green')
plt.scatter(data.index, data['Sell Signal Price'], s=100, label='Sell', marker='v', color='red')
plt.title('Apple Adj Close Price & Buy/Sell Signal (252 days)')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Adj Close Price ($)')
plt.legend(loc='upper left')
plt.show()
