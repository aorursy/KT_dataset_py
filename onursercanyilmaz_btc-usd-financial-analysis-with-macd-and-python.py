#Description : Program uses the Moving Average Convergence and Divergence (MACD) crossover

#to dtermine when to buy or sell the stock.

import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas_datareader as web

#Get the data

BTC = web.DataReader('BTCUSD=X', data_source='yahoo', start='2020-01-10')

#Show the data

BTC
#Visualization the graph of BTC

plt.figure(figsize=(12.2,5.5))

plt.xticks(rotation=45)

plt.plot(BTC.index, BTC['Close'])

plt.title('BTC/USD Close Price')

plt.xlabel('Date')

plt.ylabel('Close Price')

plt.show()
#Calculate the MACF and signal line indicator

#Calculate the short term exponential moving average (EMA)

ShortEMA = BTC.Close.ewm(span=12, adjust=False).mean()

#Calculate the long term exponential moving average (EMA)

LongEMA = BTC.Close.ewm(span=26, adjust=False).mean()

#Calculate the MACD line

MACD = ShortEMA - LongEMA

#Calculate the signal line

signal = MACD.ewm(span=9, adjust=False).mean()       
plt.figure(figsize=(12.2,4.5))



plt.plot(BTC.index, MACD, label ='BTCUSD MACD', color='red')

plt.plot(BTC.index, signal, label='Signal Line', color='blue')

plt.xticks(rotation=45)

plt.legend(loc='upper left')

plt.show()

#Create new columns for the data

BTC['MACD'] = MACD

BTC['Signal_Line'] = signal

BTC
#Create a function to signal when to buy and sell 

def buy_sell(signal):

    Buy=[]

    Sell=[]

    flag=-1

    

    for i in range(0,len(signal)):

        if signal['MACD'][i]>signal['Signal_Line'][i]:

            Sell.append(np.nan)

            if flag!= 1:

                Buy.append(signal['Close'][i])

                flag = 1

            else:

                Buy.append(np.nan)

        elif signal['MACD'][i] < signal['Signal_Line'][i]:

            Buy.append(np.nan)

            if flag != 0:

                Sell.append(signal['Close'][i])

                flag = 0

            else:

                Sell.append(np.nan)

        else:

            Buy.append(np.nan)

            Sell.append(np.nan)

    

    return(Buy,Sell)

    

        
#Create buy and sell column

a = buy_sell(BTC)

BTC['Buy_Signal_Price'] = a[0]

BTC['Sell_Signal_Price'] = a[1]
#Show the data

BTC
#Visually show the stock buy and sell signals

plt.figure(figsize=(12.2,4.5))

plt.scatter(BTC.index, BTC['Buy_Signal_Price'], color='green', label='BUY', marker='^', alpha=1)

plt.scatter(BTC.index, BTC['Sell_Signal_Price'], color='red', label='SELL', marker='v', alpha=1)

plt.plot(BTC['Close'], label='Close Price', alpha=0.35)

plt.title('Close Price Buy and Sell Signals')

plt.xlabel('Date')

plt.ylabel('Close Price USD ($)')

plt.legend(loc='upper left')

plt.show()
