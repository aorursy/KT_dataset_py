# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cryptoData=pd.read_csv("../input/crypto-markets.csv")
cryptoData.info()
cryptoData.describe()
cryptoData.head(10)
# Function with *args
# Symbol: the crypto currency symbol, Args: columns 
# Filter data according to the symbol and select columns
def GetCryptoCurrencyData(symbol,*args):
    filteredData=cryptoData[cryptoData.symbol==symbol]
    colList=[]
    if len(args)>0:
        for i in args:
            colList.append(i)
        filteredData=filteredData[colList]
    return filteredData

simpleData=GetCryptoCurrencyData('BTC','name','symbol','open','close','date','spread')
simpleData.head(10)
# Nested function
# Calculate daily change rate within a nested function

def CalculateExchangeRate(row):
    def Calculate(opn,close):
        return (close-opn)/opn*100
    return Calculate(row["open"],row["close"])

simpleData["changerate"]=simpleData.apply(CalculateExchangeRate, axis=1)
simpleData.head(10)
#  List Comprehension
#Spread is the $USD difference between the high and low values for the day.
avgSpread=sum(simpleData.spread)/len(simpleData.spread)
print("Average Spread: {0}".format(avgSpread))
simpleData["spreadstatus"]=["above" if sp>avgSpread else "below" if sp<avgSpread else "equal"for sp in simpleData.spread]
simpleData.head(10)
# Function with **kwargs
# Kwargs contains filter conditions
def GetCryptoCurrencyData(**kwargs):
    filteredData=cryptoData
    for key,value in kwargs.items():
        filteredData=filteredData[filteredData[key]==value]
    return filteredData

simpleData=GetCryptoCurrencyData(symbol='BTC',date='2018-05-30')
simpleData.head(5)