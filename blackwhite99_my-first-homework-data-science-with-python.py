# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime, pytz
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input/'))

# Any results you write to the current directory are saved as output.
#I'm apply coinbaseUSD for 60 min
#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))

# I am reading this database with  read_csv method in pandas:
data=pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv', parse_dates=[0], date_parser=dateparse)

print("Index of the Table")
data.info()
# I am changing name of columns
data.rename(columns={'Volume_(BTC)':'Volume_BTC', 'Volume_(Currency)':'Volume_USD'}, inplace= True)
data['Volume_BTC'].fillna(value=0, inplace=True)
data['Volume_USD'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)
#print(data["Timestamp"][1530057600])
#print("....")
data.tail()
data2=data[1260000:]
print(data2)
#

months=np.unique(data2['Timestamp'].dt.month)
print(months)
meanOpen=[]
meanClose=[]
meanHigh=[]
meanLow=[]
sumUSDVolume=[]
sumBTCVolume=[]
for month in months:
    meanOpen.append(data2[(data2['Timestamp'].dt.month)==month]['Open'].mean())
    meanClose.append(data2[(data2['Timestamp'].dt.month)==month]['Close'].mean())
    meanHigh.append(data2[(data2['Timestamp'].dt.month)==month]['High'].mean())
    meanLow.append(data2[(data2['Timestamp'].dt.month)==month]['Low'].mean())
    sumUSDVolume.append(data2[(data2['Timestamp'].dt.month)==month]['Volume_USD'].sum())
    sumBTCVolume.append(data2[(data2['Timestamp'].dt.month)==month]['Volume_BTC'].sum())
print(meanClose)

print("I am importing matplotlib and seaborn for plotting data")
import matplotlib.pyplot as plt
import seaborn as snd
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  

plt.subplot(2,1,1)
plt.plot(months, meanClose, color = 'r', label = 'BTC Price Close')
plt.ylabel('BTC Price (USD)', fontsize=40)
plt.legend(loc=2, prop={'size':25})
plt.subplot(2,1,2)
plt.plot(months,sumBTCVolume, color='b', label='Volume BTC')
plt.ylabel('Volume (BTC)', fontsize=40)
#x=dataTrain.index
#labels=dataTrain.index
#plt.xticks(x, labels, rotation='vertical')
plt.xlabel('Time ', fontsize=40)
#plt.title("BTC Price and Volume Figure", fontsize=40) # not runned
plt.legend(loc=2, prop={'size':25})
plt.show()
# I converted Timestamp columns to Date columns
data['Date']=pd.to_datetime(data['Timestamp'], unit='s').dt.date
# and I groupped for Date columns
group=data.groupby('Date')
# I got the average on Weighted Price columns for groupped Date columns
meanWeightPrice = group['Weighted_Price'].mean()
sumVolumeBTC = group['Volume_BTC'].sum()
# I will obtain 60 day test data.
predictionDays=60
dataTrainPrice = meanWeightPrice[:len(meanWeightPrice)-predictionDays]
dataTestPrice = meanWeightPrice[len(meanWeightPrice)-predictionDays:]
#dataTrainPrice.tail(10)
dataTrainVolume = sumVolumeBTC[:len(sumVolumeBTC)-predictionDays]
dataTestVolume = sumVolumeBTC[len(sumVolumeBTC)-predictionDays:]


# Train dataset converted to DataFrame
dfTrainPrice=pd.DataFrame(dataTrainPrice)
dfTrainVolume=pd.DataFrame(dataTrainVolume)
dataTrain=pd.merge(dfTrainPrice, dfTrainVolume, on='Date')
dataTrain.tail()
# Test dataset converted to DataFrame
dfTestPrice=pd.DataFrame(dataTestPrice)
dfTestVolume=pd.DataFrame(dataTestVolume)
dataTest=pd.merge(dfTestPrice, dfTestVolume, on='Date')
dataTest.info()
# is corelation map?
#f,ax=plt.subplots(figsize=(25,25))
#snd.heatmap(dataTrain.corr(), annot= True, linewidths=0.5, fmt='.2f', ax=ax)
#snd.heatmap(dataTest.corr(), annot= True, linewidths=0.5, fmt='.2f', ax=ax)
print(dataTrain.corr())
print(dataTest.corr())

print(dataTrain.info())
# I did the price and volume graph for train dataset.
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  

plt.subplot(2,1,1)
plt.plot(dataTrain['Weighted_Price'], color = 'r', label = 'BTC Price')
plt.ylabel('BTC Price (USD)', fontsize=40)
plt.legend(loc=2, prop={'size':25})
plt.subplot(2,1,2)
plt.plot(dataTrain['Volume_BTC'], color='b', label='Volume BTC')
plt.ylabel('Volume (BTC)', fontsize=40)
#x=dataTrain.index
#labels=dataTrain.index
#plt.xticks(x, labels, rotation='vertical')
plt.xlabel('Time ', fontsize=40)
#plt.title("BTC Price and Volume Figure", fontsize=40) # not runned
plt.legend(loc=2, prop={'size':25})
plt.show()
# I did the price and volume graph for test dataset.
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  

plt.subplot(2,1,1)
plt.plot(dataTest['Weighted_Price'], color = 'r', label = 'BTC Price')
plt.ylabel('BTC Price (USD)', fontsize=40)
plt.legend(loc=2, prop={'size':25})
plt.subplot(2,1,2)
plt.plot(dataTest['Volume_BTC'], color='b', label='Volume BTC')
plt.ylabel('Volume (BTC)', fontsize=40)
x=dataTest.index
labels=dataTest.index
plt.xticks(x, labels, rotation='vertical')
plt.xlabel('Time ', fontsize=40)
#plt.title("BTC Price and Volume Figure", fontsize=40) # not runned
plt.legend(loc=2, prop={'size':25})
plt.show()
# candlestick graph
#import matplotlib.finance
btcOHLC=data.groupby(['Date'])['Open', 'High', 'Low', 'Close'].mean()
btcOHLC=btcOHLC[1180:]
btcOHLC.info()
#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))
