# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


#Change Time Type.





import datetime, pytz

#define a conversion function for the native timestamps in the csv file

def dateparse (time_in_secs):    

    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))





data = pd.read_csv('/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', parse_dates=[0], date_parser=dateparse)


#Change NaN Values.



# First thing is to fix the data for bars/candles where there are no trades. 

# Volume/trades are a single event so fill na's with zeroes for relevant fields...

data['Volume_(BTC)'].fillna(value=0, inplace=True)

data['Volume_(Currency)'].fillna(value=0, inplace=True)

data['Weighted_Price'].fillna(value=0, inplace=True)



# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so

# lets fill forwards those values...

data['Open'].fillna(method='ffill', inplace=True)

data['High'].fillna(method='ffill', inplace=True)

data['Low'].fillna(method='ffill', inplace=True)

data['Close'].fillna(method='ffill', inplace=True)
data.head()
data.tail()
data.info()
data.describe()
data.corr()
f,ax= plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),linewidths=.5,annot=True,fmt='.4f',ax=ax)

plt.show()
data.columns
fig, ax = plt.subplots(figsize=(20, 10))

data.Open.plot(kind = 'line', color = 'b',label = 'Open',linewidth=5,alpha = 1,grid = True,linestyle = ':')

data.Close.plot(kind='line',color = 'r',label = 'Close',linewidth=5, alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper left')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

data["Volume_(BTC)"].plot(kind = 'line', color = 'b',label = 'Volume_(BTC)',linewidth=1.5,alpha = .5,grid = True,linestyle = ':')

data["Volume_(Currency)"].plot(kind='line',color = 'r',label = 'Volume_(Currency)',linewidth=1.5, alpha = 0.5,grid = True,linestyle = '-')



plt.legend(loc='upper left')  

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')           

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

data["Volume_(BTC)"].plot(kind = 'line', color = 'b',label = 'Volume_(BTC)',linewidth=3,alpha = .5,grid = True,linestyle = ':')

data["Weighted_Price"].plot(kind='line',color = 'r',label = 'Weighted_Price',linewidth=3, alpha = 0.5,grid = True,linestyle = '-')



plt.legend(loc='upper left')    

plt.xlabel('x axis')            

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
fig, ax = plt.subplots(figsize=(18, 10))

plt.scatter(data.Open,data.Close,color='red',alpha=.5)

plt.title('Open-Close Scatter Plot')
fig, ax = plt.subplots(figsize=(18, 10))

plt.scatter(data.High,data.Low,color='red',alpha=.5)

plt.title('High-Low Scatter Plot')
data.Open.plot(kind='hist',figsize=(10,5))

plt.show()

data.Close.plot(kind='hist',figsize=(10,5))

plt.show()
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row

melted = pd.melt(frame=conc_data_row,id_vars = 'Timestamp', value_vars= ['Open','High','Low','Close'])

melted
melted.pivot(index = 'Timestamp', columns = 'variable',values='value')
data3 = data['Open'].head()

data4= data['Close'].head()

conc_data_col2 = pd.concat([data3,data4],axis =1) # axis = 0 : adds dataframes in row

conc_data_col2