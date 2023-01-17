# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/nifty50-stock-market-data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Importing required modules

import pandas as pd          

import numpy as np               # For mathematical calculations 

import matplotlib.pyplot as plt  # For plotting graphs 

import datetime as dt

from datetime import datetime    # To access datetime 

from pandas import Series        # To work on series 

%matplotlib inline 



import warnings                   # To ignore the warnings 

warnings.filterwarnings("ignore")





# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()



df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/MARUTI.csv")

df.head()
# For the sake of this notebook, I shall limit the number of columns to keep things simple. 



data = df[['Date','Open','High','Low','Close','Volume','VWAP']]

data.info()
# Convert string to datetime64

data['Date'] = data['Date'].apply(pd.to_datetime)

data.head()
from datetime import datetime

my_year = 2019

my_month = 4

my_day = 21

my_hour = 10

my_minute = 5

my_second = 30
test_date = datetime(my_year, my_month, my_day)

test_date

test_date = datetime(my_year, my_month, my_day, my_hour, my_minute, my_second)

print('The day is : ', test_date.day)

print('The hour is : ', test_date.hour)

print('The month is : ', test_date.month)
print(data.index.max())

print(data.index.min())
# Earliest date index location

print('Earliest date index location is: ',data.index.argmin())



# Latest date location

print('Latest date location: ',data.index.argmax())

df_vwap = df[['Date','VWAP']]

df_vwap['Date'] = df_vwap['Date'].apply(pd.to_datetime)

df_vwap.head()
df_vwap['year'] = df_vwap.Date.dt.year

df_vwap['month'] = df_vwap.Date.dt.month

df_vwap['day'] = df_vwap.Date.dt.day

df_vwap['day of week'] = df_vwap.Date.dt.dayofweek



#Set Date column as the index column.

df_vwap.set_index('Date', inplace=True)

df_vwap.head()
# Visualising the VWAP 



plt.figure(figsize=(16,8)) 

plt.plot(df_vwap['VWAP'], label='VWAP') 

plt.title('Time Series') 

plt.xlabel("Time(year)") 

plt.ylabel("Volume Weighted Average Price") 

plt.legend(loc='best')
# Yearly VWAP of Maruti Stocks



df_vwap.groupby('year')['VWAP'].mean().plot.bar()
# Monthly VWAP of Maruti Stocks



df_vwap.groupby('month')['VWAP'].mean().plot.bar()
# Daily VWAP of Maruti Stocks



df_vwap.groupby('day')['VWAP'].mean().plot.bar()
# Analysing w.r.t day of the week



df_vwap.groupby('day of week')['VWAP'].mean().plot.bar()
df_vwap.resample(rule = 'A').mean()[:5]
df_vwap['VWAP'].resample('A').mean().plot(kind='bar',figsize = (10,4))

plt.title('Yearly Mean VWAP for Maruti')

df_vwap['VWAP'].resample('AS').mean().plot(kind='bar',figsize = (10,4))

plt.title('Yearly start Mean VWAP for Maruti')

df_vwap.head()
df_vwap.shift(1).head()
df_vwap.shift(-1).head()
df_vwap.tshift(periods=3, freq = 'M').head()
df_vwap['VWAP'].plot(figsize = (10,6))
df_vwap.rolling(7).mean().head(10)
df_vwap['VWAP'].plot()

df_vwap.rolling(window=30).mean()['VWAP'].plot(figsize=(16, 6))