# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output

# Try prophet as seen in https://www.kaggle.com/samuelbelko/predicting-prices-of-avocados
from fbprophet import Prophet
# Not sure what this does - show graphs in line?
%matplotlib inline


# From the above, I have the data for this dataset. Read it in, and do some simple examinations
data = pd.read_csv('../input/Jan20_NDXT.csv')

# First 10 records
data.head()




# Info on the columns - For some reason if I do this in the above, I only see output for one of them so 
# I do separately
data.info()

#Desc
# Not alot of data, but we are just playing around for now, let's look at max/min and if there are any 
# null values
data.describe()
data['DailyDif'] = data['NDXT Close'] - data['NDXT Open']
data['DailyRange'] = data['NDXT High'] - data['NDXT Low']
data.drop(columns=['NDXT Volume','NDXT Close','NDXT High','NDXT Low','NDXT Adj Close'], inplace=True)
data.head()


#I want to convert the date to something Pandas likes
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.head()
 #Plot theh open price versus time
data.plot(x='Date', y='NDXT Open', kind="line")
#Next I want to create a column called month. So I tried this
# data['Month'] = data['Date'].month
# But that didn't work, even though I looked up python datetime and it supports that property
# The error I got said that 'object' did not have the property 'month' so i guess i need some kind of
# cast. So i try
# data['Data'].astype(datetime) but that doesn't work, it says it doesn't recognize datetime
# Finally I use data.info() to dump the columns and their types and i see that Date is of type
# datetime64. But no matter what I did, I couldn't get it ti work. Finally I found some example where
# I see someone refer to a dt property. Now it works - see below
data.info()


# Use dt property to access the actual date time properties
data['Month'] = data['Date'].dt.month
data.head()
pdata = data[['Date', 'NDXT Open']].reset_index(drop=True)
pdata = pdata.rename(columns={'Date':'ds', 'NDXT Open':'y'})
pdata.head()

m = Prophet()
m.fit(pdata)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
pdata = pdata[pdata.ds.dt.year < 2017]
pdata.tail()
m = Prophet()
m.fit(pdata)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
data.describe(include='all')
#I can see that I have unique dates (good) but not values for every row. So let's look at those.



data.loc[data['NDXT Open'].isnull()]
data = data.loc[data['NDXT Open'].isnull()==False]
data.describe(include='all')