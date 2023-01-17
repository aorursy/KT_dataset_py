#!pip3 install matplotlib
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statsmodels.api as sm
uber_raw_apr14 =pd.read_csv('../input/uber-raw-data-apr14.csv')

uber_raw_may14 = pd.read_csv("../input/uber-raw-data-may14.csv")

uber_raw_jun14 = pd.read_csv("../input/uber-raw-data-jun14.csv")

uber_raw_jul14 = pd.read_csv("../input/uber-raw-data-jul14.csv")

uber_raw_aug14 = pd.read_csv("../input/uber-raw-data-aug14.csv")

uber_raw_sep14 = pd.read_csv("../input/uber-raw-data-sep14.csv")



uber_2014_train = [uber_raw_apr14, uber_raw_may14, uber_raw_jun14, uber_raw_jul14,uber_raw_aug14]

uber_2014_test = uber_raw_sep14



uber_data = pd.concat(uber_2014_train,axis=0,ignore_index=True)

df = uber_data

df1= uber_2014_test

uber_data.head()

df.head()
#df=df.groupby(pd.Grouper(key='Date/Time'))

df.head(564530)
df1.tail()
#df=df[~df.index.duplicated()]
df.Timestamp = pd.to_datetime(df['Date/Time'],format='%m/%d/%Y %H:%M:%S') 

df.index = df.Timestamp 

df['Date'] = df.Timestamp.dt.date

#df.tail()                        ##Run this cell twice
df.Timestamp = pd.to_datetime(df['Date/Time'],format='%m/%d/%Y %H:%M:%S') 

df.index = df.Timestamp 

df['Date'] = df.Timestamp.dt.date

df.tail()  
count=df.groupby(pd.Grouper(key='Date')).count()

count.tail()

#df1['Count']=count
df1.tail()
df1.Timestamp = pd.to_datetime(df1['Date/Time'],format='%m/%d/%Y %H:%M:%S') 

df1.index = df1.Timestamp 

df1['Date'] = df1.Timestamp.dt.date

#df1.tail()       ##Run this cell twice
df1.Timestamp = pd.to_datetime(df1['Date/Time'],format='%m/%d/%Y %H:%M:%S') 

df1.index = df1.Timestamp 

df1['Date'] = df1.Timestamp.dt.date

df1.tail()       ##Run this cell twice
count1=df1.groupby(pd.Grouper(key='Date')).count()

count1.tail()

#df1['Count']=count
count= count.drop(columns = ['Lat','Lon','Base'])

count1= count1.drop(columns = ['Lat','Lon','Base'])
count.head()
count1.head()
train=count

test=count1
train['Date/Time'].plot(kind='line',figsize=(15,8), title= 'Daily Ridership', fontsize=14)

test['Date/Time'].plot(figsize=(15,5), title= 'Daily Ridership', fontsize=14)

plt.ylabel('Total Journeys')

plt.xlabel('Month')

plt.show()
y_hat_avg = test.copy()

fit1 = sm.tsa.statespace.SARIMAX(train['Date/Time'], order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()

y_hat_avg['SARIMA'] = fit1.predict(start="2014-09-01", end="2014-09-30", dynamic=True)

plt.figure(figsize=(15,6))

plt.plot( train['Date/Time'], label='Train')

plt.plot(test['Date/Time'], label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.ylabel('Total Journeys')

plt.xlabel('Months')

plt.show()

y_hat_avg = test.copy()

fit1 = ExponentialSmoothing(np.asarray(train['Date/Time']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

plt.figure(figsize=(15,5))

plt.plot( train['Date/Time'], label='Train')

plt.plot(test['Date/Time'], label='Test')

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')

plt.legend(loc='best')

plt.ylabel('Total Journeys')

plt.xlabel('Months')

plt.show()
y_hat_avg = test.copy()



fit1 = Holt(np.asarray(train['Date/Time'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(test))



plt.figure(figsize=(16,5))

plt.plot(train['Date/Time'], label='Train')

plt.plot(test['Date/Time'], label='Test')

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')

plt.legend(loc='best')

plt.show()