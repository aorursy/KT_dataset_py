import pandas as pd          

import numpy as np          # For mathematical calculations 

import matplotlib.pyplot as plt  # For plotting graphs 

from datetime import datetime    # To access datetime 

from pandas import Series        # To work on series 

%matplotlib inline 

import warnings                   # To ignore the warnings warnings.filterwarnings("ignore")
import os

print(os.listdir("../input/bullettrain-timeseries-data/"))

train=pd.read_csv("../input/bullettrain-timeseries-data/Train.csv")

test=pd.read_csv("../input/bullettrain-timeseries-data/Test.csv")
train.head()

train.columns

train.describe()

train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
train.head()

for i in (train, test):

    i['year']=i.Datetime.dt.year 

    i['month']=i.Datetime.dt.month 

    i['day']=i.Datetime.dt.day

    i['Hour']=i.Datetime.dt.hour 
train.head()
train['dayofweek']=train['Datetime'].dt.dayofweek

test['dayofweek']=test['Datetime'].dt.dayofweek

def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train['Datetime'].apply(applyer) 

temp = test['Datetime'].apply(applyer) 

train['weekend']=temp2

test['weekend']=temp


train.head()

train.index = train['Datetime']

train.drop('ID',1)
ts = train['Count'] 
plt.figure(figsize=(16,8))

plt.plot(ts,label='count')

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend()
train.groupby('year')['Count'].mean().plot.bar()



train.groupby('month')['Count'].mean().plot.bar()
train.groupby(['year','month'])['Count'].mean().plot()
df =train.groupby('year')['month'].nunique()

df
df.plot.bar()

train.groupby('day')['Count'].mean().plot.bar()
train.groupby('Hour')['Count'].mean().plot.bar()
train.groupby('weekend')['Count'].mean().plot.bar()

train.groupby('dayofweek')['Count'].mean().plot.bar()
train.drop('ID',1)
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

train.index
# Hourly time series 

hourly = train.resample('H').mean() 

# Converting to daily mean 

daily = train.resample('D').mean() 

# Converting to weekly mean 

weekly = train.resample('W').mean() 

# Converting to monthly mean 

monthly = train.resample('M').mean()

ig, axs = plt.subplots(4,1) 

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])

daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])

weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2])

monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 

plt.show()

test.head()


train=train.resample('D').mean()

Train=train.ix['2012-08-25':'2014-06-24'] 

valid=train.ix['2014-06-25':'2014-09-25']

Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')

plt.xlabel("Datetime")

plt.ylabel("Passenger count")

plt.legend(loc='best') 

plt.show()
import statsmodels.api as sm 

sm.tsa.seasonal_decompose(Train.Count).plot() 

result = sm.tsa.stattools.adfuller(train.Count) 

plt.show()
from statsmodels.tsa.holtwinters import Holt

y_hat_avg = valid.copy() 

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 

plt.legend(loc='best') 

plt.show()
