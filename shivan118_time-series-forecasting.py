# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # For plotting graphs 

from datetime import datetime    # To access datetime 

from pandas import Series        # To work on series 

%matplotlib inline 

import warnings                   # To ignore the warnings warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path1 = '/kaggle/input/Train.csv'

path2 = '/kaggle/input/Test.csv'
train=pd.read_csv(path1) 

test=pd.read_csv(path2)
# copy of train and test data so that even if we do changes in these dataset we do not lose the original dataset.



train_original=train.copy() 

test_original=test.copy()
print(train.columns)
print(test.columns)
train.head()
test.head()
train.shape
test.shape
train.dtypes
test.dtypes
train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M') 

train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')
for i in (train, test, test_original, train_original):

    i['year']=i.Datetime.dt.year 

    i['month']=i.Datetime.dt.month 

    i['day']=i.Datetime.dt.day

    i['Hour']=i.Datetime.dt.hour 

train['day of week']=train['Datetime'].dt.dayofweek 

temp = train['Datetime']


# Let’s assign 1 if the day of week is a weekend and 0 if the day of week in not a weekend.



def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train['Datetime'].apply(applyer) 

train['weekend']=temp2
# look at the time series.



train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis. 

df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Passenger Count') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')
# Let us try to verify our hypothesis using the actual data.



# Our first hypothesis was traffic will increase as the years pass by. So let’s look at yearly passenger count.



train.groupby('year')['Count'].mean().plot.bar()
# Our second hypothesis was about increase in traffic from May to October. So, let’s see the relation between count and month.



train.groupby('month')['Count'].mean().plot.bar()
# Let’s look at the monthly mean of each year separately.



temp=train.groupby(['year', 'month'])['Count'].mean() 

temp.plot(figsize=(15,5), title= 'Passenger Count(Monthwise)', fontsize=14)
# Let’s look at the daily mean of passenger count.



train.groupby('day')['Count'].mean().plot.bar()


# We also made a hypothesis that the traffic will be more during peak hours. So let’s see the mean of hourly passenger count.



train.groupby('Hour')['Count'].mean().plot.bar()
# Let’s try to validate our hypothesis in which we assumed that the traffic will be more on weekdays.



train.groupby('weekend')['Count'].mean().plot.bar()
# Now we will try to look at the day wise passenger count.



# Note - 0 is the starting of the week, i.e., 0 is Monday and 6 is Sunday.



train.groupby('day of week')['Count'].mean().plot.bar()
train=train.drop('ID',1)
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

# Hourly time series 

hourly = train.resample('H').mean() 

# Converting to daily mean 

daily = train.resample('D').mean() 

# Converting to weekly mean 

weekly = train.resample('W').mean() 

# Converting to monthly mean 

monthly = train.resample('M').mean()
# Let’s look at the hourly, daily, weekly and monthly time series.



fig, axs = plt.subplots(4,1) 

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 

daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1]) 

weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 

monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3])
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 

test.index = test.Timestamp 
# Converting to daily mean 

test = test.resample('D').mean() 
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 
# Converting to daily mean 

train = train.resample('D').mean()
Train=train.loc['2012-08-25':'2014-06-24']

valid=train.loc['2014-06-25':'2014-09-2']
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Passenger count") 

plt.legend(loc='best') 

plt.show()

Train=train.loc['2012-08-25':'2014-06-24'] 

valid=train.loc['2014-06-25':'2014-09-25']
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Passenger count") 

plt.legend(loc='best') 

plt.show()
dd= np.asarray(Train.Count) 

y_hat = valid.copy() 

y_hat['naive'] = dd[len(dd)-1] 

plt.figure(figsize=(12,8)) 

plt.plot(Train.index, Train['Count'], label='Train') 

plt.plot(valid.index,valid['Count'], label='Valid') 

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 

plt.legend(loc='best') 

plt.title("Naive Forecast") 

plt.show()
y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 20 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

y_hat_avg = valid.copy() 

fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False) 

y_hat_avg['SES'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show()
# Lets visualize all these parts.



import statsmodels.api as sm 

sm.tsa.seasonal_decompose(Train.Count).plot() 

result = sm.tsa.stattools.adfuller(train.Count) 

plt.show()
y_hat_avg = valid.copy() 

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 

plt.legend(loc='best') 

plt.show()