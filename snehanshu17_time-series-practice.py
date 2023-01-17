# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
from datetime import datetime   
from pandas import Series     
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={'figure.figsize':(16.7,6.27)})
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Train.csv")
print(data.head())
print("-----------------------Data Types-----------------")
print(data.dtypes)
print("-------------------------Data Frame------------")
print(data.shape)
#changing data field to datetime
data.Datetime=pd.to_datetime(data.Datetime,format='%d-%m-%Y %H:%M')
data.dtypes
i=0
year=[]
month=[]
day=[]
day_of_week=[]
hour=[]
while i<data.shape[0]:
    temp=data.iloc[i,1]
    year.append(temp.year)
    month.append(temp.month)
    day.append(temp.day)
    day_of_week.append(temp.dayofweek)
    hour.append(temp.hour)
    i+=1
train=data
train["year"]=year
train["month"] = month
train["day"] = day
train["day_of_week"]=day_of_week
train["hour"]=hour
train=train.drop("ID",1)
train.head()
plt.figure(figsize=(20,12))
sns.barplot(x='year',y='Count',data=train)
sns.set_style("whitegrid")
plt.title("Capturing the Trend ")
plt.figure(figsize=(20,12))
sns.barplot(x='month',y='Count',data=train)
sns.set_style("whitegrid")
plt.title("Capturing the Trend in month ")
plt.figure(figsize=(20,12))
sns.pointplot(x='day',y='Count',data=train)
sns.set_style("whitegrid")
plt.title("Capturing the Trend in day ")
fig, axs = plt.subplots(2,1)
sns.pointplot(x='day_of_week',y='Count',ax=axs[0],data=train).set_title("Trends in WeekDays")
sns.barplot(x='day_of_week',y='Count',ax=axs[1],data=train).set_title("Trends in WeekDays")
sns.set_style("whitegrid")
plt.show()
plt.figure(figsize=(20,12))
sns.pointplot(x='hour',y='Count',data=train)
sns.set_style("whitegrid")
plt.title("Capturing the Trend in hour ")
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp

hourly = train.resample('H').mean()
daily = train.resample('D').mean()
weekly = train.resample('W').mean()
monthly = train.resample('M').mean()
hourly
fig, axs = plt.subplots(3,1)

sns.set(rc={'figure.figsize':(16.7,10.27)})
# sns.pointplot(data=hourly,y="Count",x=hourly.index,ax=axs[0]).set_title("Hourly")

sns.pointplot(data=daily,y="Count",x=daily.index,ax=axs[0]).set_title("Daily")
sns.pointplot(data=weekly,y="Count",x=weekly.index,ax=axs[1]).set_title("Weekly")
sns.pointplot(data=monthly,y="Count",x=monthly.index,ax=axs[2]).set_title("Monthly")
plt.show()
train.Timestamp=pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train.index=train.Timestamp

train = train.resample('D').mean()
train.to_csv("train_eda.csv")
train.head()
