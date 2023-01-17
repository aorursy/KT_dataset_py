# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd
import boto3
from io import StringIO
import io
import string
import random
import json
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# Any results you write to the current directory are saved as output.
# Loading 100k data rows
# Load train data
train = pd.read_csv('../input/train.csv', nrows=100000)

# Load test data
test = pd.read_csv('../input/test.csv', nrows=100000)

# Load destination data
destination = pd.read_csv('../input/destinations.csv', nrows=100000)

train.head()
train.columns
train.info()
# Check the percentage of Nan in dataset
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train['hotel_cluster'].count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.heatmap(train.corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)
# Frequency of posa continent
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot('posa_continent', data=train,order=[0,1,2,3,4],ax=ax)

# frequency of hotel continent
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot('hotel_continent', data=train,order=[0,2,3,4,5,6],ax=ax)

# Frequency of booking through mobile
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot(x='is_mobile',data=train, order=[0,1],ax=ax)

# frequency of bookings with package
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot(x='is_package',data=train, order=[0,1], ax=ax)

train.info()
# Function to convert date object into relevant attributes
def convert_date_into_days(df):
    df['srch_ci'] = pd.to_datetime(df['srch_ci'])
    df['srch_co'] = pd.to_datetime(df['srch_co'])
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).astype('timedelta64[D]')
    df['no_of_days_bet_booking'] = (df['srch_ci'] - df['date_time']).astype('timedelta64[D]')
    
    # For hotel check-in
    # Month, Year, Day
    df['Cin_day'] = df["srch_ci"].apply(lambda x: x.day)
    df['Cin_month'] = df["srch_ci"].apply(lambda x: x.month)
    df['Cin_year'] = df["srch_ci"].apply(lambda x: x.year)

convert_date_into_days(train)
train.info()
# Count the bookings in each month
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot('Cin_month',data=train[train["is_booking"] == 1],order=list(range(1,13)),ax=ax)
# Count the bookings as per the day
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot('Cin_day',data=train[train["is_booking"] == 1],order=list(range(1,32)),ax=ax)
# Count the bookings as per the stay_duration
fig, ax = plt.subplots()
fig.set_size_inches(13, 8)
sns.countplot('stay_dur',data=train[train["is_booking"] == 1],ax=ax)
# Check the percentage of Nan in dataset
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train['hotel_cluster'].count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
#train['Cin_day'].value_counts() = 26
#train['Cin_month'].value_counts() = 8
#train['Cin_year'].value_counts() = 2014
#train['stay_dur'].value_counts() = 1
#train['no_of_days_bet_booking'].value_counts() = 0
train['Cin_day'] = train['Cin_day'].fillna(26.0)
train['Cin_month'] = train['Cin_month'].fillna(8.0)
train['Cin_year'] = train['Cin_year'].fillna(2014.0)
train['stay_dur'] = train['stay_dur'].fillna(1.0)
train['no_of_days_bet_booking'] = train['no_of_days_bet_booking'].fillna(0.0)
# Fill average values in place for nan, fill with mean
train['orig_destination_distance'].fillna(train['orig_destination_distance'].mean(), inplace=True)
train.head()
## Remove datetime object from the dataset
#columns to remove
user_id = train['user_id']
columns = ['date_time', 'srch_ci', 'srch_co','user_id','srch_destination_type_id','srch_destination_id']
train.drop(columns=columns,axis=1,inplace=True)
train.info()