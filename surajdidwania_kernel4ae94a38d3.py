import pandas as pd 

import numpy as np

import os

import datetime

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')
train_data
print ("Train Data Shape ", train_data.shape)

train_data.describe(include='all')
train_data.isnull().sum()
test_data
print ("Test Data Shape ", test_data.shape)

test_data.describe(include='all')
test_data.isnull().sum()
sns.jointplot(x = "popularity",y="revenue",data=train_data)

plt.show()
sns.jointplot(x = "runtime",y="revenue",data=train_data)

plt.show()
sns.distplot(train_data['revenue'])
train_data['log_revenue'] = np.log(train_data['revenue'])

sns.distplot(train_data['log_revenue'])
train_data['release_day','release_month','release_year'] = train_data['release_date'].apply(lambda x : x.split('/'))

train_data['release_date'] = pd.to_datetime(train_data['release_date'])

train_data['dayofweek']=train_data['release_date'].dt.dayofweek
train_data
##leave only important columsn for revenue

## add revenue column in test data as nan

##