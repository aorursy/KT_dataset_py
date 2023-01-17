# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train_users_2.csv',parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])

data_test = pd.read_csv('../input/test_users.csv',parse_dates=['timestamp_first_active','date_account_created','date_first_booking'])

data_train.head()
data_test.head()

# Note that 'date_first_booking' is completely missing in test data
# Note incorrect minimum and maximum age values

data_train.describe()
print ('Number of lines in the training data are,',data_train.shape[0])
countries = pd.read_csv('../input/countries.csv')

countries.head(10)
test_ids = data_test['id']

Nrows_train = data_train.shape[0]  



# Store country names

labels = data_train['country_destination'].values

data_train1 = data_train.drop(['country_destination'], axis=1)



# Combining the test and train data. If this is not done, the number of dummy variable columns do not match in test and train data.

# Some items present in train data and are not present in test data. For example, browser type. 

data_all = pd.concat((data_train1, data_test), axis = 0, ignore_index = True)



# Dropping ids which are saved separately and date of first booking which is completely absent in the test data

data_all = data_all.drop(['id','date_first_booking'], axis=1)

data_all.head()
print(data_all.isnull().sum())
data_all.gender.replace('-unknown-', np.nan, inplace=True)

data_all.first_browser.replace('-unknown-', np.nan, inplace=True)
data_all.loc[data_all.age > 100, 'age'] = np.nan

data_all.loc[data_all.age < 18, 'age'] = np.nan
# Splitting date time data for date account created

data_all['dac_year'] = data_all.date_account_created.dt.year

data_all['dac_month'] = data_all.date_account_created.dt.month

data_all['dac_day'] = data_all.date_account_created.dt.day



# Splitting date time data for time first active

data_all['tfa_year'] = data_all.timestamp_first_active.dt.year

data_all['tfa_month'] = data_all.timestamp_first_active.dt.month

data_all['tfa_day'] = data_all.timestamp_first_active.dt.day



data_all.drop('date_account_created',1, inplace=True)

data_all.drop('timestamp_first_active',1, inplace=True)
data_all.head()
data_all.describe()
data_all.groupby('gender').age.agg(['min','max','mean','count'])
data_all.groupby('gender').age.mean().plot(kind='bar')
data_all.gender.value_counts(dropna=False).plot(kind='bar')
data_all.dac_year.value_counts(sort=False).plot(kind='bar', title='Number of User Accounts Created in a Year')
data_all.tfa_year.value_counts(sort=False).plot(kind='bar', title = 'Number of Users by First Active Year')
data_train.country_destination.value_counts(normalize=True).plot(kind='bar',title='Countries Visited by AirBNB Users')
data_all.language.value_counts(sort=True)
#Note no null data now left

print(data_all.isnull().sum())
# Import sklearn.preprocessing.StandardScaler

#from sklearn.preprocessing import MinMaxScaler



# Initialize a MinMax scaler, then apply it to the numerical features

#scaler = MinMaxScaler()

#numerical = ['age','dac_year','dac_month','dac_day','tfa_year','tfa_month','tfa_day']

#data_all[numerical] = scaler.fit_transform(data_all[numerical])



# Create categorical columns

features = ['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider',\

            'first_affiliate_tracked','signup_app','first_device_type','first_browser']



# get dummies

data_all = pd.get_dummies(data_all,columns=features)

data_all.describe()
data_all.head()
# Splitting train and test for the classifier

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder



V = data_all.values

X_train = V[:Nrows_train]

X_test = V[Nrows_train:]



#Create labels

labler = LabelEncoder()

y = labler.fit_transform(labels)



# Implementation of the classifier (decision tree)

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=22,

                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0)               

xgb.fit(X_train, y)

y_pred = xgb.predict_proba(X_test) 
#Taking the 5 classes with highest probabilities

ids = []  #list of ids

cts = []  #list of countries

for i in range(len(test_ids)):

    idx = test_ids[i]

    ids += [idx] * 5

    cts += labler.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()



#Generate submission

sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])

sub.to_csv('submission.csv',index=False)