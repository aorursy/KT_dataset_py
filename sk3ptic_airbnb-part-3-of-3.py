# Import relevant libraraies

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import os
# Load the data from Part 2

train = pd.read_csv("../input/airbnb-recruiting-new-user-bookings/train_users_2.csv")

test = pd.read_csv("../input/airbnb-recruiting-new-user-bookings/test_users.csv")

train.head()
country = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/countries.csv')

country.head()
test_ids = test['id']

Nrows_train = train.shape[0]  



# Store country names

labels = train['country_destination'].values

train1 = train.drop(['country_destination'], axis=1)



# Combining the test and train data. If this is not done, the number of dummy variable columns do not match in test and train data.

# Some items present in train data and are not present in test data. For example, browser type. 

data_all = pd.concat((train1,test), axis = 0, ignore_index = True)



# Dropping ids which are saved separately and date of first booking which is completely absent in the test data

data_all = data_all.drop(['id','date_first_booking'], axis=1)
data_all.head()
print(data_all.isnull().sum())
data_all.gender.replace('-unknown-', np.nan, inplace=True)

data_all.first_browser.replace('-unknown-', np.nan, inplace=True)
data_all.loc[data_all.age > 85, 'age'] = np.nan

data_all.loc[data_all.age < 18, 'age'] = np.nan
print(data_all.isnull().sum())
data_all.date_account_created = pd.to_datetime(data_all.date_account_created)

data_all.timestamp_first_active = pd.to_datetime(data_all.timestamp_first_active)
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
print(data_all.isnull().sum())
# Import sklearn.preprocessing.StandardScaler

from sklearn.preprocessing import MinMaxScaler



# Initialize a MinMax scaler, then apply it to the numerical features

scaler = MinMaxScaler()

numerical = ['age','dac_year','dac_month','dac_day','tfa_year','tfa_month','tfa_day']

data_all[numerical] = scaler.fit_transform(data_all[numerical])
# Create categorical columns

features = ['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider',\

            'first_affiliate_tracked','signup_app','first_device_type','first_browser']



# get dummies

data_all = pd.get_dummies(data_all,columns=features)
data_all.describe()
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
y_pred
#Taking the 5 classes with highest probabilities

ids = []  #list of ids

cts = []  #list of countries

for i in range(len(test_ids)):

    idx = test_ids[i]

    ids += [idx] * 5

    cts += labler.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
#Generate submission

submission = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])

submission.to_csv('submission.csv',index=False)