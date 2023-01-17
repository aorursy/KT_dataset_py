# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import OneHotEncoder

import datetime as dt

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

submission=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
train.columns
test.columns
#train['Province/State'].unique()
train['Province/State'].fillna('No Province',inplace=True)
test['Province/State'].fillna('No Province',inplace=True)
ohe=OneHotEncoder(handle_unknown='ignore')
train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 
def create_time_features(df):

    df['date'] = df['Date']

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(train)

create_time_features(test)
train=train.drop(columns=['Date'],axis=1)

test=test.drop(columns=['Date'],axis=1)
train=train.drop(columns=['date'],axis=1)

test=test.drop(columns=['date'],axis=1)
train.head(3)
s = (train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
train_dummies=pd.get_dummies(train['Country/Region'])
test_dummies=pd.get_dummies(test['Country/Region'])
train1_dummies=pd.get_dummies(train['Province/State'])

test1_dummies=pd.get_dummies(test['Province/State'])
train=train.drop(['Country/Region','Province/State','Id'],axis=1)

test=test.drop(['Country/Region','Province/State','ForecastId'],axis=1)
train=pd.concat([train,train_dummies,train1_dummies],axis=1)
test=pd.concat([test,test_dummies,test1_dummies],axis=1)
train.head(3)
features=train.drop(['ConfirmedCases','Fatalities'],axis=1)

target1=train['ConfirmedCases']

target2=train['Fatalities']
rf=DecisionTreeRegressor(criterion='mse', splitter='best')
print(features.shape)

print(target1.shape)

print(test.shape)
rf.fit(features,target1)
predict_cases=rf.predict(test)
submission['ConfirmedCases']=predict_cases
rf.fit(features,target2)

predict_fatalities=rf.predict(test)

submission['Fatalities']=predict_fatalities
submission.round().astype(int)
submission.head(3)
submission.to_csv('submission.csv',index=False)