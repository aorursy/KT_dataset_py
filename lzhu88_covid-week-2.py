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
# Preparation

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
# Load the data

train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

submission=pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')



print(np.shape(train))

print(np.shape(test))
train.head(10)
train.describe()
train.info() # Found that only Province_State has different row numbers
train['Province_State'].value_counts()
train['Province_State'].unique()[:50] # nan found
train['Province_State'].fillna('No Data',inplace=True)

test['Province_State'].fillna('No Data',inplace=True)
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



# drop the original date columns

train=train.drop(columns=['Date'],axis=1)

test=test.drop(columns=['Date'],axis=1)



train=train.drop(columns=['date'],axis=1)

test=test.drop(columns=['date'],axis=1)
train.head()
train.info()
train_dummies_Province_State = pd.get_dummies(train['Province_State'])

test_dummies_Province_State = pd.get_dummies(test['Province_State'])



train_dummies_Country_Region = pd.get_dummies(train['Country_Region'])

test_dummies_Country_Region = pd.get_dummies(test['Country_Region'])



train=train.drop(['Country_Region','Province_State','Id'],axis=1)

test=test.drop(['Country_Region','Province_State','ForecastId'],axis=1)



train=pd.concat([train,train_dummies_Province_State,train_dummies_Country_Region],axis=1)

test=pd.concat([test,test_dummies_Province_State,test_dummies_Country_Region],axis=1)



train.head()
features=train.drop(['ConfirmedCases','Fatalities'],axis=1)

target1=train['ConfirmedCases']

target2=train['Fatalities']



print(features.shape)

print(target1.shape)

print(target2.shape)

print(test.shape)
model = DecisionTreeRegressor(criterion='mse', splitter='best')
model.fit(features,target1)
submission['ConfirmedCases'] = model.predict(test)
model.fit(features,target2)

submission['Fatalities'] = model.predict(test)
submission.round().astype(int)
submission.head()
submission.to_csv('submission.csv',index=False)