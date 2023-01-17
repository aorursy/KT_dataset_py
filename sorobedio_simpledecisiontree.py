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
import pandas as pd
from pathlib import Path
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
base_dir='/kaggle/input/covid19-global-forecasting-week-3/'
train_file='train.csv'
test_file='test.csv'
submit_file='submission.csv'
train_df = pd.read_csv(base_dir+'train.csv')
test_df = pd.read_csv(base_dir+'test.csv')
submission = pd.read_csv(base_dir+'submission.csv')
train_df.head()
test_df.head()
train=train_df.copy()
test= test_df.copy()
train['Province_State'][train.Province_State.isna()] = train['Country_Region'][train.Province_State.isna()]
test['Province_State'][test.Province_State.isna()] = test['Country_Region'][test.Province_State.isna()]
train.head()
test.head()
def decomposedate(df):
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    df['Day_of_Week']=df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Week_of_Year'] = df['Date'].dt.weekofyear
    df['Quarter'] = df['Date'].dt.quarter 
    df.drop('Date',1,inplace=True)
    return df
train=decomposedate(train)
test=decomposedate(test)
train.head()
test.head()
submission=pd.DataFrame(columns=submission.columns)
l1=LabelEncoder()
l2=LabelEncoder()
train['Country_Region']=l1.fit_transform(train['Country_Region'])
train['Province_State']=l2.fit_transform(train['Province_State'])

test['Country_Region']=l1.fit_transform(test['Country_Region'])
test['Province_State']=l2.fit_transform(test['Province_State'])
y1=train[['ConfirmedCases']]
y2=train[['Fatalities']]
test_id=test['ForecastId'].values.tolist()
train.pop('Id')
test.pop('ForecastId')
train.head()
test.head()
features_cols=['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']
train_x=train[features_cols]
test_x = test[features_cols]
model_1=DecisionTreeClassifier()
model_2=DecisionTreeClassifier()
model_1.fit(train_x,y1)
model_2.fit(train_x,y2)
test_y1=model_1.predict(test_x)
test_y2=model_2.predict(test_x)
submission=pd.DataFrame(columns=submission.columns)
submission['ForecastId']=test_id
submission['ConfirmedCases']=test_y1
submission['Fatalities']=test_y2
submission.head()
submission.to_csv('submission.csv',index=False)



























