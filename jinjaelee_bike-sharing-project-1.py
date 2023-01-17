# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data import

train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

sampleSubmission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')



train.head()
train.corr()
train.isnull().sum()
# datetime 컬럼을 datetime type으로 변경

train['datetime'] = pd.to_datetime(train['datetime'])



# datetime 관련 새로운 feature 생성

train['year'] = train['datetime'].apply(lambda x : x.year)

train['month'] = train['datetime'].apply(lambda x : x.month)

train['day'] = train['datetime'].apply(lambda x : x.day)

train['hour'] = train['datetime'].apply(lambda x : x.hour)



# weather 와 season을  one-hot encording화 

weather=pd.get_dummies(train['weather'],prefix='weather')

train=pd.concat([train,weather],axis=1)



season = pd.get_dummies(train['season'], prefix='season')

train=pd.concat([train, season], axis=1)

train.head()
# test 데이터 셋에도 동일하게 적용

test['datetime'] = pd.to_datetime(test['datetime'])

test['year'] = test['datetime'].apply(lambda x : x.year)

test['month'] = test['datetime'].apply(lambda x : x.month)

test['day'] = test['datetime'].apply(lambda x : x.day)

test['hour'] = test['datetime'].apply(lambda x : x.hour)



weather=pd.get_dummies(test['weather'],prefix='weather')

test=pd.concat([test,weather],axis=1)



season = pd.get_dummies(test['season'], prefix='season')

test=pd.concat([test, season], axis=1)

test.head()

train.columns
train_label = train.columns

droped_train_label = train_label.drop(['datetime', 'season', 'weather', 'casual', 'registered', 'count'])

print(droped_train_label)



X = df[droped_train_label]

y = df['count']

print(X.shape)

print(y.shape)

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators= 500, random_state=42)


rf.fit(X, y)
# 예측을 해봅시다.

predictions = rf.predict(test_df[droped_train_label])
sampleSubmission['count'] = predictions

display(predictions)

print(sampleSubmission)

sampleSubmission.to_csv('pre0.csv', index=False)