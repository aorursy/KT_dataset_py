# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_log_error

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/bike-sharing-demand/train.csv')

test = pd.read_csv('../input/bike-sharing-demand/test.csv')

train.head()
train['year'] = [t.year for t in pd.DatetimeIndex(train.datetime)]

train['month'] = [t.month for t in pd.DatetimeIndex(train.datetime)]

train['day'] = [t.day for t in pd.DatetimeIndex(train.datetime)]

train['hour'] = [t.hour for t in pd.DatetimeIndex(train.datetime)]



test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]

test['month'] = [t.month for t in pd.DatetimeIndex(test.datetime)]

test['day'] = [t.day for t in pd.DatetimeIndex(test.datetime)]

test['hour'] = [t.hour for t in pd.DatetimeIndex(test.datetime)]
train.head()
train.drop('datetime',axis=1,inplace=True)

test.drop('datetime',axis=1,inplace=True)
train.head()
sns.set(rc={'figure.figsize':(11.7,8.27)})

fig, ax = plt.subplots(2,2)

sns.barplot(train['season'],train['count'],ax=ax[0,0]);

sns.barplot(train['holiday'],train['count'],ax=ax[0,1]);

sns.barplot(train['workingday'],train['count'],ax=ax[1,0]);

sns.barplot(train['weather'],train['count'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(11.7,8.27)})

fig, ax = plt.subplots(2,2)

sns.distplot(train['temp'],ax=ax[0,0]);

sns.distplot(train['atemp'],ax=ax[0,1]);

sns.distplot(train['humidity'],ax=ax[1,0]);

sns.distplot(train['windspeed'],ax=ax[1,1]);
sns.set(rc={'figure.figsize':(15,10)})

sns.heatmap(train.corr(),annot=True,linewidths=0.5);
train.drop(['casual','registered'],axis=1,inplace=True)
sns.set(rc={'figure.figsize':(20,5)})

sns.barplot(x=train['month'],y=data['count']);
season = pd.get_dummies(train['season'],prefix='season')

train = pd.concat([train,season],axis=1)
train.drop('season',axis=1,inplace=True)

train.head()
weather = pd.get_dummies(train['weather'],prefix='weather')



train = pd.concat([train,weather],axis=1)



train.drop('weather',axis=1,inplace=True)

train.head()
train.columns.to_series().groupby(data.dtypes).groups
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = train.drop('count',axis=1)

y = train['count']

X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

model_rf = rf.fit(X_train,y_train)

y_pred_rf = model_rf.predict(X_test)

np.sqrt(mean_squared_log_error(y_test,y_pred_rf))
n_estimators = [int(x) for x in range(200,2000,100)]

max_feature = ['auto','sqrt']

min_sample_split = [2,5,10]

min_sample_leaf = [1,2,4]

max_depth = [int(x) for x in range(10,110,11)]

max_depth.append(None)
random_grid = {'n_estimators': n_estimators,

              'max_depth': max_depth,

              'max_features': max_feature,

              'min_samples_leaf': min_sample_leaf,

              'min_samples_split': min_sample_split}
random_grid
rf_tune = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator=rf_tune,param_distributions=random_grid,n_iter=100,cv=5,verbose= 2,n_jobs=-1)
final_rf = RandomForestRegressor(max_depth=87,max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=1300)

final_model_rf = final_rf.fit(X_train,y_train)

y_final_pred = final_model_rf.predict(X_test)

np.sqrt(mean_squared_log_error(y_test,y_final_pred))