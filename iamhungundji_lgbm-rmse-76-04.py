import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/electricity-consumption/train.csv')

test = pd.read_csv('/kaggle/input/electricity-consumption/test.csv')

test1 = pd.read_csv('/kaggle/input/electricity-consumption/test.csv')
train.head()
train['datetime'] = pd.to_datetime(train.datetime,format='%Y-%m-%d %H:%M:%S') 

test['datetime'] = pd.to_datetime(test.datetime,format='%Y-%m-%d %H:%M:%S')
for i in (train, test):

    i['Year'] = i.datetime.dt.year

    i['Month'] = i.datetime.dt.month

    i['Day'] = i.datetime.dt.day

    i['Hour'] = i.datetime.dt.hour
train = train.drop(['ID','datetime'], axis=1)

test = test.drop(['ID','datetime'], axis=1)
train.head()
train.shape, test.shape
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train.shape, test.shape
train['electricity_consumption'] = np.log(np.log(train['electricity_consumption']))

train['windspeed'] = np.log(train['windspeed'])

train['pressure'] = np.log(train['pressure'])



test['pressure'] = np.log(test['pressure'])

test['windspeed'] = np.log(test['windspeed'])
from sklearn.preprocessing import StandardScaler

cols = ['temperature', 'var1']

scaler = StandardScaler().fit(train[cols])



train[cols] = scaler.transform(train[cols])

test[cols] = scaler.transform(test[cols])

scaler.mean_
train.head()
train['electricity_consumption'].hist()
from sklearn.model_selection import train_test_split

X = train.drop('electricity_consumption', axis=1)

y = train['electricity_consumption']



X_test = test.copy()



x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

model = LGBMRegressor(learning_rate=0.5, max_depth=5)

model.fit(x_train, y_train)
model.score(x_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = model.predict(x_val)

np.sqrt(mean_squared_error(np.exp(np.exp(y_pred)), np.exp(np.exp(y_val))))
import lightgbm

lightgbm.plot_importance(model)
pred_test = model.predict(X_test)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import KFold



fold = KFold(n_splits=15, shuffle=True, random_state=2020)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMRegressor(max_depth=5, 

                      learning_rate=0.4)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='rmse',

          verbose=200)

    pred_y = m.predict(x_val)

    print("err_lgm: ",np.sqrt(mean_squared_error(np.exp(np.exp(pred_y)), np.exp(np.exp(y_val)))))

    err.append(np.sqrt(mean_squared_error(np.exp(np.exp(pred_y)), np.exp(np.exp(y_val)))))

    pred_test = m.predict(X_test)

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err)
submission = pd.DataFrame()

submission['ID'] = test1['ID']

submission['electricity_consumption'] = np.exp(np.exp(np.mean(y_pred_tot_lgm, 0)))

submission.to_csv('LGB.csv', index=False)