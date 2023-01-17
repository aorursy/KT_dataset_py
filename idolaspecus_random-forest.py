import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

train['dayofweek'] = train['datetime'].astype('datetime64').dt.dayofweek

test['dayofweek'] = test['datetime'].astype('datetime64').dt.dayofweek

train['year'] = train['datetime'].astype('datetime64').dt.year

test['year'] = test['datetime'].astype('datetime64').dt.year

train['hour'] = train['datetime'].astype('datetime64').dt.hour

test['hour'] = test['datetime'].astype('datetime64').dt.hour



train['season'] = train['season'].astype('category')

test['season'] = test['season'].astype('category')



train['holiday'] = train['holiday'].astype('category')

test['holiday'] = test['holiday'].astype('category')



train['dayofweek'] = train['dayofweek'].astype('category')

test['dayofweek'] = test['dayofweek'].astype('category')



train['year'] = train['year'].astype('category')

test['year'] = test['year'].astype('category')



train['hour'] = train['hour'].astype('category')

test['hour'] = test['hour'].astype('category')



train['temp_atemp'] = np.log(train['temp'] * train['atemp'] + 0.0001)

test['temp_atemp'] = np.log(test['temp'] * test['atemp'] + 0.0001)



train['hum_atemp'] = np.log(train['humidity'] * train['atemp'] + 0.0001)

test['hum_atemp'] = np.log(test['humidity'] * test['atemp'] + 0.0001)



train['hum_temp'] = np.log(train['humidity'] * train['temp'] + 0.0001)

test['hum_temp'] = np.log(test['humidity'] * test['temp'] + 0.0001)



train['hum_2temp'] = np.log(train['humidity'] * train['atemp'] * train['temp'] + 0.0001)

test['hum_2temp'] = np.log(test['humidity'] * test['atemp'] * test['temp'] + 0.0001)
import seaborn as sns

import matplotlib.pyplot as plt
dt64 = train['datetime'].astype('datetime64')

ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

X = train.drop(['datetime', 'hum_2temp', 'hum_atemp'], axis=1)

#X['ts'] = ts
dt64 = test['datetime'].astype('datetime64')

ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

Y = test.drop(['datetime', 'hum_2temp', 'hum_atemp'], axis=1)

#Y['ts'] = ts
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=2000, max_depth=None, max_features=8)



rf.fit(X.drop(['count','casual','registered'], axis=1), 

       np.log(pd.concat([X['casual'],X['registered']],axis=1)+1))
import xgboost

xgb = xgboost.XGBRegressor()

rf.score(X.drop(['count','casual','registered'], axis=1), 

         np.log(pd.concat([X['casual'],X['registered']],axis=1)+1))
importance_df = pd.DataFrame(rf.feature_importances_)

importance_df['columns'] = X.drop(['count','casual','registered'], axis=1).columns

importance_df = importance_df.sort_values(0, ascending = False)

print(importance_df)


pred = np.exp(rf.predict(Y))-1

pred = pred.sum(axis=1)



submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

submission["count"] = pred

submission.to_csv("/kaggle/working/submission.csv", index=False)
