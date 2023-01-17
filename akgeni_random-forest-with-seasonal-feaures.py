from sklearn import preprocessing

import pandas as pd

import numpy as np

from math import pi, sin, cos

from datetime import datetime
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

data = pd.concat([train, test])

test_split = train.shape[0]
data.head()
data['date'] = pd.to_datetime(data.datetime)

data['hour'] = data['date'].dt.hour

data['weekday'] = data['date'].dt.weekday

data['month'] = data['date'].dt.month

data['year'] = data['date'].dt.year
data['hour_sin'] = data.apply(lambda x: sin(x['hour'] / (24.0 * 2 * pi)), axis=1)

data['hour_cos'] = data.apply(lambda x: cos(x['hour'] / (24.0 * 2 * pi)), axis=1)

data['weekday_sin'] = data.apply(lambda x: sin(x['weekday'] / (7.0 * 2 * pi)), axis=1)

data['weekday_cos'] = data.apply(lambda x: cos(x['weekday'] / (7.0 * 2 * pi)), axis=1)

data['month_cos'] = data.apply(lambda x: cos( ((x['month']-5)%12) / (12.0 * 2 * pi)), axis=1)

data['month_sin'] = data.apply(lambda x: sin( ((x['month']-5)%12) / (12.0 * 2 * pi)), axis=1)

data['season_cos'] = data.apply(lambda x: cos( ((x['season']-3)%4) / (4.0 * 2 * pi)), axis=1)

data['season_sin'] = data.apply(lambda x: sin( ((x['season']-3)%4) / (4.0 * 2 * pi)), axis=1)
# datetime_test will keep the datetime column which we will use for submission

datetime_test = data[test_split:]['datetime'].copy()
#Train Test split

X_train = data[:test_split].drop(['date', 'datetime', 'casual', 'registered', 'count'], inplace=False, axis=1)

y_train = data[:test_split]['count']

X_test = data[test_split:].drop(['date', 'datetime', 'casual', 'registered', 'count'], inplace=False, axis=1)

y_test = data[test_split:]['count']
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

# can do parameter tunning

rf = RandomForestRegressor(n_estimators=500)

scores = cross_val_score(rf, X_train, y_train, cv=5)

scores  
rf.fit(X_train, y_train)
result = rf.predict(X_test)
result_df = pd.DataFrame({'datetime': datetime_test, 'count': result})

result_df = result_df[['datetime', 'count']]
result_df.to_csv('bike_sharing.csv', index=None)