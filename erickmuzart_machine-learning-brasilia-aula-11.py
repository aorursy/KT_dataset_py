import os 
os.listdir('../input')
import pandas as pd
df = pd.read_csv('../input/train.csv', parse_dates=[0])
df.head()
test = pd.read_csv('../input/test.csv', parse_dates=[0])
test.head()
df_all = df.append(test, sort=False)
df_all['hour'] = df['datetime'].dt.hour
import numpy as np
df_all['count'] = np.log(df_all['count'] + 1)
df_all['registered'] = np.log(df_all['registered'] + 1)
df_all['casual'] = np.log(df_all['casual'] + 1)
df_all.shape, df.shape, test.shape
df_all.shape
from fastai.imports import *
from fastai.structured import *
add_datepart(df_all, 'datetime', drop=False)
df_all.info()
df = df_all[~df_all['count'].isnull()]
test = df_all[df_all['count'].isnull()]
df.shape, test.shape
train = df[df['datetimeDay'] <= 15]
valid = df[df['datetimeDay'] > 15]
train.shape, valid.shape
feats = [c for c in df.columns if c not in ['casual', 'registered', 'count']]
feats
feats = ['season',
 'holiday',
 'workingday',
 'weather',
 'temp',
 'atemp',
 'humidity',
 'windspeed',
 'datetimeDayofweek',
        'hour',
        'datetimeYear']
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train[feats], train['count'])
rf.predict(valid[feats])
from sklearn.metrics import mean_squared_error
mean_squared_error(valid['count'], rf.predict(valid[feats])) ** (1/2)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
test['count'] = (np.exp(rf.predict(test[feats])) - 1)
test[['datetime', 'count']].to_csv('submission.csv', index=False)








