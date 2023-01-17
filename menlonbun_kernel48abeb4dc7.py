# -*- coding: utf-8 -*-

"""

Created on Mon Jan 13 16:53:36 2020



@author: Gu Yuhang

"""

%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing



train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

y_t = 'winPlacePerc'



train.drop(2744604, inplace=True)



train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)

train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

train['healsANDboosts'] = train['heals']+train['boosts']

train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']

train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

train['headshot_rate'] = train['headshotKills'] / train['kills']

train['headshot_rate'] = train['headshot_rate'].fillna(0)



test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['maxPlaceNorm'] = test['maxPlace']*((100-test['playersJoined'])/100 + 1)

test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)

test['healsANDboosts'] = test['heals']+test['boosts']

test['totalDistance'] = test['walkDistance']+test['rideDistance']+test['swimDistance']

test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))

test['headshot_rate'] = test['headshotKills'] / test['kills']

test['headshot_rate'] = test['headshot_rate'].fillna(0)



train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

train.drop(train[train['roadKills'] > 10].index, inplace=True)

train.drop(train[train['kills'] > 30].index, inplace=True)

train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)

train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)

train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

train.drop(train[train['heals'] >= 40].index, inplace=True)



train = pd.get_dummies(train, columns=['matchType'])

test = pd.get_dummies(test, columns=['matchType'])
s = 500000

train = train.sample(s)



k = list(train.columns)

k.remove("Id")

k.remove("matchId")

k.remove("groupId")

y_ = np.array(train[y_t])

k.remove(y_t)



x_ = train[k]

x_test = test[k]



random_seed=1

x_train, x_train_test, y_train, y_train_test = train_test_split(x_, y_, test_size = 0.1, random_state=random_seed)

md = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
%%time

md.fit(x_train, y_train)

print('train_MAE: ', mean_absolute_error(md.predict(x_train), y_train))

print('test_MAE: ', mean_absolute_error(md.predict(x_train_test), y_train_test))
importance = pd.DataFrame({'cols':x_train.columns, 'importance':md.feature_importances_}).sort_values('importance', ascending=False)

print(importance[:20])
%%time

p = md.predict(x_test)

test['winPlacePerc'] = p

submission = test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)