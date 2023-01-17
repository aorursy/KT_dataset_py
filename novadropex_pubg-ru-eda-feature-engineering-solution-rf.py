import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

np.random.seed(7)



train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv') #Считываем файл CSV

test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv') #Считываем файл CSV

print(train.shape)

print(test.shape)
count = train.shape # Cоздадим переменную, чтобы в итоге узнать сколько было удалено строк.

train[train['winPlacePerc'].isnull()]
train.drop(2744604, inplace = True)
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

train['killWithoutMoving'] = ((train['kills'] > 2) & (train['totalDistance'] == 0))
train['hs_ratio'] = train['headshotKills'] /  train['kills']

train['hs_ratio'] = train['hs_ratio'].fillna(0)
train[train['killWithoutMoving'] == True]
train.drop(train[train['killWithoutMoving'] == True].index, inplace = True)
train = train.drop('killWithoutMoving', 1) 

train = train.drop('totalDistance', 1)
train[train['roadKills'] > 10]
train.drop(train[train['roadKills'] > 10].index, inplace = True)
plt.figure(figsize=(12,4))

sns.distplot(train['hs_ratio'], bins=10)

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['longestKill'], bins = 10)

plt.show()
train.drop(train[train['longestKill'] > 600].index, inplace = True)
train[(train['hs_ratio'] == 1) & (train['kills'] > 9) & (train['assists'] == 0 )].head(20)
train.drop(train[(train['hs_ratio'] == 1) & (train['kills'] > 9) & (train['assists'] == 0 )].index, inplace = True)
plt.figure(figsize=(12,4))

sns.distplot(train['kills'], bins = 10)

plt.show()
train.drop(train[train['kills'] > 45].index, inplace = True)
plt.figure(figsize=(12,4))

sns.distplot(train['walkDistance'], bins = 10)

plt.show()
train.drop(train[train['walkDistance'] > 10000].index, inplace = True)
plt.figure(figsize=(12,4))

sns.distplot(train['swimDistance'], bins = 10)

plt.show()
train.drop(train[train['swimDistance'] > 1000].index, inplace = True)
plt.figure(figsize=(12,4))

sns.distplot(train['rideDistance'], bins = 10)

plt.show()
train.drop(train[train['rideDistance'] > 20000].index, inplace = True)
print(train.shape)
features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills', 'killStreaks', 

            'longestKill', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired']

infos = ['matchDuration', 'matchType', 'maxPlace', 'numGroups']

ELO = ['rankPoints', 'killPoints', 'winPoints']

label = ['winPlacePerc']
sample = train.sample(100000)



f,ax = plt.subplots(figsize=(15, 12))

sns.heatmap(sample[ELO + features + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(15,10))

sns.countplot(train[train['playersJoined']>=75]['playersJoined'])

plt.title('playersJoined')

plt.show()
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)

train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['log_players']=np.log10(train['playersJoined'])
train['skill'] = train['headshotKills'] + 0.01 * train['longestKill'] - train['teamKills'] / (train['kills'] + 1)
data = train.sample(10000)

plt.figure(figsize=(15,10))

sns.scatterplot(x='skill', y='winPlacePerc', data=data)

plt.show()
data = train.sample(10000)

plt.figure(figsize=(15,10))

sns.scatterplot(x='hs_ratio', y='winPlacePerc', data=data)

plt.show()
def transform_hsRatio(x):

    if x == 1 or x == 0:

        return 0.5

    else: 

        return x
train['hs_ratio'] = train['hs_ratio'].apply(transform_hsRatio)
data = train.sample(10000)

plt.figure(figsize=(15,10))

sns.scatterplot(x='hs_ratio', y='winPlacePerc', data=data)

plt.show()
train['distance'] = (train['walkDistance'] + 0.4 * train['rideDistance'] + train['swimDistance']) * (1 / train['matchDuration'])
data = train.sample(10000)

plt.figure(figsize=(15,10))

sns.scatterplot(x='distance', y='winPlacePerc', data=data)

plt.show()
train['boostRatio'] = train['boosts'] ** 2 / train['walkDistance'] ** 0.5

train['boostRatio'].fillna(0, inplace = True)

train['boostRatio'].replace(np.inf, 0, inplace=True)
data = train.sample(10000)

plt.figure(figsize=(15,10))

sns.scatterplot(x='boostRatio', y='winPlacePerc', data=data)

plt.show()
train['healsRatio'] = train['heals'] / train['matchDuration'] ** 0.1

train['healsRatio'].fillna(0, inplace = True)

train['healsRatio'].replace(np.inf, 0, inplace=True)
data = train.sample(10000)

plt.figure(figsize = (15, 10))

sns.scatterplot(x='healsRatio', y='winPlacePerc', data=data)

plt.show()
train['killsRatio'] = train['kills'] / train['matchDuration']**0.1

train['killsRatio'].fillna(0, inplace=True)

train['killsRatio'].replace(np.inf, 0, inplace=True)
data = train.sample(10000)

plt.figure(figsize = (15, 10))

sns.scatterplot(x='killsRatio', y='winPlacePerc', data=data)

plt.show()
engineered = ['log_players', 'matchDurationNorm', 'skill', 'hs_ratio', 'damageDealtNorm', 'distance', 'boostRatio', 'healsRatio', 'killsRatio', 'killsNorm', 'maxPlaceNorm']
sample = train.sample(100000)



f,ax = plt.subplots(figsize=(15, 12))

sns.heatmap(sample[engineered + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
test['skill'] = test['headshotKills'] + 0.01 * test['longestKill'] - test['teamKills'] / (test['kills'] + 1)

def transform_hsRatio(x):

    if x == 1 or x == 0:

        return 0.5

    else: 

        return x

test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['log_players'] = np.log10(test['playersJoined'])



test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['maxPlaceNorm'] = test['maxPlace']*((100-test['playersJoined'])/100 + 1)

test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)



test['hs_ratio'] = test['headshotKills'] /  test['kills']

test['hs_ratio'] = test['hs_ratio'].fillna(0)        

test['hs_ratio'] = test['hs_ratio'].apply(transform_hsRatio)



test['distance'] = (test['walkDistance'] + 0.4 * test['rideDistance'] + test['swimDistance']) * (1 / test['matchDuration'])



test['boostRatio'] = test['boosts'] ** 2 / test['walkDistance'] ** 0.5

test['boostRatio'].fillna(0, inplace = True)

test['boostRatio'].replace(np.inf, 0, inplace=True)



test['healsRatio'] = test['heals'] / test['matchDuration'] ** 0.1

test['healsRatio'].fillna(0, inplace = True)

test['healsRatio'].replace(np.inf, 0, inplace=True)



test['killsRatio'] = test['kills'] / test['matchDuration']**0.1

test['killsRatio'].fillna(0, inplace=True)

test['killsRatio'].replace(np.inf, 0, inplace=True)



test.shape
target = 'winPlacePerc'

new = ['log_players', 'killsNorm', 'damageDealtNorm', 'maxPlaceNorm', 'matchDurationNorm']

y_train = train[target]

features = list(train.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")

features.remove("matchType")

features.remove("winPlacePerc")

features.remove("kills")

features.remove("damageDealt")

features.remove("maxPlace")

features.remove("matchDuration")

x_train = train[features + new]

x_test = test[features + new]

print(x_test.shape, x_train.shape, y_train.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 1)
rf = RandomForestRegressor(n_estimators = 70, min_samples_leaf = 3, max_depth = 23, max_features = 0.5,

                          n_jobs = -1)
rf.fit(x_train, y_train)
print('mae train: ', mean_absolute_error(rf.predict(x_train), y_train))

print('mae val: ', mean_absolute_error(rf.predict(x_val), y_val))
pred = rf.predict(x_test)

test['winPlacePerc'] = pred

submission = test[['Id', 'winPlacePerc']]

submission.to_csv('submission_rf.csv', index=False)