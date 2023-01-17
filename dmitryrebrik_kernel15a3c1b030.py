import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
train.info()
train.head()
train.describe()
train[train['winPlacePerc'].isnull()]
train.drop(2744604, inplace=True)
# Создание признака totalDistance

train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

# Создание признака killsWithoutMoving

train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
# Создание признака headshot_rate

train['headshot_rate'] = train['headshotKills'] / train['kills']

train['headshot_rate'] = train['headshot_rate'].fillna(0)
display(train[train['killsWithoutMoving'] == True].shape)

train[train['killsWithoutMoving'] == True].head(10)
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)

train.drop('killsWithoutMoving', axis=1, inplace=True)
plt.figure(figsize=(12, 4))

sns.distplot(train['roadKills'], bins=50)

plt.show()
train[train['roadKills'] >= 10][['Id', 'kills', 'roadKills', 'rideDistance', 'walkDistance']]
train.drop(train[train['roadKills'] >= 10].index, inplace=True)
plt.figure(figsize=(12,4))

sns.countplot(data=train, x=train['kills']).set_title('Kills')

plt.show()
train[(train['kills'] > 30) & (train['totalDistance'] < 2000)][['kills','totalDistance']]
train.drop(train[(train['kills'] > 30) & (train['totalDistance'] < 2000)].index, inplace=True)

train.drop('totalDistance', axis=1, inplace=True)
plt.figure(figsize=(15,4))

sns.distplot(train['headshot_rate'], bins=10)

plt.show()
display(train[(train['headshot_rate'] == 1) & (train['kills'] >= 10)].shape)

train[(train['headshot_rate'] == 1) & (train['kills'] >= 10)].head(10)
train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] >= 10)].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train['longestKill'], bins=10)

plt.show()
display(train[train['longestKill'] >= 1000].shape)

train[train['longestKill'] >= 1000].head(10)
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train[train['walkDistance'] >= 10000]['walkDistance'], bins=10)

plt.show()
display(train[train['walkDistance'] >= 10000].shape)

train[train['walkDistance'] >= 10000].head(10)
v = 22.5 / 3.6 # км/ч в м/с
train[train['walkDistance'] >= v * train['matchDuration']]
train.drop(train[train['walkDistance'] >= v * train['matchDuration']].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train['rideDistance'], bins=10)

plt.show()
display(train[train['rideDistance'] >= 20000].shape)

train[train['rideDistance'] >= 20000].head(10)
v = 100 / 3.6 # км/ч в м/с

train[train['rideDistance'] >= v * train['matchDuration']]
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train['swimDistance'], bins=10)

plt.show()
train[train['swimDistance'] >= 2000]
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train['weaponsAcquired'], bins=100)

plt.show()
display(train[train['weaponsAcquired'] >= 50].shape)

train[train['weaponsAcquired'] >= 50].head(10)
train.drop(train[train['weaponsAcquired'] >= 50].index, inplace=True)
plt.figure(figsize=(12,4))

sns.distplot(train['heals'], bins=10)

plt.show()
display(train[train['heals'] >= 40].shape)

train[train['heals'] >= 40].head(10)
train.drop(train[train['heals'] >= 40].index, inplace=True)
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(15,10))

sns.countplot(train[train['playersJoined']>=75]['playersJoined'])

plt.title('playersJoined')

plt.show()
# создание нормализованных признаков

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)

train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

# сравнение нормализованных признаков

to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm', 'playersJoined']

train[to_show][0:11]
# Turn groupId and match Id into categorical types

train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')



# Get category coding for groupId and matchID

train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes



# Get rid of old columns

train.drop(columns=['groupId', 'matchId'], inplace=True)



# Lets take a look at our newly created features

train[['groupId_cat', 'matchId_cat']].head()
train.drop(columns = ['Id'], inplace=True)
train['matchType'].value_counts().plot.barh()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()



mapped_education = pd.Series(label_encoder.fit_transform(train['matchType']))

mapped_education.value_counts().plot.barh()

print(dict(enumerate(label_encoder.classes_)))
# Label Encoding

categorical_columns = train.columns[train.dtypes == 'object'].union(['matchType'])

for column in categorical_columns:

    train[column] = label_encoder.fit_transform(train[column])

train.head(10)
target = 'winPlacePerc'

features = list(train.columns)





y_train = np.array(train[target])

features.remove(target)

x_train = train[features]



print(x_train.shape,y_train.shape)
test['headshot_rate'] = test['headshotKills'] / test['kills']

test['headshot_rate'] = test['headshot_rate'].fillna(0)

test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')

test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)

test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)

test['maxPlaceNorm'] = test['maxPlace']*((100-train['playersJoined'])/100 + 1)

test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)

test['healsandboosts'] = test['heals'] + test['boosts']



# Turn groupId and match Id into categorical types

test['groupId'] = test['groupId'].astype('category')

test['matchId'] = test['matchId'].astype('category')



# Get category coding for groupId and matchID

test['groupId_cat'] = test['groupId'].cat.codes

test['matchId_cat'] = test['matchId'].cat.codes





categorical_columns = test.columns[test.dtypes == 'object'].union(['matchType'])

for column in categorical_columns:

    test[column] = label_encoder.fit_transform(test[column])



# Remove irrelevant features from the test set

x_test = test[features].copy()



# Fill NaN with 0 (temporary)

x_test.fillna(0, inplace=True)



print(x_test.shape)
x_test.head()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 1)
rf = RandomForestRegressor(n_estimators = 80, min_samples_leaf = 3, max_depth = 26, max_features = 0.5,

                          n_jobs = -1)
x_train.head()
x_train.shape
%%time

rf.fit(x_train, y_train)
print('mae train: ', mean_absolute_error(rf.predict(x_train), y_train))

print('mae val: ', mean_absolute_error(rf.predict(x_val), y_val))
%%time

pred = rf.predict(x_test)

test['winPlacePerc'] = pred

submission = test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)