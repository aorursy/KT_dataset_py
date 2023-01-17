import pandas as pd
import numpy as np
trainRaw = pd.read_csv("../input/train_V2.csv")
testRaw = pd.read_csv("../input/test_V2.csv")
trainRaw[trainRaw['winPlacePerc'].isnull()]
train = trainRaw.drop(2744604, axis=0)
train.shape
train['headshotRate'] = train['headshotKills'] / train['kills']
train['headshotRate'] = train['headshotRate'].fillna(0)
groupMax = train[['groupId', 'assists', 'walkDistance', 'kills', 'DBNOs']].groupby('groupId').max()
groupMin = train[['groupId', 'assists', 'walkDistance', 'kills', 'DBNOs']].groupby('groupId').min()
groupMax = groupMax.rename(columns = {'assists': 'assistsMax', 'walkDistance': 'walkDistanceMax', 'kills': 'killsMax', 'DBNOs': 'DBNOsMax'})
groupMin = groupMin.rename(columns = {'assists': 'assistsMin', 'walkDistance': 'walkDistanceMin', 'kills': 'killsMin', 'DBNOs': 'DBNOsMin'})
groupMax.head()
groupMin.head()
dataAll = pd.merge(train, groupMax, on = 'groupId')
dataAll = pd.merge(dataAll, groupMin, on = 'groupId')
dataAll.shape
dataAll.head()
dataAll.tail()
# matchRanks = train[['matchId', 'assists', 'boosts', 'damageDealt', 'heals', 'killStreaks', 'longestKill', 'walkDistance', \
#      'swimDistance', 'rideDistance']].groupby('matchId').rank(ascending = False)
# matchRanks.head()
# matchRanks = matchRanks.rename(columns = {'assists': 'assistsPlace', 'boosts': 'boostsPlace', 
#                                          'damageDealt': 'damageDealtPlace', 
#                                          'heals': 'heals', 'killStreaks': 'killStreaksPlace', 
#                                          'longestKill': 'longestKillPlace', 'walkDistance': 'walkDistancePlace', 
#                                          'swimDistance': 'swimDistancePlace', 'rideDistance': 'rideDistancePlace'})
# matchRanks['matchId'] = train['matchId']
# matchRanks.head()
# matchRanks.tail()
# matchRanks.shape
# dataAll = pd.concat([dataAll.reset_index(drop=True),matchRanks.reset_index(drop=True)], axis=1)
dataAll.shape
dataAll.tail()
dataAll.head()
features = pd.get_dummies(dataAll.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis = 1))
labels = np.array(dataAll['winPlacePerc'])
labels[np.isnan(labels)]
features.head()
print('features', features.shape, 'labels', labels.shape)
from sklearn.ensemble import RandomForestRegressor
RFmodel = RandomForestRegressor(n_estimators=80, random_state=1937, n_jobs=3, min_samples_leaf=3, max_features='sqrt')
print('start training')
RFmodel.fit(features, labels)
test = testRaw.copy()
test['headshotRate'] = test['headshotKills'] / test['kills']
test['headshotRate'] = test['headshotRate'].fillna(0)
groupMaxTest = test[['groupId', 'assists', 'walkDistance', 'kills', 'DBNOs']].groupby('groupId').max()
groupMinTest = test[['groupId', 'assists', 'walkDistance', 'kills', 'DBNOs']].groupby('groupId').min()
groupMaxTest = groupMaxTest.rename(columns = {'assists': 'assistsMax', 'walkDistance': 'walkDistanceMax', 'kills': 'killsMax', 'DBNOs': 'DBNOsMax'})
groupMinTest = groupMinTest.rename(columns = {'assists': 'assistsMin', 'walkDistance': 'walkDistanceMin', 'kills': 'killsMin', 'DBNOs': 'DBNOsMin'})
test = pd.merge(test, groupMaxTest, on = 'groupId')
test = pd.merge(test, groupMinTest, on = 'groupId')
# matchRanksTest = test[['matchId', 'assists', 'boosts', 'damageDealt', 'heals', 'killStreaks', 'longestKill', 'walkDistance', \
#      'swimDistance', 'rideDistance']].groupby('matchId').rank()
# matchRanksTest = matchRanksTest.rename(columns = {'assists': 'assistsPlace', 'boosts': 'boostsPlace', 
#                                           'damageDealt': 'damageDealtPlace', 
#                                           'heals': 'heals', 'killStreaks': 'killStreaksPlace', 
#                                           'longestKill': 'longestKillPlace', 'walkDistance': 'walkDistancePlace', 
#                                           'swimDistance': 'swimDistancePlace', 'rideDistance': 'rideDistancePlace'})
# test = pd.concat([test.reset_index(drop=True),matchRanksTest.reset_index(drop=True)], axis=1)
test.head()
features.shape
test = test.drop(['Id', 'groupId', 'matchId'],axis=1)
test = pd.get_dummies(test)
test.shape
features.head()
test.head()
prediction = RFmodel.predict(test)
submission = pd.DataFrame({"Id":testRaw['Id'], "winPlacePerc":prediction})
submission.head()
submission.tail()
submission.to_csv("submission.csv", index=False)