import pandas as pd

train_data_df = pd.read_csv('../input/train.csv')
test_data_df = pd.read_csv('../input/test.csv')
new_train_data_df = train_data_df.groupby(['matchId','groupId'])['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints'].agg('mean').reset_index()
new_train_data_df['winPlacePerc'] = train_data_df['winPlacePerc']
new_train_data_df['winPlacePerc'] = train_data_df.groupby(['matchId', 'groupId'])['winPlacePerc'].agg('mean').reset_index()['winPlacePerc']
new_test_data_df = test_data_df.groupby(['matchId','groupId'])['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints'].agg('mean').reset_index()
import numpy as np
train_data = train_data_df.values
test_data = test_data_df.values

train_features = new_train_data_df.values[:, 2:24][0:np.int32(0.8*len(new_train_data_df))]
train_targets = new_train_data_df.values[:, 24][0:np.int32(0.8*len(new_train_data_df))]

val_features = new_train_data_df.values[:, 2:24][np.int32(0.8*len(new_train_data_df)):len(new_train_data_df)]
val_targets = new_train_data_df.values[:, 24][np.int32(0.8*len(new_train_data_df)):len(new_train_data_df)]

test_features = new_test_data_df.values[:, 2:24]
test_data = test_data[:, 0:3]
import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_features)

train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
import catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=20000, learning_rate=0.1, eval_metric='MAE')
model.fit(train_features, train_targets, eval_set=(val_features, val_targets))
predictions = model.predict(test_features)
new_test_data_df['winPlacePercPred'] = predictions
group_preds = new_test_data_df.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
group_preds = group_preds['winPlacePercPred']
dictionary = dict(zip(new_test_data_df['groupId'].values, group_preds))
new_preds = []

for i in test_data_df['groupId'].values:
    new_preds.append(dictionary[i])
test_data_df['winPlacePercPred'] = new_preds
test_data_df.sort_values(by=['matchId', 'groupId'])
import numpy as np
predictions = pd.DataFrame(np.transpose(np.array([test_data[:, 0], test_data_df['winPlacePercPred']])))
predictions.columns = ['Id', 'winPlacePerc']
predictions['Id'] = np.int32(predictions['Id'])
predictions.head(10)
predictions.to_csv('PUBG_preds3.csv', index=False)