import pandas as pd

train_data_df = pd.read_csv('../input/train.csv')
test_data_df = pd.read_csv('../input/test.csv')
train_data = train_data_df.values
test_data = test_data_df.values

train_features = train_data[:, 3:25][0:3485869]
train_targets = train_data[:, 25][0:3485869]

val_features = train_data[:, 3:25][3485869:4357336]
val_targets = train_data[:, 25][3485869:4357336]

test_features = test_data[:, 3:25]
import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_features)

train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
import catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=10000, learning_rate=0.1, eval_metric='MAE', max_depth=8)
model.fit(train_features, train_targets, eval_set=(val_features, val_targets))
predictions = list(model.predict(test_features))
test_data_df['winPlacePercPred'] = predictions
test_data_preds = test_data_df.groupby(['matchId','groupId', 'Id'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
test_data_df['winPlacePerc'] = test_data_preds['winPlacePercPred']
predictions = list(test_data_df['winPlacePerc'])
import numpy as np
ids = list(np.int32(test_data_preds.values[:, 2]))
submission = pd.DataFrame(np.transpose(np.array([ids, predictions])))
submission.columns = ['Id', 'winPlacePerc']
submission['Id'] = np.int32(submission['Id'])
submission = submission.sort_values(by=['Id'])
submission.head()
submission.to_csv('PUBG_preds.csv', index=False)