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
test_data = test_data[:, 0:3]
import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_features)

train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
from sklearn.ensemble import BaggingRegressor

model = BaggingRegressor()
model.fit(train_features, train_targets)
predictions = model.predict(test_features)
test_data_df['winPlacePercPred'] = predictions
test_data_preds = test_data_df.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
test_data_df['winPlacePerc'] = test_data_preds['winPlacePercPred']
import numpy as np
predictions = pd.DataFrame(np.transpose(np.array([test_data[:, 0], list(predictions)])))
predictions.columns = ['Id', 'winPlacePerc']
predictions['Id'] = np.int32(predictions['Id'])
predictions.to_csv('PUBG_preds2.csv', index=False)