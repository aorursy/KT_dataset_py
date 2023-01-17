import pandas as pd

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data = train_data.values
test_data = test_data.values

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
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(n_estimators=20, learning_rate=0.1, loss='square')
model.fit(train_features, train_targets)
predictions = list(model.predict(test_features))
predictions
import numpy as np
ids = list(np.int32(test_data[:, 0]))
submission = pd.DataFrame(np.transpose(np.array([ids, predictions])))
submission.columns = ['Id', 'winPlacePerc']
submission['Id'] = np.int32(submission['Id'])
submission.to_csv('PUBG_preds.csv', index=False)