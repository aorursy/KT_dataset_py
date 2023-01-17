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
import numpy as np

train_targets = np.reshape(train_targets, (len(train_targets), 1))
val_targets = np.reshape(val_targets, (len(val_targets), 1))
import sklearn
from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler(feature_range=(-1, 1)).fit(train_features)
# scaler2 = MinMaxScaler(feature_range=(-1, 1)).fit(train_targets)

train_features = scaler1.transform(train_features)
# train_targets = scaler2.transform(train_targets)

val_features = scaler1.transform(val_features)
# val_targets = scaler2.transform(val_targets)

test_features = scaler1.transform(test_features)
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import L1L2

model = Sequential()

model.add(Dense(20, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.175))

model.add(Dense(30, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.175))

model.add(Dense(10, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.175))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='rmsprop')
model.fit(x=train_features, y=train_targets, validation_data=(val_features, val_targets), epochs=60, batch_size=10000)
# import xgboost

# model = xgboost.XGBRegressor(n_estimators=10, max_depth=6, objective='reg:tweedie')
# model.fit(train_features, train_targets)
# model.predict(train_features)
# train_targets
predictions = list(np.reshape(model.predict(test_features), (len(test_data))))
predictions
ids = list(np.int32(test_data[:, 0]))
submission = pd.DataFrame(np.transpose(np.array([ids, predictions])))
submission.columns = ['Id', 'winPlacePerc']
submission['Id'] = np.int32(submission['Id'])
submission.to_csv('PUBG_preds.csv', index=False)
