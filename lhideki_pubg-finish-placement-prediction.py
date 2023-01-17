import pandas as pd
import os

train_filename = os.path.join('..', 'input', 'train_V2.csv')
train_df = pd.read_csv(train_filename, index_col=0)
train_df = train_df.fillna(0)
import keras

cleansed_df = train_df.drop(columns=['matchType', 'DBNOs'])
cleansed_df['distance'] = cleansed_df['walkDistance'] + cleansed_df['swimDistance'] + cleansed_df['rideDistance']
cleansed_df = cleansed_df.drop(columns=['walkDistance', 'swimDistance', 'rideDistance'])
for column in cleansed_df.columns:
    if column in ['groupId', 'matchId', 'winPlacePerc']:
        continue
    normalized_val = keras.utils.normalize(cleansed_df[column].values)[0]
    cleansed_df[column] = normalized_val
match_mean_df = cleansed_df[['matchId', 'groupId']].merge(cleansed_df.groupby(['matchId']).mean().reset_index(), on='matchId')
match_mean_df = match_mean_df.groupby(['matchId', 'groupId']).max().drop(columns='winPlacePerc')
display(match_mean_df.head())
train_grouped = cleansed_df.groupby(['matchId', 'groupId'])
labels_df = train_grouped.max()['winPlacePerc']

features_max_df = train_grouped.max().drop(columns=['winPlacePerc'])
features_min_df = train_grouped.min().drop(columns=['winPlacePerc'])
features_mean_df = train_grouped.mean().drop(columns=['winPlacePerc'])
import numpy as np

features_len = len(features_max_df) - 1
train_len = int(features_len * 0.9)

train_match_mean = match_mean_df[:train_len]
train_features_max = features_max_df[:train_len]
train_features_min = features_min_df[:train_len]
train_features_mean = features_mean_df[:train_len]
train_labels = labels_df[:train_len]

valid_match_mean = match_mean_df[train_len:]
valid_features_max = features_max_df[train_len:]
valid_features_min = features_min_df[train_len:]
valid_features_mean = features_mean_df[train_len:]
valid_labels = labels_df[train_len:]

print(train_match_mean.shape)
print(train_features_min.shape)
print(train_features_max.shape)
print(train_features_mean.shape)
print(train_labels.shape)
print(valid_match_mean.shape)
print(valid_features_min.shape)
print(valid_features_max.shape)
print(valid_features_mean.shape)
print(valid_labels.shape)
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import layers
from keras.layers import concatenate, add, subtract, Reshape, dot
from keras import Input, Model

input_tensor_match_mean = Input(train_match_mean.values[0].shape)
input_tensor_max = Input(train_features_max.values[0].shape)
input_tensor_min = Input(train_features_min.values[0].shape)
input_tensor_mean = Input(train_features_mean.values[0].shape)

x1 = subtract([input_tensor_max, input_tensor_match_mean])
x2 = subtract([input_tensor_min, input_tensor_match_mean])
x3 = subtract([input_tensor_mean, input_tensor_match_mean])
x4 = input_tensor_match_mean
x5 = concatenate([x1, x2, x3, x4])
x5 = Dense(32, activation='relu')(x5)
x5 = Dense(32, activation='relu')(x5)
last = x5

output_tensor = Dense(1)(last)

model = Model([input_tensor_match_mean, input_tensor_max, input_tensor_min, input_tensor_mean], output_tensor)
model.compile(loss='mae', optimizer='nadam', metrics=['acc', 'mae'])
model.summary()
model_filename = 'model.h5'

epoch = 100
history = model.fit([train_match_mean, train_features_max, train_features_min, train_features_mean],
          train_labels,
          epochs = epoch,
          batch_size = 2048,
          validation_split = 0.1,
          callbacks = [
              LearningRateScheduler(lambda e: np.linspace(0.01, 0.0001, epoch)[e]),
              EarlyStopping(patience=10, monitor='val_mean_absolute_error'),
              ModelCheckpoint(model_filename, monitor='val_mean_absolute_error', save_best_only=True)
          ])
history_df = pd.DataFrame(history.history)
display(history_df)
history_df.to_csv('history.csv')
import keras

model = keras.models.load_model(model_filename)
predicted_valid_labels = model.predict([
    valid_match_mean,
    valid_features_max,
    valid_features_min,
    valid_features_mean,
]).reshape(-1)
predicted_valid_labels = predicted_valid_labels.clip(0, 1)
valid_predicted_labels = valid_labels.values.reshape(-1)
%matplotlib inline
import seaborn as sns

predicted_df = pd.DataFrame()
predicted_df['valid_labels'] = [ max(101 - int(valid_label * 100), 0) for valid_label in valid_predicted_labels]
predicted_df['predicted_labels'] = [ max(101 - int(predicted_valid_label * 100), 0) for predicted_valid_label in predicted_valid_labels]
sns.jointplot(data=predicted_df, x='valid_labels', y='predicted_labels', kind='hex')
import pandas as pd
import os

test_filename = os.path.join('..', 'input', 'test_V2.csv')
test_df = pd.read_csv(test_filename, index_col=0)
test_df = test_df.fillna(0)
import keras

test_features_df = test_df.drop(columns=['matchType', 'DBNOs'])
test_features_df['distance'] = test_features_df['walkDistance'] + test_features_df['swimDistance'] + test_features_df['rideDistance']
test_features_df = test_features_df.drop(columns=['walkDistance', 'swimDistance', 'rideDistance'])
for column in test_features_df.columns:
    if column in ['groupId', 'matchId', 'winPlacePerc']:
        continue
    normalized_val = keras.utils.normalize(test_features_df[column].values)[0]
    test_features_df[column] = normalized_val
test_match_mean_df = test_features_df[['matchId', 'groupId']].merge(test_features_df.groupby(['matchId']).mean().reset_index(), on='matchId')
test_match_mean_df = test_match_mean_df.groupby(['matchId', 'groupId']).max()
display(test_match_mean_df.head())
test_grouped = test_features_df.groupby(['matchId', 'groupId'])
test_features_max_df = test_grouped.max()
test_features_min_df = test_grouped.min()
test_features_mean_df = test_grouped.mean()
import os
import keras
import tensorflow as tf
import keras.backend as K

test_predicted_labels = model.predict(
    [test_match_mean_df,
     test_features_max_df,
     test_features_min_df,
     test_features_mean_df
    ]).reshape(-1)
test_predicted_labels = test_predicted_labels.clip(0, 1)
test_predicted = test_features_max_df
test_predicted['winPlacePerc'] = test_predicted_labels
test_predicted = test_predicted.merge(test_features_df.reset_index()[['matchId', 'groupId', 'Id']], on=['matchId', 'groupId'])
test_predicted.head()
import datetime

datestr = datetime.datetime.now().strftime('%Y%m%dT%H%M')

results_df = test_predicted[['Id', 'winPlacePerc']].sort_values('Id')

display(results_df.head())
results_df.to_csv(f'submission_{datestr}.csv', index=False)