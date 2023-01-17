import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
df_train = pd.read_csv('../input/train.csv').astype('float32')
df_test = pd.read_csv('../input/test.csv')
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
# https://www.kaggle.com/sidjhanji/pubg-kill-them-all
# Some Feature Engineering
df_train["distance"] = df_train["rideDistance"]+df_train["walkDistance"]+df_train["swimDistance"]
# df_train["healthpack"] = df_train["boosts"] + df_train["heals"]
df_train["skill"] = df_train["headshotKills"]+df_train["roadKills"]
df_test["distance"] = df_test["rideDistance"]+df_test["walkDistance"]+df_test["swimDistance"]
# df_test["healthpack"] = df_test["boosts"] + df_test["heals"]
df_test["skill"] = df_test["headshotKills"]+df_test["roadKills"]
"""
it is a team game, scores within the same group is same, so let's get the feature of each group
"""
df_train_size = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_test_size = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')

df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()

df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()
df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()

df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()
df_test_min = df_test.groupby(['matchId','groupId']).min().reset_index()

"""
although you are a good game player, 
but if other players of other groups in the same match is better than you, you will still get little score
so let's add the feature of each match
"""
df_train_match_mean = df_train.groupby(['matchId']).mean().reset_index()
df_test_match_mean = df_test.groupby(['matchId']).mean().reset_index()

df_train = pd.merge(df_train, df_train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del df_train_mean
del df_test_mean

df_train = pd.merge(df_train, df_train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del df_train_max
del df_test_max

df_train = pd.merge(df_train, df_train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del df_train_min
del df_test_min

df_train = pd.merge(df_train, df_train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
df_test = pd.merge(df_test, df_test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del df_train_match_mean
del df_test_match_mean

df_train = pd.merge(df_train, df_train_size, how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_size, how='left', on=['matchId', 'groupId'])
del df_train_size
del df_test_size

target = 'winPlacePerc'
train_columns = list(df_test.columns)
""" remove some columns """
train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
train_columns.remove("Id_mean")
train_columns.remove("Id_max")
train_columns.remove("Id_min")
train_columns.remove("Id_match_mean")
"""
in this game, team skill level is more important than personal skill level 
maybe you are a good player, but if your teammates is bad, you will still lose
so let's remove the features of each player, just select the features of group and match
"""
train_columns_new = []
for name in train_columns:
    if '_' in name:
        train_columns_new.append(name)
train_columns = train_columns_new    
print(train_columns)
# train_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
#                 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
#                 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 
#                 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
#                 'weaponsAcquired', 'winPoints']

X = df_train[train_columns]
Y = df_test[train_columns]
T = df_train[target]

del df_train

# X = np.array(X, dtype=np.float32)
# Y = np.array(Y, dtype=np.float32)
# T = np.array(T, dtype=np.float32)
x_train, x_test, t_train, t_test = train_test_split(X, T, test_size = 0.2, random_state = 1234)

# scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
scaler = preprocessing.QuantileTransformer().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
Y = scaler.transform(Y)

print("x_train", x_train.shape, x_train.min(), x_train.max())
print("x_test", x_test.shape, x_test.min(), x_test.max())
print("Y", Y.shape, Y.min(), Y.max())
model = Sequential()
model.add(Dense(512, kernel_initializer='he_normal', input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# model.summary()
# optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-4, nesterov=True)
optimizer = optimizers.Adam(lr=0.01, epsilon=1e-8, decay=1e-4, amsgrad=False)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule, verbose)

lr_sched = step_decay_schedule(initial_lr=0.1, decay_factor=0.9, step_size=1, verbose=1)
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode = 'min', patience=4, verbose=1)
# model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_mean_absolute_error', mode = 'min', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', mode = 'min',factor=0.5, patience=3, min_lr=0.0001, verbose=1)
history = model.fit(x_train, t_train, 
                 validation_data=(x_test, t_test),
                 epochs=30,
                 batch_size=32768,
                 callbacks=[lr_sched,early_stopping], 
                 verbose=1)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation mae values
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Abosulte Error')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
pred = model.predict(Y)
pred = pred.ravel()

# pred = (pred + 1) / 2
df_test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)


aux = df_test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
df_test = df_test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)
