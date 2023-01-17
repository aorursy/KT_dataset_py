import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os
import sys

sns.set_style('darkgrid')
sns.set_palette('bone')
pd.options.display.float_format = '{:,.3f}'.format

print(os.listdir('../input/pubg-finish-placement-prediction'))
pubg_path = '../input/pubg-finish-placement-prediction'
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
train = pd.read_csv(pubg_path + '/train_V2.csv')
train = reduce_mem_usage(train)
test = pd.read_csv(pubg_path + '/test_V2.csv')
test = reduce_mem_usage(test)
print(train.shape, test.shape)
train['winPlacePerc'].fillna(0, inplace = True)
all_data = train.append(test, sort=False).reset_index(drop = True)
del train, test
gc.collect()
all_data['HeadAvg'] = all_data['kills'] / all_data['headshotKills']
all_data['TotalHeals'] = all_data['boosts'] + all_data['heals'] + all_data['revives']
all_data['Yeopo'] = all_data['kills'] / all_data['killStreaks']
all_data['Distance'] = all_data['swimDistance'] + all_data['walkDistance'] + all_data['rideDistance']
all_data['KillContribution'] = all_data['kills']/ all_data['damageDealt']

all_data.drop(['headshotKills','boosts','heals','revives','killStreaks','swimDistance','walkDistance',
              'rideDistance','damageDealt'], axis = 1, inplace =True)

def fillInf(df, val):
    numcols = df.select_dtypes(include = 'number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols:
        df[c].fillna(val, inplace = True)
    
fillInf(all_data,0)
all_data.loc[all_data['maxPlace'] == 80]
col = ['HeadAvg','TotalHeals','Yeopo','Distance','KillContribution']
group = all_data.groupby(['matchId','groupId','matchType'])
match = all_data.groupby('matchId')


match_data = pd.concat([
    match.size().to_frame('m.players'),
    match[col].sum().rename(columns = lambda s : 'm.sum.' + s),
    match[col].max().rename(columns = lambda s : 'm.max.' + s),
    match[col].mean().rename(columns = lambda s : 'm.mean.' + s)
], axis = 1).reset_index()    

reduce_mem_usage(match_data)

match_data.head()

group_data = pd.concat([
    group.size().to_frame('g.player'),
    group[col].sum().rename(columns = lambda x : 'g.sum.' + x),
    group[col].max().rename(columns = lambda x : 'g.max.' + x),
    group[col].mean().rename(columns = lambda x : 'g.mean.' + x)
], axis = 1).reset_index()

reduce_mem_usage(group_data)
group_data.head()
data = pd.merge(match_data, group_data)
fillInf(data, 0)
data
all_data = pd.merge(all_data, data)
all_data['PlayerTime'] = all_data['m.players'] / all_data['matchDuration']
all_data['enemyPlayer'] = all_data['m.players'] - all_data['g.player']
all_data['SavePlayer'] = all_data['enemyPlayer'] - all_data['kills']

cols = ['PlayerTime', 'enemyPlayer', 'SavePlayer']
group = all_data.groupby(['matchId','groupId'])

group_data = pd.concat([
    group[cols].mean().rename(columns = lambda x : 'mean.' + x),
    group[cols].sum().rename(columns = lambda x : 'sum.' + x),
    
], axis = 1).reset_index()

all_data = pd.merge(all_data, group_data)
reduce_mem_usage(all_data)
all_data.columns
all_data.drop(['roadKills','teamKills','vehicleDestroys'], axis = 1, inplace =True)
idea = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis = 1).reset_index(drop=True)
all_data = pd.get_dummies(all_data, columns = ['matchType'])
all_data.drop(['Id'], axis =1, inplace =True)
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop= True)

X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis = 1).reset_index(drop=True)


del all_data
gc.collect()

Y_train = X_train.pop('winPlacePerc')
X_test_grp = X_test[['matchId','groupId']].copy()

X_train.drop(['matchId','groupId'], axis = 1, inplace = True)
X_test.drop(['matchId','groupId'], axis = 1, inplace =True)


X_train_cols = X_train.columns

print(X_train.shape, X_test.shape)
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, PReLU
from keras.models import load_model
from keras.models import Sequential

def creatModel():
    model = Sequential()
    model.add(Dense(512, kernel_initializer = 'he_normal', input_dim = X_train.shape[1], activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, shared_axes = None))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1, kernel_initializer = 'he_normal'))
    model.add(PReLU(alpha_initializer = 'zeros', alpha_regularizer=None, shared_axes = None))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    optimizer = optimizers.Adam(lr = 0.005)
    model.compile(optimizer = optimizer, loss='mse', metrics = ['mae'])
    
    return model
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):
    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule, verbose)

lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.97, step_size=1, verbose=1)
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', mode='min', patience=10, verbose=1)
from sklearn import preprocessing
np.random.seed(42)

scaler = preprocessing.StandardScaler().fit(X_train.astype(float))
X_train = scaler.transform(X_train.astype(float))
X_test = scaler.transform(X_test.astype(float))


model = creatModel()
history = model.fit(
        X_train, Y_train,
        epochs=200,
        batch_size=2**15,
        validation_split=0.2,
        callbacks=[lr_sched, early_stopping],
        verbose=2)
pred = model.predict(X_test).ravel()
submission = pd.DataFrame({
    'Id' : idea['Id'],
    'winPlacePerc' : pred
})

submission.to_csv('submission.csv', index = False)