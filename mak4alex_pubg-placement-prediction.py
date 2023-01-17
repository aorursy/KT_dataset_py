import os
import sys
import numpy as np
np.random.seed(1)

import pandas as pd
pd.set_option("display.max_rows", 300)

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.losses import MAE
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gc
gc.enable()

y_label = 'winPlacePerc'
x_categorical = ['matchType']
standard_modes = ['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp']
x_numeric_columns = [
    'DBNOs', 'assists', 'boosts', 'damageDealt', 'headshotKills', 'heals', 'killPlace', 'killPoints',
    'killStreaks', 'kills', 'longestKill', 'matchDuration', 'rankPoints', 'revives', 'rideDistance',
    'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',
    'winPoints', 'numGroups', 'maxPlace'
    
]
idx_columns = ['Id', 'matchId', 'groupId']

epochs = 1
batch_size = 2 ** 13
train_nrows = None
validation_split=0.0

print("batch size: {}".format(batch_size))
print("Input dir ../input: {}".format(os.listdir("../input")))
def optimize_memory(df):
    before_mem_mb = df.memory_usage().sum() / 1048576
    
    for col in df.columns:
        col_type = df[col].dtype
        if  col_type != object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int' and col not in x_numeric_columns:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)       
        elif str(col_type) != 'category':
            df[col] = df[col].astype('category')
    
    after_mem_mb = df.memory_usage().sum() / 1048576
    print("Memory size before: {:.2f} mb, after: {:.2f} mb".format(before_mem_mb, after_mem_mb))
    gc.collect()
    
    return df

def load_csv(path, nrows=None, dtype=None):
    df = pd.read_csv(path, dtype=dtype, nrows=nrows)
    df = optimize_memory(df)
    return df

str_to_int_dict = {}
int_to_str_dict = {}

def str_to_int(string):
    if string in str_to_int_dict:
        return str_to_int_dict[string]
    else:
        index = len(str_to_int_dict) + 1
        str_to_int_dict[string] = index
        int_to_str_dict[index]  = string
        return index


def optimize_data(data):
    for col in idx_columns:
        data[col] = data[col].map(lambda x: str_to_int(x)).astype(np.int64, inplace=True)

    data['matchType'].cat.add_categories(['custom'], inplace=True)
    data.loc[~data['matchType'].isin(standard_modes), ['matchType']] = 'custom'
    data['matchType'].cat.remove_unused_categories(inplace=True)


source_dtypes = { 
    'Id': 'category', 'groupId': 'category', 'matchId': 'category', 'assists': 'float16',
    'boosts': 'float16', 'damageDealt': 'float16', 'DBNOs': 'float16', 'headshotKills': 'float16',
    'heals': 'float16', 'killPlace': 'float16', 'killPoints': 'float16', 'kills': 'float16',
    'killStreaks': 'float16', 'longestKill': 'float16', 'matchDuration': 'float16', 'matchType': 'category',
    'maxPlace': 'float16', 'numGroups': 'float16', 'rankPoints': 'float16', 'revives': 'float16',
    'rideDistance': 'float16', 'roadKills': 'float16', 'swimDistance': 'float16', 'teamKills': 'float16',
    'vehicleDestroys': 'float16', 'walkDistance': 'float16', 'weaponsAcquired': 'float16', 'winPoints': 'float16',
    'winPlacePerc': 'float16' 
}
source_data    = load_csv('../input/train_V2.csv', nrows=train_nrows) 
if train_nrows is None or train_nrows > 2744603:
    source_data.drop(2744604, inplace=True)

optimize_data(source_data)
source_y = source_data[y_label]
print("Data sizes: train {}".format(len(source_data)))
print(sys.getsizeof(source_data) * 1e-6)
display(source_data.sample(10))
display(source_data.info())
display(source_data.describe())
display(source_data.isna().sum())
def add_features(df):
#     df = df.assign(totalDistance=lambda x: x['rideDistance'] + x['walkDistance'] + x['swimDistance'])
#     df = df.assign(totalPoints=lambda x: x['killPoints'] + x['winPoints'])
#     df = df.assign(totalMedicine=lambda x: x['heals'] + x['boosts'])
#     df = df.assign(headshotRate=lambda x: x['headshotKills'] / (x['kills'] + 0.00001 ))
    
    
#     for new_column in ['totalDistance', 'totalPoints', 'totalMedicine', 'headshotRate']:
#         if new_column not in x_numeric_columns:
#             x_numeric_columns.append(new_column)
    
    print('New columns added')
    match_group_by = df.groupby(['matchId', 'groupId'])[x_numeric_columns]   
    print('match_group_by done')
    
    match_group_mean = match_group_by.mean()
    print('match_group_mean done')
    match_group_mean_rank = match_group_mean.groupby('matchId')[x_numeric_columns].rank(pct=True)
    df = pd.merge(df, match_group_mean.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_mean"], copy=False)
    df = pd.merge(df, match_group_mean_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_mean_rank"], copy=False)
    del match_group_mean, match_group_mean_rank; gc.collect()
    print('all match_group_mean done')
    
    match_group_median = match_group_by.median()
    match_group_median_rank = match_group_median.groupby('matchId')[x_numeric_columns].rank(pct=True)
    df = pd.merge(df, match_group_median.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_median"], copy=False)
    df = pd.merge(df, match_group_median_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_median_rank"], copy=False)
    del match_group_median, match_group_median_rank; gc.collect()
    print('match_group_median done')
    
    df = optimize_memory(df)
    
#     match_group_std  = match_group_by.std()  
#     match_group_std.replace([np.inf, -np.inf], np.nan, inplace=True)
#     match_group_std.fillna(0, inplace=True)    
#     match_group_std_rank = match_group_std.groupby('matchId')[x_numeric_columns].rank(pct=True)
#     df = pd.merge(df, match_group_std.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_std"], copy=False)
#     df = pd.merge(df, match_group_std_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_std_rank"], copy=False)
#     del match_group_std, match_group_std_rank; gc.collect()
    
    match_group_size = match_group_by.size().reset_index(name='group_size')  
    df = pd.merge(df, match_group_size, how='left', on=['matchId', 'groupId'], copy=False)
    del match_group_size; gc.collect()
    
    match_group_max  = match_group_by.max()
    match_group_max_rank = match_group_max.groupby('matchId')[x_numeric_columns].rank(pct=True)
    df = pd.merge(df, match_group_max.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_max"], copy=False)
    df = pd.merge(df, match_group_max_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_max_rank"], copy=False)
    del match_group_max, match_group_max_rank; gc.collect()
    
    match_group_min  = match_group_by.min()
    match_group_min_rank = match_group_min.groupby('matchId')[x_numeric_columns].rank(pct=True)
    df = pd.merge(df, match_group_min.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_min"], copy=False)
    df = pd.merge(df, match_group_min_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_min_rank"], copy=False)
    del match_group_min, match_group_min_rank; gc.collect()
    
    df = optimize_memory(df)
    
    match_group_sum  = match_group_by.sum()
    match_group_sum.replace([np.inf, -np.inf], np.nan, inplace=True)
    match_group_sum.fillna(0, inplace=True) 
    match_group_sum_rank = match_group_sum.groupby('matchId')[x_numeric_columns].rank(pct=True)
    df = pd.merge(df, match_group_sum.reset_index(), on=['matchId', 'groupId'], how='left', suffixes=["", "_group_sum"], copy=False)
    df = pd.merge(df, match_group_sum_rank, on=['matchId', 'groupId'], how='left', suffixes=["", "_group_sum_rank"], copy=False)
    del match_group_sum, match_group_sum_rank, match_group_by; gc.collect()
    
    df = optimize_memory(df)
    print('Group matches done')
    match_by = df.groupby(['matchId'])[x_numeric_columns]

    match_mean = match_by.mean().reset_index()
    df = pd.merge(df, match_mean, on=['matchId'], how='left', suffixes=["", "_match_mean"], copy=False)
    del match_mean; gc.collect()
    print('Matches mean done')
    
    match_median = match_by.median().reset_index()
    df = pd.merge(df, match_median, on=['matchId'], how='left', suffixes=["", "_match_median"], copy=False)
    del match_median; gc.collect()
    print('Matches median done')
    
#     match_std  = match_by.std().reset_index()
#     match_std.replace([np.inf, -np.inf], np.nan, inplace=True)
#     match_std.fillna(0, inplace=True)    
#     df = pd.merge(df, match_std, on=['matchId'], how='left', suffixes=["", "_match_std"], copy=False)
#     del match_std; gc.collect()
#     print('Matches std done')
    
    match_max  = match_by.max().reset_index()
    df = pd.merge(df, match_max, on=['matchId'], how='left', suffixes=["", "_match_max"], copy=False)
    del match_max; gc.collect()
    print('Matches max done')
    
    match_min  = match_by.min().reset_index()
    df = pd.merge(df, match_min, on=['matchId'], how='left', suffixes=["", "_match_min"], copy=False)
    del match_min, match_by; gc.collect()
    print('Matches min done')
    
    return df


def prepare_data(data, scaler=None):
    data = add_features(data)
    print("Added more features")
    columns = [col for col in data.columns if col.split('_')[0] in x_numeric_columns] + ['group_size']
        
    drop_columns = list(set(data.columns) - set(columns))
    print("Columns to drop: {}".format(drop_columns))
        
    match_type_cats = pd.get_dummies(data[x_categorical])
    data = pd.merge(data, match_type_cats, left_index=True, right_index=True, how='left', suffixes=["", ""], copy=False)
    del match_type_cats
    
    data.drop(columns=drop_columns, inplace=True)
    gc.collect()
    
    print('Start scaler')
    
    data_len = len(data)
    step = 50000
    batches = np.arange(0, data_len, step)
    
    if scaler is None:
        scaler = StandardScaler(copy=False)
        print('start scaler fitting')
        for batch in batches:
            print("scaler fit batch {}".format(batch))
            scaler.partial_fit(data.loc[batch:batch+step, columns].astype(np.float32))

    gc.collect()
    print('Start transform')
    for batch in batches:
        print("scaler transform batch {}".format(batch))
        data.loc[batch:batch+step, columns] = scaler.transform(data.loc[batch:batch+step, columns]) 
        gc.collect()   

    data = optimize_memory(data)
    return (data, scaler)


def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    dense_1 = Dense(512, activation='relu')(input_layer)
    dense_2 = Dense(256, activation='relu')(dense_1)
    dense_3 = Dense(256, activation='relu')(dense_2)
    output_layer = Dense(1, activation='sigmoid')(dense_3)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae')
    return model


def normalize_predictions(predicted, max_place):
    predicted_place = max_place - predicted * (max_place - 1)
    predicted_place = predicted_place.round()
    norm_predicted  = (max_place - predicted_place) / ( max_place - 1 + 0.0001 )
    return norm_predicted


class LRSchedulerByLoss(Callback):    
    def __init__(self, decay=0.95, tolerans=5, verbose=0):
        super(LRSchedulerByLoss, self).__init__()
        self.decay = decay
        self.tolerans = tolerans
        self.verbose = verbose
        self.loss_not_decrease_epochs = 0
        self.min_lost  = float('inf')
    
    def on_train_begin(self, logs=None):
         self.current_lr = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get('loss')

        if self.min_lost - current_loss > 0.0001:
            self.min_lost = current_loss
            self.loss_not_decrease_epochs = 0
            print('Improved')
        else:
            self.loss_not_decrease_epochs += 1
            print('Not improved')

        
        if self.loss_not_decrease_epochs > self.tolerans:
            self.loss_not_decrease_epochs = self.loss_not_decrease_epochs // 2
            self.current_lr = self.current_lr * self.decay
            K.set_value(self.model.optimizer.lr, self.current_lr)
            if self.verbose > 0:
                print('\nEpoch %05d: LRSchedulerByLoss setting lr to %s.' % (epoch + 1, self.current_lr))
        
train_x, scaler = prepare_data(source_data)
input_shape = (train_x.shape[1],)
print("Input shape {}".format(input_shape))
display(train_x.head())
display(train_x.info())
model = build_model(input_shape)
gc.collect()

early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='auto', restore_best_weights=True)

lr_scheduler_by_loss = LRSchedulerByLoss(verbose=1)
model.fit(train_x.values, source_y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[lr_scheduler_by_loss, early_stopping])

# 180000/180000 [==============================] - 3s 16us/step - loss: 0.0187 - val_loss: 0.0766
# 350 180000/180000 [==============================] - 4s 21us/step - loss: 0.0144 - val_loss: 0.0739
# train_solos = train_data[train_data['numGroups']>50]
# train_y_solos = train_y[train_solos.index]
# dev_solos = dev_data[dev_data['numGroups']>50]
# dev_y_solos = dev_y[dev_solos.index]

# solos = train_data[train_data['numGroups']>50]
# duos = train_data[(train_data['numGroups']>25) & (train_data['numGroups']<=50)]
# squads = train_data[train_data['numGroups']<=25]
# print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(
#     len(solos), 100*len(solos)/len(train_data), len(duos), 100*len(duos)/len(train_data), 
#     len(squads), 100*len(squads)/len(train_data),))

# train_x_solos, dev_x_solos = prepare_data(train_solos, dev_solos)
# input_shape = (train_x_solos.shape[1],)
# model_solos = build_model(input_shape)
# model_solos.fit(train_x_solos, train_y_solos, batch_size=4096, epochs=25, validation_data=(dev_x_solos, dev_y_solos))
# train_y_solos_predicted = model.predict(train_x_solos)
# train_y_solos_predicted = pd.Series(train_y_solos_predicted.reshape(-1), index=train_y_solos.index, name='winPlacePerc')
# display(train_y_solos.describe())
# display(train_y_solos_predicted.describe())
# plt.figure()
# diff = train_y_solos.subtract(train_y_solos_predicted)
# sns.distplot(diff)
# diff.describe()
del source_data, train_x
gc.collect()

submit_data = load_csv('../input/test_V2.csv')
optimize_data(submit_data)


submit_data_x, _  = prepare_data(submit_data, scaler)
submit_data['winPlacePercPred'] = model.predict(submit_data_x, batch_size=batch_size)
submit_data['winPlacePercPred'] = np.clip(submit_data['winPlacePercPred'], a_min=0, a_max=1)

results = submit_data.groupby(['matchId', 'groupId'])['winPlacePercPred'].mean().groupby('matchId').rank(pct=True).reset_index()
results.columns = ['matchId','groupId', y_label]
submit_data = submit_data.merge(results, how='left', on=['matchId','groupId'])

#submit_data[y_label] = normalize_predictions(submit_data[y_label], submit_data['maxPlace'])
submission = submit_data[['Id', y_label]]
submission['Id'] = submission['Id'].map(lambda x: int_to_str_dict[x])
display(submission.head())
submission.to_csv('./submission.csv', index=False)