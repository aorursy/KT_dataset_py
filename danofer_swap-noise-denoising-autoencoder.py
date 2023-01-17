### preprocessing

"""

code is taken from

tunguz - Surprise Me 2!

https://www.kaggle.com/tunguz/surprise-me-2/code

"""

import glob, re

import numpy as np

import pandas as pd

from sklearn import *

from datetime import datetime



import matplotlib.pyplot as plt
data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'as': pd.read_csv('../input/air_store_info.csv'),

    'hs': pd.read_csv('../input/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/air_reserve.csv'),

    'hr': pd.read_csv('../input/hpg_reserve.csv'),

    'id': pd.read_csv('../input/store_id_relation.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv'),

    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

    }



data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])



for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek

    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

    # Exclude same-week reservations

    data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']]

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])



data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['doy'] = data['tra']['visit_date'].dt.dayofyear

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['week'] = data['tra']['visit_date'].dt.week

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date



data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['doy'] = data['tes']['visit_date'].dt.dayofyear

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

data['tes']['week'] = data['tes']['visit_date'].dt.week

data['tes']['visit_date'] = data['tes']['visit_date'].dt.date



unique_stores = data['tes']['air_store_id'].unique()

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)



#sure it can be compressed...

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

#tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

#stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 



stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 

# NEW FEATURES FROM Georgii Vyshnia

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

lbl = preprocessing.LabelEncoder()

for i in range(10):

    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])



data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 



train = pd.merge(train, stores, how='inner', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



for df in ['ar','hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])



train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)



train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']

train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2

train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2



test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']

test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2

test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2



# NEW FEATURES FROM JMBULL

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

train['var_max_lat'] = train['latitude'].max() - train['latitude']

train['var_max_long'] = train['longitude'].max() - train['longitude']

test['var_max_lat'] = test['latitude'].max() - test['latitude']

test['var_max_long'] = test['longitude'].max() - test['longitude']



# NEW FEATURES FROM Georgii Vyshnia

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 

test['lon_plus_lat'] = test['longitude'] + test['latitude']



lbl = preprocessing.LabelEncoder()

train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])

test['air_store_id2'] = lbl.transform(test['air_store_id'])



col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]

train = train.fillna(-1)

test = test.fillna(-1)
# concatenate train data and test data and convert category data into one-hot

traintest = pd.concat([train, test])

value_col = ['holiday_flg','min_visitors','mean_visitors','median_visitors', # 'max_visitors',

             'count_observations',

'rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rv1_y','rs2_y','rv2_y','total_reserv_sum','total_reserv_mean',

'total_reserv_dt_diff_mean','date_int','var_max_lat','var_max_long','lon_plus_lat']



cat_col =  ['dow', 'year', 'month', 'air_store_id2', 'air_area_name', 'air_genre_name',

'air_area_name0', 'air_area_name1', 'air_area_name2', 'air_area_name3', 'air_area_name4',

'air_area_name5', 'air_area_name6', 'air_genre_name0', 'air_genre_name1',

'air_genre_name2', 'air_genre_name3', 'air_genre_name4']



nn_col = value_col + cat_col



dummys = []

for col in cat_col:

    dummy = pd.get_dummies(traintest[col], drop_first=False)

    dummys.append(dummy.as_matrix())

dummys = np.concatenate(dummys, axis=1)

value_scaler = preprocessing.MinMaxScaler() # normalization method of original is rank gaussian

for vcol in value_col:

    traintest[vcol] = value_scaler.fit_transform(traintest[vcol].values.astype(np.float64).reshape(-1, 1))

X_value = traintest[value_col].as_matrix()

X = np.concatenate([dummys, X_value], axis=1)
def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5
### train denoising autoencoder

from keras.layers import Input, Dense, BatchNormalization

from keras import Model



def get_DAE():

    # denoising autoencoder

    inputs = Input((X.shape[1],))

    x = Dense(800, activation='relu',bias=False)(inputs) # 1500 original

    x = BatchNormalization()(x) # new

    x = Dense(400, activation='relu', name="feature",bias=False)(x) # 1500 original

    x = BatchNormalization()(x) # new

    x = Dense(800, activation='relu',bias=False)(x) # 1500 original

    x = BatchNormalization()(x) # new

    outputs = Dense(X.shape[1], activation='relu')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse')



    return model





def x_generator(x, batch_size, shuffle=True):

    # batch generator of input

    batch_index = 0

    n = x.shape[0]

    while True:

        if batch_index == 0:

            index_array = np.arange(n)

            if shuffle:

                index_array = np.random.permutation(n)



        current_index = (batch_index * batch_size) % n

        if n >= current_index + batch_size:

            current_batch_size = batch_size

            batch_index += 1

        else:

            current_batch_size = n - current_index

            batch_index = 0



        batch_x = x[index_array[current_index: current_index + current_batch_size]]



        yield batch_x





def mix_generator(x, batch_size, swaprate=0.15, shuffle=True):

    # generator of noized input and output

    # swap 0.15% of values of data with values of another

    num_value = X.shape[1]

    num_swap = int(num_value * swaprate)

    gen1 = x_generator(x, batch_size, shuffle)

    gen2 = x_generator(x, batch_size, shuffle)

    while True:

        batch1 = next(gen1)

        batch2 = next(gen2)

        new_batch = batch1.copy()

        for i in range(batch1.shape[0]):

            swap_idx = np.random.choice(num_value, num_swap, replace=False)

            new_batch[i, swap_idx] = batch2[i, swap_idx]



        yield (new_batch, batch1)
# training

batch_size = 128

num_epoch = 50 # 1000 original

gen = mix_generator(X, batch_size)

dae = get_DAE()

dae.fit_generator(generator=gen,

                  steps_per_epoch=np.ceil(X.shape[0] / batch_size),

                  epochs=num_epoch,

                  verbose=3,)
### train NN with feature of DAE

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from  keras.regularizers import l2



def get_NN(DAE):

    l2_loss = l2(0.01) #l2(0.05) orig

    DAE.trainable = False

    x = dae.get_layer("feature").output

    x = Dropout(0.1)(x)

    x = Dense(1500, activation='relu', kernel_regularizer=l2_loss,bias=False)(x) # 4500 original

    x = BatchNormalization()(x) # new

    x = Dropout(0.4)(x) # .5 orig

    x = Dense(500, activation='relu', kernel_regularizer=l2_loss,bias=False)(x) # 1000 original

    x = BatchNormalization()(x) # new

    x = Dropout(0.4)(x)# .5 orig

    x = Dense(500, activation='relu', kernel_regularizer=l2_loss,bias=False)(x) # 1000 original

    x = BatchNormalization()(x) # new

    x = Dropout(0.4)(x)# .5 orig

    predictions = Dense(1, activation='relu', kernel_regularizer=l2_loss,bias=False)(x)



    model = Model(inputs=dae.input, outputs=predictions)

    model.compile(loss='mse',optimizer='adam')



    return model





def train_generator(x, y, batch_size, shuffle=True):

    batch_index = 0

    n = x.shape[0]

    while True:

        if batch_index == 0:

            index_array = np.arange(n)

            if shuffle:

                index_array = np.random.permutation(n)



        current_index = (batch_index * batch_size) % n

        if n >= current_index + batch_size:

            current_batch_size = batch_size

            batch_index += 1

        else:

            current_batch_size = n - current_index

            batch_index = 0



        batch_x = x[index_array[current_index: current_index + current_batch_size]]

        batch_y = y[index_array[current_index: current_index + current_batch_size]]



        yield batch_x, batch_y





def test_generator(x, batch_size, shuffle=False):

    batch_index = 0

    n = x.shape[0]

    while True:

        if batch_index == 0:

            index_array = np.arange(n)

            if shuffle:

                index_array = np.random.permutation(n)



        current_index = (batch_index * batch_size) % n

        if n >= current_index + batch_size:

            current_batch_size = batch_size

            batch_index += 1

        else:

            current_batch_size = n - current_index

            batch_index = 0



        batch_x = x[index_array[current_index: current_index + current_batch_size]]



        yield batch_x



        

def get_callbacks(save_path):

    save_checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss',

                                   patience=4,

                                   verbose=2,

                                   min_delta=1e-4,

                                   mode='min')

    Callbacks = [ save_checkpoint, early_stopping]

    return Callbacks
from sklearn.model_selection import KFold



batch_size = 128

num_epoch = 40 # 150 original

num_fold = 3 # 5 original



Y_train = np.log1p(train['visitors']).values

X_train = X[:Y_train.shape[0]]

X_test = X[Y_train.shape[0]:]



y_test = np.zeros([num_fold, X_test.shape[0]])

y_valid = np.zeros([X_train.shape[0]])



folds = list(KFold(n_splits=num_fold, shuffle=True, random_state=42).split(X_train))

for j, (ids_train_split, ids_valid_split) in enumerate(folds):

    print("fold", j+1, "==================")

    model = get_NN(dae)

    gen_train = train_generator(X_train[ids_train_split], Y_train[ids_train_split], batch_size)

    gen_val = train_generator(X_train[ids_valid_split], Y_train[ids_valid_split], batch_size, shuffle=False)

    gen_val_pred = test_generator(X_train[ids_valid_split], batch_size, shuffle=False)

    gen_test_pred = test_generator(X_test, batch_size, shuffle=False)



    # Fit model

    callbacks = get_callbacks("weight" + str(j) + ".hdf5")

    model.fit_generator(generator=gen_train,

                        steps_per_epoch=np.ceil(ids_train_split.shape[0] / batch_size),

                        epochs=num_epoch,

                        verbose=3,

#                         callbacks=callbacks,

                        validation_data=gen_val,

                        validation_steps=np.ceil(ids_valid_split.shape[0] / batch_size),

                        )

    # Predict on train, val and test

#     model.load_weights("weight" + str(j) + ".hdf5") # load best epoch weight

    y_valid[ids_valid_split] = model.predict_generator(generator=gen_val_pred,

                                        steps=np.ceil(ids_valid_split.shape[0] / batch_size))[:,0]

    y_test[j] = model.predict_generator(generator=gen_test_pred,

                                        steps=np.ceil(X_test.shape[0] / batch_size))[:,0]



score = RMSLE(y_valid, Y_train)

print("valid score", score)

y_test_mean = np.mean(y_test, axis=0)

id = pd.read_csv("../input/sample_submission.csv")['id']

submission = pd.DataFrame({'id': id, 'visitors': np.expm1(y_test_mean)})

submission.to_csv('submission_dae.csv', index=False)