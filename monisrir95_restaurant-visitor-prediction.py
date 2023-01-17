import glob, re

import numpy as np

import pandas as pd

from sklearn import *

from datetime import datetime

from sklearn.feature_selection import RFE
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
##DATA IMPORT
data = {

    'air_visit': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/sair_visit_data.csv'),

    'air_store': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/sair_store_info.csv'),

    'hpg_store': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/shpg_store_info.csv'),

    'air_reserve': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/sair_reserve.csv'),

    'hpg_reserve': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/shpg_reserve.csv'),

    'store_id': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/store_id_relation.csv'),

    'test': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/sample_submission.csv'),

    'holiday': pd.read_csv('C:/Users/Monisri/Documents/sem 2/CA675 ass 2/srecruit-restaurant-visitor-forecasting/sdate_info.csv').rename(columns={'calendar_date':'visit_date'})

    }

##DATA PREPROCESSING AND FEATURE EXTRACTION
data['hpg_reserve'].head(5)
## Merging hpg_reserve and store_relation_id

data['hpg_reserve'] = pd.merge(data['hpg_reserve'], data['store_id'], how='inner', on=['hpg_store_id'])

for df in ['air_reserve','hpg_reserve']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_days1', 'reserve_visitors':'reserve_visitors1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'reserve_days2', 'reserve_visitors':'reserve_visitors2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data[df].head(5)
data['air_visit']['visit_date'] = pd.to_datetime(data['air_visit']['visit_date'])

data['air_visit']['dow'] = data['air_visit']['visit_date'].dt.dayofweek

data['air_visit']['year'] = data['air_visit']['visit_date'].dt.year

data['air_visit']['month'] = data['air_visit']['visit_date'].dt.month

data['air_visit']['visit_date'] = data['air_visit']['visit_date'].dt.date

data['air_visit'].head(5)
data['test'].head(5)
data['test']['visit_date'] = data['test']['id'].map(lambda x: str(x).split('_')[2])

data['test']['air_store_id'] = data['test']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['test']['visit_date'] = pd.to_datetime(data['test']['visit_date'])

data['test']['dow'] = data['test']['visit_date'].dt.dayofweek

data['test']['year'] = data['test']['visit_date'].dt.year

data['test']['month'] = data['test']['visit_date'].dt.month

data['test']['visit_date'] = data['test']['visit_date'].dt.date
data['test'].head(5)
unique_stores = data['test']['air_store_id'].unique()

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
stores.tail(20)
tmp = data['air_visit'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

tmp = data['air_visit'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['air_visit'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['air_visit'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['air_visit'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
stores.head(20)
stores = pd.merge(stores, data['air_store'], how='left', on=['air_store_id']) 
stores.tail(20)
stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

lbl = preprocessing.LabelEncoder()

for i in range(4):

    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_area_name' +str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))



stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
stores.tail(20)
data['holiday'].head(5)
data['holiday']['visit_date'] = pd.to_datetime(data['holiday']['visit_date'])

data['holiday']['day_of_week'] = lbl.fit_transform(data['holiday']['day_of_week'])

data['holiday']['visit_date'] = data['holiday']['visit_date'].dt.date

data['holiday'].head(5)
train_data = pd.merge(data['air_visit'], data['holiday'], how='left', on=['visit_date']) 

test_data = pd.merge(data['test'], data['holiday'], how='left', on=['visit_date']) 

train_data = pd.merge(train_data, stores, how='left', on=['air_store_id','dow']) 

test_data = pd.merge(test_data, stores, how='left', on=['air_store_id','dow'])
for df in ['air_reserve','hpg_reserve']:

    train_data = pd.merge(train_data, data[df], how='left', on=['air_store_id','visit_date']) 

    test_data = pd.merge(test_data, data[df], how='left', on=['air_store_id','visit_date'])
train_data['id'] = train_data.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

train_data['reserve_sum'] = train_data['reserve_visitors1_x'] + train_data['reserve_visitors1_y']

train_data['reserve_mean'] = (train_data['reserve_visitors2_x'] + train_data['reserve_visitors2_y']) / 2

train_data['reserve_date_diff_mean'] = (train_data['reserve_days2_x'] + train_data['reserve_days2_y']) / 2
train_data.head(5)
test_data['reserve_sum'] = test_data['reserve_visitors1_x'] + test_data['reserve_visitors1_y']

test_data['reserve_mean'] = (test_data['reserve_visitors2_x'] + test_data['reserve_visitors2_y']) / 2

test_data['reserve_date_diff_mean'] = (test_data['reserve_days2_x'] + test_data['reserve_days2_y']) / 2

train_data.head(5)
train_data['date_int'] = train_data['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

test_data['date_int'] = test_data['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

train_data['var_max_latitude'] = train_data['latitude'].max() - train_data['latitude']

train_data['var_max_longitude'] = train_data['longitude'].max() - train_data['longitude']

test_data['var_max_latitude'] = test_data['latitude'].max() - test_data['latitude']

test_data['var_max_longitude'] = test_data['longitude'].max() - test_data['longitude']
train_data['lon_and_lat'] = train_data['longitude'] + train_data['latitude'] 

test_data['lon_and_lat'] = test_data['longitude'] + test_data['latitude']



lbl = preprocessing.LabelEncoder()

train_data['air_store_id2'] = lbl.fit_transform(train_data['air_store_id'])

test_data['air_store_id2'] = lbl.transform(test_data['air_store_id'])
train_data.head(25)
test_index = test_data['id']

#test_index_idx = test.index

train_data['visitors'] = np.log1p(train_data['visitors'].values)

test_data.head(5)
#DATA MODELLING
train_data['cat'] = 'train_data'

test_data['cat'] = 'test_data'

    

hot_enc_cols = ['air_genre_name0','air_genre_name1','air_genre_name2','air_genre_name3',

                 'air_area_name0','air_area_name1','air_area_name2','air_area_name3',

                 'air_genre_name','air_area_name','day_of_week','dow','year','month']



total_df = pd.concat((train_data,test_data), axis=0, ignore_index=False)

    

df_index = total_df.index

    

total_df = pd.get_dummies(total_df, columns=hot_enc_cols)



scale_columns = ['lon_and_lat','var_max_longitude','var_max_latitude','date_int','reserve_date_diff_mean','reserve_mean',

             'reserve_sum','reserve_days1_x','reserve_visitors1_x','reserve_days2_x','reserve_visitors2_x','reserve_days1_y','reserve_days2_y','reserve_visitors2_y','latitude','longitude',

             'count_observations','max_visitors','median_visitors','min_visitors','holiday_flg','reserve_visitors1_y',

              'mean_visitors','air_store_id2','date_int','var_max_longitude']



total_df = total_df.fillna(0)

from scipy.special import erfinv

def rank_gauss(x):

    N = x.shape[0]

    temp = x.argsort()

    rank_x = temp.argsort() / N

    rank_x -= rank_x.mean()

    rank_x *= 2 

    efi_x = erfinv(rank_x)

    efi_x -= efi_x.mean()

    return efi_x
for coln in scale_columns:

    total_df[coln] = rank_gauss(np.array(total_df[coln]))



total_df.index = df_index    

        

train_data = total_df[total_df['cat']=='train_data']

test_data = total_df[total_df['cat']=='test_data']    

    

drop_columns = ['cat','id', 'air_store_id', 'visit_date','visitors']
final = train_data['visitors']

train_data = train_data.drop(train_data[drop_columns],axis=1)

test_data = test_data.drop(test_data[drop_columns],axis=1)



print('Pre-processing completed')



print('train_data',train_data.shape)

print('test_data',test_data.shape)

print(final.shape)

from sklearn.model_selection import train_test_split

train_data, valid, y_train, y_valid = train_test_split(train_data, final, test_size=0.15, random_state=2018)
np.random.seed(0)

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.callbacks import EarlyStopping

from keras.models import load_model

from keras.optimizers import Adam

import h5py

from keras import backend

from tensorflow import set_random_seed

set_random_seed(99)

    

def rmsle(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: #check for negative values

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5



adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-05)

    

filepath = 'best_wt_recruit_new.hdf5'



callbacks = [EarlyStopping(monitor='val_loss', patience=30, verbose=0),

             ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, epsilon=1e-4, mode='min'),

             ModelCheckpoint(filepath=filepath,monitor='val_loss',save_best_only=True,mode='min')]



   

dropt = .25
model = Sequential()

model.add(Dense(250,activation='relu',input_shape=(train_data.shape[1],)))

model.add(Dropout(dropt)) 

model.add(Dense(30,activation='relu'))

model.add(Dropout(dropt))    

model.add(Dense(1,activation='relu'))



model.compile(loss='mse', optimizer=adam)
feed = model.fit(np.array(train_data), np.array(y_train), epochs=10000, batch_size=200, validation_data=(np.array(valid), np.array(y_valid)),

            verbose=2, callbacks=callbacks, shuffle=False)



model.load_weights('best_wt_recruit_new.hdf5')

    
prediction_value = np.expm1(model.predict(np.array(valid)))

score = rmsle(np.expm1(np.array(y_valid)), prediction_value)



print('score:',score)
prediction = np.expm1(model.predict(np.array(test_data)))



nn_df = pd.DataFrame(prediction,columns=['visitors'],index=test_index)

print(nn_df.head())

nn_df.to_csv('submit_nn.csv')



print('done')
selector = RFE(feed,10,step=1)

selector = selector.fit(train_data, y_train)

selector.support_ 

selector.ranking_     