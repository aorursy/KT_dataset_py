import pandas as pd

import numpy as np

import os,glob

import lightgbm as lgb

file_list = []

for file in glob.glob("../input/*.csv"):

    file_list.append(file)

    print(file)

store_relation = pd.read_csv(file_list[1])

store_relation.head()
hpg_store_info = pd.read_csv(file_list[-2])

print(hpg_store_info.head())

hpg_store_info = to_radians(hpg_store_info,'longitude')



print(len(set(hpg_store_info['hpg_area_name'].values)))
hpg_store_info.at[0,'latitude'] = math.radians(hpg_store_info['latitude'][0])
hpg_reserve = pd.read_csv(file_list[5])

hpg_reserve.head()
air_store_info = pd.read_csv(file_list[-1])

print(air_store_info.head())

print(len(set(air_store_info['air_area_name'].values)))
air_reserve = pd.read_csv(file_list[2])

air_reserve.head()
air_visit = pd.read_csv(file_list[4])

air_visit.head()
date_data = pd.read_csv(file_list[0])

date_data.head()
from sklearn import preprocessing

import re

import numpy as np

import pandas as pd

import math

def to_radians(df,col_name):

    for i in range(len(df)):

        df.at[i,col_name] = math.radians(df[col_name][i])

    return df



def merge_dataset(data):

    # merge hpg reserve and id relation according to hpg store id

    data['hpg_reserve'] = pd.merge(data['hpg_reserve'],data['id_relation'],how='inner',on=['hpg_store_id'])

    for df in ['air_reserve','hpg_reserve']:

        data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

        data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

        data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (

            r['visit_datetime'] - r['reserve_datetime']).days,axis=1)

        tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[

            ['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'

                                                                                 , 'reserve_datetime_diff': 'rs1'

                                                                                 , 'reserve_visitors':'rv1'})

        tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[

            ['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date',

                                                                                  'reserve_datetime_diff': 'rs2',

                                                                                  'reserve_visitors':'rv2'})

        data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

    for df in ['train','test']:

        if df is 'test':

            data[df]['visit_date'] = data['test']['id'].map(lambda x: str(x).split('_')[2])

        data[df]['visit_date'] = pd.to_datetime(data[df]['visit_date'])

        data[df]['dow'] = data[df]['visit_date'].dt.dayofweek

        data[df]['year'] = data[df]['visit_date'].dt.year

        data[df]['month'] = data[df]['visit_date'].dt.month

        data[df]['visit_date'] = data[df]['visit_date'].dt.date

    data['test']['air_store_id'] = data['test']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

    unique_stores = data['test']['air_store_id'].unique()

    stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 

                                      'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, 

                                   ignore_index=True).reset_index(drop=True)

    for col in ['min_visitors','mean_visitors','median_visitors','max_visitors','count_observations']:

        temp = data['train'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={

            'visitors':col})

        stores = pd.merge(stores, temp, how='left', on=['air_store_id','dow']) 

    stores = pd.merge(stores, data['air_store'], how='left', on=['air_store_id']) 

    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

    

    encode_label = preprocessing.LabelEncoder()

    for i in range(10):

        stores['air_genre_name'+str(i)] = encode_label.fit_transform(

            stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

        stores['air_area_name'+str(i)] = encode_label.fit_transform(

            stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_genre_name'] = encode_label.fit_transform(stores['air_genre_name'])

    stores['air_area_name'] = encode_label.fit_transform(stores['air_area_name'])

    

    data['holiday']['visit_date'] = pd.to_datetime(data['holiday']['visit_date'])

    data['holiday']['day_of_week'] = encode_label.fit_transform(data['holiday']['day_of_week'])

    data['holiday']['visit_date'] = data['holiday']['visit_date'].dt.date

    train = pd.merge(data['train'], data['holiday'], how='left', on=['visit_date']) 

    test = pd.merge(data['test'], data['holiday'], how='left', on=['visit_date']) 



    train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

    test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



    for df in ['air_reserve','hpg_reserve']:

        train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

        test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])



    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)



    train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']

    train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2

    train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2



    test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']

    test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2

    test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)



    train = to_radians(train,'latitude')

    train = to_radians(train,'longitude')

    test = to_radians(test,'latitude')

    test = to_radians(test,'longitude')



    train['var_max_lat'] = train['latitude'].max() - train['latitude']

    train['var_max_long'] = train['longitude'].max() - train['longitude']

    test['var_max_lat'] = test['latitude'].max() - test['latitude']

    test['var_max_long'] = test['longitude'].max() - test['longitude']

    encoder2 = preprocessing.LabelEncoder()

    train['air_store_id2'] = encoder2.fit_transform(train['air_store_id'])

    test['air_store_id2'] = encoder2.transform(test['air_store_id'])

    

    return train,test



def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5
data = {

    'train':pd.read_csv('../input/air_visit_data.csv'),

    'air_store':pd.read_csv('../input/air_store_info.csv'),

    'hpg_store':pd.read_csv('../input/hpg_store_info.csv'),

    'air_reserve':pd.read_csv('../input/air_reserve.csv'),

    'hpg_reserve':pd.read_csv('../input/hpg_reserve.csv'),

    'id_relation':pd.read_csv('../input/store_id_relation.csv'),

    'test':pd.read_csv('../input/sample_submission.csv'),

    'holiday':pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'}),

}
train,test = merge_dataset(data)

print(train.head())

print(test.head())
import pickle

pickle.dump(train,open('preprocessed_train.pkl','wb'))

pickle.dump(test,open('preprocessed_test.pkl','wb'))
import pickle

import numpy as np

train = pickle.load(open('preprocessed_train.pkl','rb'))

test = pickle.load(open('preprocessed_test.pkl','rb'))
from sklearn import *

def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]

train = train.fillna(-1)

test = test.fillna(-1)
import lightgbm as lgbm



params = {

        'boosting_type': 'gbdt', 'objective': 'regression', 'nthread': -1, 'verbose': 1,

        'num_leaves': 700, 'learning_rate': 0.02, 'max_depth': -1,

        'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6,

        'reg_alpha': 1, 'reg_lambda': 0.001, 'metric': 'rmse',

        'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 20, 'scale_pos_weight': 1}

    

#kf = KFold(n_splits=5, shuffle=True, random_state=seed_val)

# pred_test_y = np.zeros(test[col].shape[0])



train_set = lgbm.Dataset(train[col], np.log1p(train['visitors'].values), silent=True)

model = lgbm.train(params, train_set=train_set, num_boost_round=100,feature_name=col)

print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), model.predict(train[col])))
print('Feature names:', model.feature_name())



# feature importances

print('Feature importances:', list(model.feature_importance()))

model.save_model('model.txt')
test.head()

print (test.head())

train.head()

print (train.head())



y_pred = model.predict(test[col])

submission = pd.read_csv('../input/sample_submission.csv')  # check where you place the submission.csv

submission['visitors'] = np.asarray(y_pred)

submission.to_csv("lightgbm_baseline.csv",index=False)