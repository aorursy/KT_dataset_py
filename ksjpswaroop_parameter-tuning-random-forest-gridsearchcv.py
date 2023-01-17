# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import glob, re

import numpy as np

import pandas as pd

from sklearn import *

from datetime import datetime

from xgboost import XGBRegressor



from keras.layers import Embedding, Input, Dense

from keras.models import Model

import keras

import keras.backend as K



import matplotlib.pyplot as plt

from sklearn.externals import joblib
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

    # Exclude same-week reservations - from aharless kernel

    data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']]

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])



data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date



data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

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

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

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

print("Done")
def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5
print("Start of Data Load")

value_col = ['holiday_flg','min_visitors','mean_visitors','median_visitors','max_visitors','count_observations',

'rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rv1_y','rs2_y','rv2_y','total_reserv_sum','total_reserv_mean',

'total_reserv_dt_diff_mean','date_int','var_max_lat','var_max_long','lon_plus_lat']



nn_col = value_col + ['dow', 'year', 'month', 'air_store_id2', 'air_area_name', 'air_genre_name',

'air_area_name0', 'air_area_name1', 'air_area_name2', 'air_area_name3', 'air_area_name4',

'air_area_name5', 'air_area_name6', 'air_genre_name0', 'air_genre_name1',

'air_genre_name2', 'air_genre_name3', 'air_genre_name4']





X = train.copy()

X_test = test[nn_col].copy()



value_scaler = preprocessing.MinMaxScaler()

for vcol in value_col:

    X[vcol] = value_scaler.fit_transform(X[vcol].values.astype(np.float64).reshape(-1, 1))

    X_test[vcol] = value_scaler.transform(X_test[vcol].values.astype(np.float64).reshape(-1, 1))



X_train = list(X[nn_col].T.as_matrix())

Y_train = np.log1p(X['visitors']).values

nn_train = [X_train, Y_train]

nn_test = [list(X_test[nn_col].T.as_matrix())]

print("Train and test data prepared")
#***************************************Random Forest

model2 = ensemble.RandomForestRegressor(n_estimators=13, random_state=3, max_depth=18,

                                        min_weight_fraction_leaf=0.0002)
# *********************** Training and Validation Data *************************#

train_X = train[col]

train_X = train_X[train_X['year'] == 2016]

train_y = train[train['year'] == 2016]

test_X =  train[col]

test_X = test_X[test_X['year'] == 2017]

test_y = train[train['year'] == 2017]

# *********************** Training and Validation Data *************************#

#**********************************************************

model2.fit(train_X, np.log1p(train_y['visitors'].values))

print("Model2 trained")

preds2 = model2.predict(test_X)

print('RMSE RandomForestRegressor: ', RMSLE(np.log1p(test_y['visitors'].values), preds2))
# Commented after running on a local computer

# takes a longer time

'''

from sklearn.model_selection import GridSearchCV



# parameters for GridSearchCV

param_grid2 = {"n_estimators": [10, 18, 22],

              "max_depth": [3, 5],

              "min_samples_split": [15, 20],

              "min_samples_leaf": [5, 10, 20],

              "max_leaf_nodes": [20, 40],

              "min_weight_fraction_leaf": [0.1]}

grid_search = GridSearchCV(model2, param_grid=param_grid2)

grid_search.fit(train_X, np.log1p(train_y['visitors'].values))

'''

from operator import itemgetter



# Utility function to report best scores

def report(grid_scores, n_top):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]

    for i, score in enumerate(top_scores):

        print("Model with rank: {0}".format(i + 1))

        print("Mean validation score: {0:.4f})".format(

              score.mean_validation_score,

              np.std(score.cv_validation_scores)))

        print("Parameters: {0}".format(score.parameters))

        print("")

# Commented after running on a local computer

#report(grid_search.grid_scores_,4)
#Change parameters and test if it performs better than the prior model

#***************************************Random Forest

model2 = ensemble.RandomForestRegressor(n_estimators=18, random_state=3, max_depth=3,

                                        min_weight_fraction_leaf=0.1,max_leaf_nodes = 20,

                                       min_samples_split = 20)

#**********************************************************

model2.fit(train_X, np.log1p(train_y['visitors'].values))

print("Model2 trained")

preds2 = model2.predict(test_X)

print('RMSE RandomForestRegressor: ', RMSLE(np.log1p(test_y['visitors'].values), preds2))