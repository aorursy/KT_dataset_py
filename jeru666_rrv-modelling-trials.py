import glob, re

import numpy as np

import pandas as pd

from sklearn import *

from datetime import datetime

import seaborn as sns

from matplotlib import pyplot as plt



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
data['hr'].head()
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
data['hr'].isnull().values.any()
data['hr'].head()
data['ar'].head()
for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
data['ar'].head()
for df in ['ar','hr']:

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_visitors':'rv1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_visitors':'rv2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
tmp1.head()
tmp2.head()
data['ar'].head()
data['tra'].head()
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
data['tra'].head()
data['tes'].head()
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
data['tes'].head()
unique_stores = data['tes']['air_store_id'].unique()
unique_stores
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
stores.head()
print('Range of dow is from {} to {}'.format(min(stores.dow), max(stores.dow)))
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
tmp.head()
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
stores.head()
stores.shape
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
stores.shape
stores.head()
lbl = preprocessing.LabelEncoder()

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
stores.head()
data['hol'].head()
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
data['hol'].head()
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 
train.head()
test.head()
train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
train.head()
train.shape
test.head()
test.shape
for df in ['ar','hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
train.head()
train.shape
test.head()
test.shape
train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)
train.head()
train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']

train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']

test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]

print(train.describe())

print(train.head())
train.columns
train = train.fillna(-1)

test = test.fillna(-1)
def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5
train.head()
len(train.columns)
from sklearn.model_selection import train_test_split



#features= [c for c in train.columns.values if c  not in ['air_store_id', 'visitors']]

#numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]

#target = 'author'



X_train, X_val, y_train, y_val = train_test_split(train[col], train['visitors'], test_size=0.33, random_state=42)

X_train.head()
print(len(X_train))

print(len(X_val))
len(X_train.columns)
''' 

from sklearn import ensemble.RandomForestRegressor

from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    #('features',feats),

    #'gbr', ensemble.GradientBoostingRegressor(random_state = 2017),

    ('rfr', ensemble.RandomForestRegressor(random_state = 2017))

    #('gbr', ensemble.GradientBoostingRegressor(random_state = 2017))

    #('knn', neighbors.KNeighborsRegressor(n_jobs = -1))

    #('rfr', ensemble.RandomForestRegressor(random_state = 2017))

])

''' 
'''

pipeline.fit(X_train, np.log1p(y_train.values))



preds = pipeline.predict(X_val)

preds1 = np.expm1(preds)#.clip(lower=0.)

np.mean(preds1 == y_val)

'''
'''

from sklearn.model_selection import GridSearchCV



hyperparameters = { 

                    #'classifier__learning_rate': [0.1, 0.2],

                    #'gbr__n_estimators': [350],

                    #'gbr__max_depth': [6],

                    #'gbr__min_samples_leaf': [2]

                    'knn__n_neighbors': [7]

                  }

clf = GridSearchCV(pipeline, hyperparameters, cv = 3)

 

# Fit and tune model

clf.fit(X_train, np.log1p(y_train.values))

'''
#clf.best_params_
'''

#refitting on entire training data using best settings

clf.refit



preds = clf.predict(X_val)



preds1 = np.expm1(preds)



#np.mean(preds1 == y_val)

#print('RMSE GradientBoostingRegressor Validation: ', RMSLE(y_val, preds1))

print('RMSE KNNRegressor Validation: ', RMSLE(y_val, preds1))

'''
model1 = ensemble.GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 375, max_depth = 6, min_samples_leaf = 2)

#model2 = neighbors.KNeighborsRegressor(n_jobs = -1, n_neighbors = 8)
model1.fit(train[col], np.log1p(train['visitors'].values))

#model2.fit(train[col], np.log1p(train['visitors'].values))




print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), model1.predict(train[col])))

#print('RMSE KNeighborsRegressor: ', RMSLE(np.log1p(train['visitors'].values), model2.predict(train[col])))

#test['visitors'] = (model1.predict(test[col]) + model2.predict(test[col])) / 2

test['visitors'] = model1.predict(test[col])

test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)

sub1 = test[['id','visitors']].copy()

#del train; del data;
sub1.head()
sub1[['id', 'visitors']].to_csv('submission.csv', index=False)