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
# GBM prediction



# inspirations:

# https://www.kaggle.com/the1owl/surprise-me/

import math

import numpy as np

import pandas as pd

from sklearn import *

import datetime as dt

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

import xgboost as xgb



def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred) ** 0.5



start_time = dt.datetime.now()

print("Started at ", start_time)
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

#datetime transform+date feature

for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])

    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])

    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

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

stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],

                   axis=0, ignore_index=True).reset_index(drop=True)
def find_outliers(series):

    return (series - series.mean())>2.4*series.std()



data['tra']['is_outlier']=data['tra'].groupby('air_store_id').apply(lambda g: find_outliers(g['visitors'])).values
# mean max min sure it can be compressed... 

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(

    columns={'visitors': 'min_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(

    columns={'visitors': 'mean_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(

    columns={'visitors': 'median_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(

    columns={'visitors': 'max_visitors'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(

    columns={'visitors': 'count_observations'})

stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])



stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])



print("Store df info:")

print(stores.info())
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



train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
# day_of_week label encode

lbl = preprocessing.LabelEncoder()

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])



data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])



train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])

test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])



for df in ['ar', 'hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date'])

    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])



print(train.describe())

print(train.head())


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



# NEW FEATURES FROM Georgii Vyshnia

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 

test['lon_plus_lat'] = test['longitude'] + test['latitude']



lbl = preprocessing.LabelEncoder()

train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])

test['air_store_id2'] = lbl.transform(test['air_store_id'])

train['x'] = np.cos(train['latitude']*math.pi/180.0) * np.cos(train['longitude']*math.pi/180.0)

train['y'] = np.cos(train['latitude']*math.pi/180.0) * np.sin(train['longitude']*math.pi/180.0)

test['x'] = np.cos(test['latitude']*math.pi/180.0) * np.cos(test['longitude']*math.pi/180.0)

test['y'] = np.cos(test['latitude']*math.pi/180.0) * np.sin(test['longitude']*math.pi/180.0)

train['monthday_int']=train['date_int']%10000

train['day_int']=train['date_int']%100

test['monthday_int']=test['date_int']%10000

test['day_int']=test['date_int']%100

# def calc_shifted_ewm(series, alpha, adjust=True):

#     return series.shift().ewm(alpha=alpha, adjust=adjust).mean()



# train['ewm'] = train.set_index('visit_date').groupby(['air_store_id', 'dow'])\

# .apply(lambda g: calc_shifted_ewm(g['visitors'], 0.1))\

# .sort_index(level=['air_store_id', 'visit_date']).values



# test['ewm'] = test.set_index('visit_date').groupby(['air_store_id', 'dow'])\

# .apply(lambda g: calc_shifted_ewm(g['visitors'], 0.1))\

# .sort_index(level=['air_store_id', 'visit_date']).values
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date', 'visitors']]
X = train[col]

y = pd.DataFrame()

y['visitors'] = np.log1p(train['visitors'].values)



# print(X.info())



y_test_pred = 0



# do a hideout split for information leak-free last-minute check

X, X_hideout, y, y_hideout = model_selection.train_test_split(X, y, test_size=0.13, random_state=42)



print("Finished data pre-processing at ", dt.datetime.now())
X2=X[(X.month==4)&(X.year==2017)]

y2=y[(X.month==4)&(X.year==2017)]

X1=X[np.logical_not((X.month==4)&(X.year==2017))]

y1=y[np.logical_not((X.month==4)&(X.year==2017))]
print("Start tuning at ", dt.datetime.now())

param_grid = {

   # 'max_depth':[5,7]

    'reg_lambda':[0,1,2]

               # 'min_child_weight':[1,3]

    

              # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)

              }

est = xgb.XGBRegressor(

    learning_rate=0.01,

    objective='reg:linear',

    n_estimators=1000,

    max_depth=6,

    #min_child_weight=1,

    subsample=0.8,

    colsample_bytree=1,

    scale_pos_weight=1,

    gamma=0.1,

    min_child_weight=3,

    seed=1000)

# this may take some minutes

gs_cv = GridSearchCV(est, param_grid, cv=3, scoring = 'neg_mean_squared_error').fit(X_small, y_small)



print("End up tuning at ", dt.datetime.now())

# best hyperparameter setting

gs_cv.grid_scores_
# Set up folds

K = 3

kf = model_selection.KFold(n_splits = K, random_state = 1, shuffle = True)

np.random.seed(1)





# model

# xgboost

boost_params = {'eval_metric': 'rmse'}

model = xgb.XGBRegressor(

    learning_rate=0.01,

    objective='reg:linear',

    n_estimators=10000,

    max_depth=5,

    min_child_weight=2,

    subsample=0.8,

    colsample_bytree=1,

    scale_pos_weight=1,

#     gamma=4,

#     min_child_weight=3,

    seed=1000)

#seed=27
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn import metrics

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import math



def RMSLE(y, pred):

    return metrics.mean_squared_error(y, pred)**0.5

n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train[col].values)

    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model.fit(X1,y1)
pred=model.predict(X1)
importances=model.feature_importances_

fig, ax = plt.subplots(figsize=(20, 10))

plt.bar(range(train[col].head(1000).shape[1]),importances.tolist())

plt.xticks(range(train[col].head(1000).shape[1]),list(train[col].head(10)),fontsize=16, rotation=90)

plt.show()
RMSLE(y, pred)
RMSLE(y1, pred)
pred_hideout=model.predict(X_hideout)

RMSLE(y_hideout,pred_hideout)
pred2=model.predict(X2)

RMSLE(y2,pred2)


print("Finished setting up CV folds and regressor at ", dt.datetime.now())

# Run CV



print("Started CV at ", dt.datetime.now())

for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Create data for this fold

    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]

    X_train, X_valid = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()

    X_test = test[col]

    print("\nFold ", i)



    fit_model = model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    print('RMSLE GBM Regressor, validation set, fold ', i, ': ', RMSLE(y_valid, pred))



    pred_hideout = model.predict(X_hideout)

    print('RMSLE GBM Regressor, hideout set, fold ', i, ': ', RMSLE(y_hideout, pred_hideout))

    print('Prediction length on validation set, GBM Regressor, fold ', i, ': ', len(pred))

    # Accumulate test set predictions



    pred = model.predict(X_test)

    print('Prediction length on test set, GBM Regressor, fold ', i, ': ', len(pred))

    y_test_pred += pred



    del X_test, X_train, X_valid, y_train



print("Finished CV at ", dt.datetime.now())

y_test_pred /= K  # Average test set predictions

print("Finished average test set predictions at ", dt.datetime.now())
test['is_outlier']=np.nan
y_test_pred=model.predict(test[col])
test.head()
# Create submission file

sub = pd.DataFrame()

sub['id'] = test['id']

sub['visitors'] = np.expm1(y_test_pred) # .clip(lower=0.)

sub.to_csv('C:/Users/Administrator/Desktop/input/xgboost_submit2.csv', float_format='%.6f', index=False)



print('We are done. That is all, folks!')

finish_time = dt.datetime.now()

print("Started at ", finish_time)

elapsed = finish_time - start_time

print("Elapsed time: ", elapsed)