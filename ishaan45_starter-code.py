# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Just to ignore the unneccesary warnings

import sys        

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

path = '../input/ashrae-energy-prediction'



# Any results you write to the current directory are saved as output.
%%time

# unimportant features (see importance below)

unimportant_cols = ['sea_level_pressure']

target = 'meter_reading'



def load_data(source='train', path=path):

    ''' load and merge all tables '''

    assert source in ['train', 'test']

    

    building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})

    weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],

                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,

                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,

                                                                  'precip_depth_1_hr':np.float16},

                                                           usecols=lambda c: c not in unimportant_cols)

    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])

    df = df.merge(building, on='building_id', how='left')

    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')

    return df



# load and display some samples

train = load_data('train')
%%time

test = load_data('test')
# # Code from https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction

# # Function to reduce the DF size



# def reduce_mem_usage(df, verbose=True):

#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#     start_mem = df.memory_usage().sum() / 1024**2    

#     for col in df.columns:

#         col_type = df[col].dtypes

#         if col_type in numerics:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)    

#     end_mem = df.memory_usage().sum() / 1024**2

#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

#     return df
# def average_imputation(df, column_name):

#     imputation = df.groupby(['timestamp'])[column_name].mean()

    

#     df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)

#     del imputation

#     return df
# average_imputation(train,'wind_speed')

# average_imputation(train,'wind_direction')

# average_imputation(train,'dew_temperature')

# average_imputation(train,'air_temperature')

# # average_imputation(train,'sea_level_pressure')
# average_imputation(test,'wind_speed')

# average_imputation(test,'wind_direction')

# average_imputation(test,'dew_temperature')

# average_imputation(test,'air_temperature')

# average_imputation(test,'sea_level_pressure')
def drop_cols(data):

    cols_to_drop = []

    for col in data.columns:

        if data[col].isna().any():

            cols_to_drop.append(col)

    return cols_to_drop



cols_to_drop = drop_cols(train)
%%time

train = train.drop(cols_to_drop, axis = 1)

test = test.drop(cols_to_drop, axis = 1)
def timestamp_decomposition(df):

    df['hour'] = np.uint8(df['timestamp'].dt.hour)

    df['day'] = np.uint8(df['timestamp'].dt.day)

    df['weekday'] = np.uint8(df['timestamp'].dt.weekday)

    df['month'] = np.uint8(df['timestamp'].dt.month)

    df['year'] = np.uint8(df['timestamp'].dt.year-1900)

    return df
train = timestamp_decomposition(train)

test = timestamp_decomposition(test)
# dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')

# us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())



# train['is_holiday'] = (train['timestamp'].dt.date.astype('datetime64').isin(us_holidays))

# test['is_holiday'] = (test['timestamp'].dt.date.astype('datetime64').isin(us_holidays))
def encode_cyclic_feature(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    del df[col]

    return df
train = encode_cyclic_feature(train, 'weekday', 7)

train = encode_cyclic_feature(train, 'hour', 24)

train = encode_cyclic_feature(train, 'day', 31)

train = encode_cyclic_feature(train, 'month', 12)
test = encode_cyclic_feature(test, 'weekday', 7)

test = encode_cyclic_feature(test, 'hour', 24)

test = encode_cyclic_feature(test, 'day', 31)

test = encode_cyclic_feature(test, 'month', 12)
le = LabelEncoder()

train['primary_use'] = le.fit_transform(train['primary_use'])

test['primary_use'] = le.fit_transform(test['primary_use'])
train['square_feet_log'] = np.log(train['square_feet'])

test['square_feet_log'] = np.log(test['square_feet'])
target = np.log1p(train["meter_reading"])

del train['timestamp']

del train['meter_reading']
del test['row_id']

del test['timestamp']
categorical_feats = ['building_id','meter','primary_use']
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm



params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.25,

            'subsample_freq': 1,

            'learning_rate': 0.4,

            'num_leaves': 20,

            'feature_fraction': 0.9,

            'lambda_l1': 1,  

            'lambda_l2': 1

            }



folds = 4

seed = 666



kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)



models = []

for train_index, val_index in kf.split(train, train['building_id']):

    train_X = train.iloc[train_index]

    val_X = train.iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical_feats)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical_feats)

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    models.append(gbm)
i=0

res=[]

step_size = 50000

for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):

    res.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))

    i+=step_size
res = np.concatenate(res)
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

submission['meter_reading'] = res

submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

submission.to_csv('submission.csv', index=False)