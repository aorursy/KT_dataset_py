import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import dask_xgboost as xgb

import dask.dataframe as dd

from sklearn import preprocessing, metrics

import gc

import os

import pickle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

sales_train_validation_dtype = {'id': np.dtype('O'),

 'item_id': np.dtype('O'),

 'dept_id': np.dtype('O'),

 'cat_id': np.dtype('O'),

 'store_id': np.dtype('O'),

 'state_id': np.dtype('O')}



for i in range(1, 1914):

    sales_train_validation_dtype['d_'+str(i)] = np.int16

    

    

sell_prices_dtype = {'store_id': np.dtype('O'),

                     'item_id': np.dtype('O'),

                     'wm_yr_wk': np.dtype('int16'),

                     'sell_price': np.dtype('float16')}



calendar_dtype = {'date': np.dtype('O'),

                 'wm_yr_wk': np.dtype('int16'),

                 'weekday': np.dtype('O'),

                 'wday': np.dtype('int8'),

                 'month': np.dtype('int8'),

                 'year': np.dtype('int16'),

                 'd': np.dtype('O'),

                 'event_name_1': np.dtype('O'),

                 'event_type_1': np.dtype('O'),

                 'event_name_2': np.dtype('O'),

                 'event_type_2': np.dtype('O'),

                 'snap_CA': np.dtype('int8'),

                 'snap_TX': np.dtype('int8'),

                 'snap_WI': np.dtype('int8')}    



sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv', 

                                    dtype=sales_train_validation_dtype)

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv',

                      dtype=calendar_dtype)

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv',

                         dtype=sell_prices_dtype)

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')



print(sales_train_validation.info(verbose=False), 

      calendar.info(verbose=False), sell_prices.info(verbose=False), submission.info(verbose=False))
# Standarsize the data

sales_train_validation['id'] = sales_train_validation['id'].apply(lambda x:'_'.join(x.split('_')[:-1]))

submission['id'] = submission['id'].apply(lambda x:'_'.join(x.split('_')[:-1]))



# Convert columns 'd_n' to n as well as the column 'd' in calendar

day_map = {'d_'+str(i):i for i in range(1, 1970)}

sales_train_validation = sales_train_validation.rename(columns=day_map)

calendar['d'] = calendar['d'].apply(lambda x:day_map[x]).astype(np.int16)

train_cat_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

calendar_cat_cols = ['weekday', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']



train_lbl_encoders = {}

for col in train_cat_cols:

    lbl = preprocessing.LabelEncoder()

    if col in ['item_id', 'id']:

        sales_train_validation[col] = lbl.fit_transform(sales_train_validation[col].fillna('UNK')).astype(np.int16)

    else:

        sales_train_validation[col] = lbl.fit_transform(sales_train_validation[col].fillna('UNK')).astype(np.int8)

    train_lbl_encoders[col] = lbl



calendar_lbl_encoders = {}    

for col in calendar_cat_cols:

    lbl = preprocessing.LabelEncoder()

    calendar[col] = lbl.fit_transform(calendar[col].fillna('UNK')).astype(np.int8)

    train_lbl_encoders[col] = lbl



for col in ['store_id', 'item_id']:

    sell_prices[col] = train_lbl_encoders[col].transform(sell_prices[col].fillna('UNK'))

    

pickle.dump(train_lbl_encoders, open('train_lbl_encoders.pkl', 'wb'))

pickle.dump(calendar_lbl_encoders, open('calendar_lbl_encoders.pkl', 'wb'))

pickle.dump(day_map, open('day_map.pkl', 'wb'))



# Unique products

product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()



# Added validation/ evaluation days

for i in range(1914, 1970):

    sales_train_validation[i] = 0
# Date-time features

calendar['date'] = pd.to_datetime(calendar['date'])

calendar['week'] = calendar['date'].dt.week.astype(np.int8)

calendar['day'] = calendar['date'].dt.day.astype(np.int8)

calendar.drop(['weekday'], axis=1, inplace=True)

calendar.head()


%%time

sales_train_validation = pd.melt(sales_train_validation,

                                    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                                    var_name='d', value_name='sales')

sales_train_validation['d'] = sales_train_validation['d'].astype(np.int16)

sales_train_validation['sales'] = sales_train_validation['sales'].astype(np.int16)



sales_train_validation = pd.merge(sales_train_validation, 

                                  calendar, 

                                  how='left', 

                                  on='d')



sales_train_validation = pd.merge(sales_train_validation,

                                  sell_prices, 

                                  on=['store_id', 'item_id', 'wm_yr_wk'], 

                                  how = 'left')

print(sales_train_validation.info())
del sell_prices, calendar