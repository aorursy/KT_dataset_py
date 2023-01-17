# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import json
import time
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

import datetime
import gc
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
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
df_train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
df_test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])

df_train["month"] = df_train["first_active_month"].dt.month
df_test["month"] = df_test["first_active_month"].dt.month

df_train["year"] = df_train["first_active_month"].dt.year
df_test["year"] = df_test["first_active_month"].dt.year

df_train['elapsed_time'] = (datetime.date(2018, 2, 1) - df_train['first_active_month'].dt.date).dt.days
df_test['elapsed_time'] = (datetime.date(2018, 2, 1) - df_test['first_active_month'].dt.date).dt.days


df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])
df_test = pd.get_dummies(df_test, columns=['feature_1', 'feature_2'])

df_train.head()
df_merch = pd.read_csv('../input/merchants.csv')
df_merch.head()
df_hist = pd.read_csv('../input/historical_transactions.csv')
df_hist.head()
df_new = pd.read_csv('../input/new_merchant_transactions.csv')
df_new.head()
df_merch['category_1'] = df_merch['category_1'].map({'Y': 1, 'N': 0})
df_merch['category_4'] = df_merch['category_4'].map({'Y': 1, 'N': 0})
df_merch = pd.get_dummies(df_merch,columns=['category_2','most_recent_sales_range','most_recent_purchases_range'])
df_merch.head()
df_hist = reduce_mem_usage(df_hist)
df_new = reduce_mem_usage(df_new)
df_merch = reduce_mem_usage(df_merch)
def aggregate_transactions_merchants(df, prefix):
    
    agg_func = {
        'numerical_1': ['sum', 'mean'],
        'numerical_2': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_4': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'most_recent_sales_range_A': ['mean'],
        'most_recent_sales_range_B': ['mean'],
        'most_recent_sales_range_C': ['mean'],
        'most_recent_sales_range_D': ['mean'],
        'most_recent_sales_range_E': ['mean'],
        'most_recent_purchases_range_A': ['mean'],
        'most_recent_purchases_range_B': ['mean'],
        'most_recent_purchases_range_C': ['mean'],
        'most_recent_purchases_range_D': ['mean'],
        'most_recent_purchases_range_E': ['mean'],
        'merchant_group_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'avg_sales_lag3': ['sum','mean'],
        'avg_purchases_lag3': ['sum','mean'],
        'active_months_lag3': ['sum', 'mean'],
        'avg_sales_lag6': ['sum','mean'],
        'avg_purchases_lag6': ['sum','mean'],
        'active_months_lag6': ['sum', 'mean'],
        'avg_sales_lag12': ['sum','mean'],
        'avg_purchases_lag12': ['sum','mean'],
        'active_months_lag12': ['sum', 'mean'],
    }
    agg_df = df.groupby(['merchant_id']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    
    df = (df.groupby('merchant_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_df = pd.merge(df, agg_df, on='merchant_id', how='left')
    
    return agg_df
df_merch = aggregate_transactions_merchants(df_merch, prefix='merch_')
df_merch = reduce_mem_usage(df_merch)
gc.collect()
# df_hist = reduce_mem_usage(pd.merge(df_hist, df_merch, on='merchant_id',how='left'))
# Leaving this out for now since memory can't handle it
df_new = pd.merge(df_new, df_merch, on='merchant_id',how='left')
df_new = reduce_mem_usage(df_new)
def aggregate_transactions_hist_new(df, prefix):  
    df.loc[:, 'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max'],
        
    }
    agg_df = df.groupby(['card_id']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    
    df = (df.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_df = pd.merge(df, agg_df, on='card_id', how='left')
    
    return agg_df
df_hist = pd.get_dummies(df_hist, columns=['category_2', 'category_3'])
df_hist['authorized_flag'] = df_hist['authorized_flag'].map({'Y': 1, 'N': 0})
df_hist['category_1'] = df_hist['category_1'].map({'Y': 1, 'N': 0})

df_hist = aggregate_transactions_hist_new(df_hist, prefix='hist_')


df_hist = reduce_mem_usage(df_hist)


gc.collect()
df_train = pd.merge(df_train, df_hist, on='card_id',how='left')
df_test = pd.merge(df_test, df_hist, on='card_id',how='left')

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

print(df_train.shape, df_test.shape)
df_new = pd.get_dummies(df_new, columns=['category_2', 'category_3'])
df_new['authorized_flag'] = df_new['authorized_flag'].map({'Y': 1, 'N': 0})
df_new['category_1'] = df_new['category_1'].map({'Y': 1, 'N': 0})

df_new = aggregate_transactions_hist_new(df_new, prefix='new_')

df_new = reduce_mem_usage(df_new)


gc.collect()
df_train = pd.merge(df_train, df_new, on='card_id',how='left')
df_test = pd.merge(df_test, df_new, on='card_id',how='left')

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

print(df_train.shape, df_test.shape)
target = df_train['target']
cols_to_drop = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in df_train.columns if c not in cols_to_drop]
features = list(df_train[use_cols].columns)
lgb_params = {'num_leaves': 100,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': 10,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_lgb = np.zeros(len(df_train))
predictions_lgb = np.zeros(len(df_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=250, early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits
validation_score = np.sqrt(mean_squared_error(target, oof_lgb))
validation_score
df_submission = pd.DataFrame({"card_id": df_test["card_id"].values})
df_submission["target"] = predictions_lgb
df_submission.to_csv("submission.csv", index=False)