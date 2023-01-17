# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.linear_model import Ridge

import time

from sklearn import preprocessing

import warnings

import datetime

warnings.filterwarnings("ignore")

import gc

from tqdm import tqdm



from scipy.stats import describe

%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
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
import os

print(os.listdir("../input"))
#Loading Train and Test Data

train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])

test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])

hist_trans = pd.read_csv('../input/historical_transactions.csv')

new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')

print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))

print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))
train.head()
for df in [hist_trans,new_merchant_trans]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
for df in [hist_trans,new_merchant_trans]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 

    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
def aggregate(df,name):

    aggs = {}

    for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

        aggs[col] = ['nunique']



    aggs['purchase_amount'] = ['sum','max','min','mean','var']

    aggs['installments'] = ['sum','max','min','mean','var']

    aggs['purchase_date'] = ['max','min']

    aggs['month_lag'] = ['max','min','mean','var']

    aggs['month_diff'] = ['mean']

    aggs['authorized_flag'] = ['sum', 'mean']

    aggs['weekend'] = ['sum', 'mean']

    aggs['category_1'] = ['sum', 'mean']

    aggs['card_id'] = ['size']



    for col in ['category_2','category_3']:

        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')

        aggs[col+'_mean'] = ['mean'] 

            

    new_columns = get_new_columns(name,aggs)

    new_df = df.groupby('card_id').agg(aggs)

    new_df.columns = new_columns

    new_df.reset_index(drop=False,inplace=True)

    new_df[name + '_purchase_date_diff'] = (new_df[name + '_purchase_date_max'] - new_df[name + '_purchase_date_min']).dt.days

    new_df[name + '_purchase_date_average'] = new_df[name + '_purchase_date_diff']/new_df[name + '_card_id_size']

    new_df[name + '_purchase_date_uptonow'] = (datetime.datetime.today() - new_df[name + '_purchase_date_max']).dt.days

    

    return new_df
hist_agged_df = aggregate(hist_trans,'hist');

new_hist_agged_df = aggregate(new_merchant_trans,'new_hist');



for df in [hist_agged_df,new_hist_agged_df]:

    train = train.merge(df,on='card_id',how='left')

    test = test.merge(df,on='card_id',how='left')
display(train.head())

display(test.head())

display(train.describe())
train['outliers'] = 0

train.loc[train['target'] < -30, 'outliers'] = 1

train['outliers'].value_counts()
for df in [train,test]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\

                     'new_hist_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']



for f in ['feature_1','feature_2','feature_3']:

    order_label = train.groupby([f])['outliers'].mean()

    train[f] = train[f].map(order_label)

    test[f] = test[f].map(order_label)
train_columns = [c for c in train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = train['target']

del train['target']
param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4590}



oof_lgb_3 = np.zeros(len(train))

predictions_lgb_3 = np.zeros(len(test))

start = time.time()



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):    

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][train_columns], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)

    oof_lgb_3[val_idx] = clf.predict(train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    

    predictions_lgb_3 += clf.predict(test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.save('oof_lgb_3', oof_lgb_3)

np.save('predictions_lgb_3', predictions_lgb_3)

print("CV score: {:<8.5f}".format(mean_squared_error(oof_lgb_3, target)**0.5))
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = predictions_lgb_3

sample_submission.to_csv('stacker.csv', index=False)