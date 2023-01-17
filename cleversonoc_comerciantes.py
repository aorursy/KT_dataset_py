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
import numpy as np

import pandas as pd

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')

np.random.seed(2)
#Reduce the memory usage

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

#Loading data
#Loading data

df_new_merchant_trans = reduce_mem_usage(pd.read_csv('../input/novas-transacoes/novas_transacoes_comerciantes.csv',parse_dates=['purchase_date']))

df_hist_trans = reduce_mem_usage(pd.read_csv('../input/historicas/transacoes_historicas.csv',parse_dates=['purchase_date']))

df_train = reduce_mem_usage(pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv',parse_dates=["first_active_month"]))

df_test = reduce_mem_usage(pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv',parse_dates=["first_active_month"]))
df_hist_trans.head()
df_new_merchant_trans.head()
df_hist_trans.isna().sum()
#movendo valores para campos com NA

for df in [df_hist_trans,df_new_merchant_trans]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('B',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
for df in [df_hist_trans,df_new_merchant_trans]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0}).astype(int)

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}).astype(int) 

    df['category_3'] = df['category_3'].map({'A':0, 'B':1 , 'C':2}).astype(int)

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
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



print(aggs)



for col in ['category_2','category_3']:

    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']    

    

print(aggs)



new_columns = get_new_columns('hist',aggs)

df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days

df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']

df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
aggs = {}

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']

aggs['installments'] = ['sum','max','min','mean','var']

aggs['purchase_date'] = ['max','min']

aggs['month_lag'] = ['max','min','mean','var']

aggs['month_diff'] = ['mean']

aggs['weekend'] = ['sum', 'mean']

aggs['category_1'] = ['sum', 'mean']

aggs['card_id'] = ['size']



for col in ['category_2','category_3']:

    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']

    

new_columns = get_new_columns('new_hist',aggs)

df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days

df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']

df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
del df_hist_trans;gc.collect()

del df_new_merchant_trans;gc.collect()

df_train.head(5)
df_train['outliers'] = 0

df_train.loc[df_train['target'] < -30, 'outliers'] = 1

df_train['outliers'].value_counts()
for df in [df_train,df_test]:

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

    order_label = df_train.groupby([f])['outliers'].mean()

    df_train[f] = df_train[f].map(order_label)

    df_test[f] = df_test[f].map(order_label)
target = df_train['target']

del df_train['target']

train = df_train

test = df_test
#Restringindo as colunas por importancia

#df_train_columns = ['hist_month_diff_mean', 'hist_authorized_flag_mean', 'new_hist_purchase_amount_max', 'new_hist_purchase_date_uptonow','hist_category_1_sum', 'hist_month_lag_mean',

#                    'hist_purchase_date_min', 'hist_purchase_date_max']



df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

features = df_train_columns
param = {'num_leaves': 111,

         'min_data_in_leaf': 149, 

         'objective':'regression',

         'max_depth': 9,

         'learning_rate': 0.005,

         "boosting": "gbdt",

         "feature_fraction": 0.7522,

         "bagging_freq": 1,

         "bagging_fraction": 0.7083 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2634,

         "random_state": 133,

         "verbosity": -1}

folds = StratifiedKFold(n_splits=5, n_repeats=1, random_state=133)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 150)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = df_train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))
#Applying RepeatedKFolds

from sklearn.model_selection import RepeatedKFold



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



folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)

oof_2 = np.zeros(len(train))

predictions_2 = np.zeros(len(test))

feature_importance_df_2 = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 11000

    clf_r = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)

    oof_2[val_idx] = clf_r.predict(train.iloc[val_idx][features], num_iteration=clf_r.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf_r.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df_2 = pd.concat([feature_importance_df_2, fold_importance_df], axis=0)

    

    predictions_2 += clf_r.predict(test[features], num_iteration=clf_r.best_iteration) / (5 * 2)



print("CV score: {:<8.5f}".format(mean_squared_error(oof_2, target)**0.5))
#Applying BayesianRidge

from sklearn.linear_model import BayesianRidge



train_stack = np.vstack([oof,oof_2]).transpose()

test_stack = np.vstack([predictions, predictions_2]).transpose()



folds_stack = RepeatedKFold(n_splits=5, shuffle=True, random_state=15)

oof_stack = np.zeros(train_stack.shape[0])

predictions_3 = np.zeros(test_stack.shape[0])



for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):

    print("fold {}".format(fold_))

    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values

    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    

    clf_3 = BayesianRidge()

    clf_3.fit(trn_data, trn_y)

    

    oof_stack[val_idx] = clf_3.predict(val_data)

    predictions_3 += clf_3.predict(test_stack) / 5

    

np.sqrt(mean_squared_error(target.values, oof_stack))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_3

sub_df.to_csv("submission.csv", index=False)