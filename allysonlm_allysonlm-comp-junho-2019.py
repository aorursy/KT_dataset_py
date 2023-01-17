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

np.random.seed(2019)

pd.set_option('display.max_columns', 500)
df_hist_trans = pd.read_csv('../input/dsa-comp4/transacoes_historicas.csv')
df_new_merchant_trans = pd.read_csv('../input/dsa-comp4/novas_transacoes_comerciantes.csv')
df_train = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv')

df_test = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv')
# valores na

for df in [df_hist_trans,df_new_merchant_trans]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
# funcoes extras

def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]



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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
# features

now = datetime.datetime(2019, 1, 1, 23, 0, 0)

for df in [df_hist_trans,df_new_merchant_trans]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 

    df['month_diff'] = (( now - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
# hist

now = datetime.datetime(2019, 1, 1, 23, 0, 0)

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

    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']    



new_columns = get_new_columns('hist',aggs)

df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days

df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']

df_hist_trans_group['hist_purchase_date_uptonow'] = (now - df_hist_trans_group['hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
# novas

now = datetime.datetime(2019, 1, 1, 23, 0, 0)

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

df_hist_trans_group['new_hist_purchase_date_uptonow'] = (now - df_hist_trans_group['new_hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
del df_hist_trans;gc.collect()

del df_new_merchant_trans;gc.collect()

df_train.head(5)
df_train['outliers'] = 0

df_train.loc[df_train['target'] < -30, 'outliers'] = 1

df_train['outliers'].value_counts()
# novas features apos merge

now = datetime.datetime(2019, 1, 1, 23, 0, 0)

for df in [df_train,df_test]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (now - df['first_active_month']).dt.days

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

df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = df_train['target']

del df_train['target']
param = {'num_leaves': 28,

         'min_data_in_leaf': 28, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.005,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.8,

         "bagging_freq": 1,

         "bagging_fraction": 0.8 ,

         "bagging_seed": 10,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2019}



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = df_train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("comp4_v3_3_teste_xxxx.csv", index=False)