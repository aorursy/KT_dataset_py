import gc

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error

from tqdm import tqdm
# reduce memory

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
Debug = False

if Debug:

    train_df = reduce_mem_usage(pd.read_csv('../input/train.csv',nrows=100000))

    test_df = reduce_mem_usage(pd.read_csv('../input/test.csv'))

    historical_trans_df = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv',nrows=1000000))

    new_merchant_trans_df = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv',nrows=1000000))

else:

    train_df = reduce_mem_usage(pd.read_csv('../input/train.csv'))

    test_df = reduce_mem_usage(pd.read_csv('../input/test.csv'))

    historical_trans_df = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv'))

    new_merchant_trans_df = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv'))
train_df.head()
train_df.columns.tolist()
#histogram

f, ax = plt.subplots(figsize=(14, 6))

sns.distplot(train_df['target'])
train_df.groupby(['feature_1','feature_2','feature_3'])['target'].agg({'min','mean','max','std','skew'})
historical_trans_df.head()
# new_merchant also same

historical_trans_df.isnull().sum()
#process NA2 for transactions

for df in [historical_trans_df, new_merchant_trans_df]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    

#define function for aggregation

def create_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
# The trick deal with outliers - Aleksandr Kosolapov (Rank 1st)

train_df['rounded_target'] = train_df['target'].round(0)

train_df = train_df.sort_values('rounded_target').reset_index(drop=True)

vc = train_df['rounded_target'].value_counts()

vc = dict(sorted(vc.items()))

df = pd.DataFrame()

train_df['indexcol'],i = 0,1

for k,v in vc.items():

    step = train_df.shape[0]/v

    indent = train_df.shape[0]/(v+1)

    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)

    for j in range(0, v):

        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i

    df = pd.concat([df2,df])

    i+=1

train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)

del train_df['indexcol'], train_df['rounded_target']
for df in [historical_trans_df, new_merchant_trans_df]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 

    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}) 

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']

    df['price'] = df['purchase_amount'] / df['installments']

aggs = {}



for col in ['subsector_id','merchant_id','merchant_category_id', 'state_id', 'city_id']:

    aggs[col] = ['nunique']

for col in ['month', 'hour', 'weekofyear', 'dayofweek']:

    aggs[col] = ['nunique', 'mean', 'min', 'max']

    

   

aggs['purchase_amount'] = ['sum','max','min','mean','var', 'std']

aggs['installments'] = ['sum','max','min','mean','var', 'std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['month_lag'] = ['max','min','mean','var','nunique']

aggs['month_diff'] = ['mean', 'min', 'max', 'var','nunique']

aggs['authorized_flag'] = ['sum', 'mean', 'nunique']

aggs['weekend'] = ['sum', 'mean', 'nunique']

aggs['year'] = ['nunique', 'mean']

aggs['category_1'] = ['sum', 'mean', 'min', 'max', 'nunique', 'std']

aggs['category_2'] = ['sum', 'mean', 'min', 'nunique', 'std']

aggs['category_3'] = ['sum', 'mean', 'min', 'nunique', 'std']

aggs['card_id'] = ['size', 'count']

aggs['price'] = ['min','mean','std','max']





for col in ['category_2','category_3']:

    historical_trans_df[col+'_mean'] = historical_trans_df.groupby([col])['purchase_amount'].transform('mean')

    historical_trans_df[col+'_min'] = historical_trans_df.groupby([col])['purchase_amount'].transform('min')

    historical_trans_df[col+'_max'] = historical_trans_df.groupby([col])['purchase_amount'].transform('max')

    historical_trans_df[col+'_sum'] = historical_trans_df.groupby([col])['purchase_amount'].transform('sum')

    historical_trans_df[col+'_std'] = historical_trans_df.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']    



new_columns = create_new_columns('hist',aggs)

historical_trans_group_df = historical_trans_df.groupby('card_id').agg(aggs)

historical_trans_group_df.columns = new_columns

historical_trans_group_df.reset_index(drop=False,inplace=True)

historical_trans_group_df['hist_purchase_date_diff'] = (historical_trans_group_df['hist_purchase_date_max'] - historical_trans_group_df['hist_purchase_date_min']).dt.days

historical_trans_group_df['hist_purchase_date_average'] = historical_trans_group_df['hist_purchase_date_diff']/historical_trans_group_df['hist_card_id_size']

historical_trans_group_df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_max']).dt.days

historical_trans_group_df['hist_purchase_date_uptomin'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_min']).dt.days



#merge with train, test

train_df = train_df.merge(historical_trans_group_df,on='card_id',how='left')

test_df = test_df.merge(historical_trans_group_df,on='card_id',how='left')
del historical_trans_group_df; gc.collect()



#define aggregations with new_merchant_trans_df 

aggs = {}

for col in ['subsector_id','merchant_id','merchant_category_id','state_id', 'city_id']:

    aggs[col] = ['nunique']

for col in ['month', 'hour', 'weekofyear', 'dayofweek']:

    aggs[col] = ['nunique', 'mean', 'min', 'max']



    

aggs['purchase_amount'] = ['sum','max','min','mean','var','std']

aggs['installments'] = ['sum','max','min','mean','var','std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['month_lag'] = ['max','min','mean','var', 'nunique']

aggs['month_diff'] = ['mean', 'max', 'min', 'var','nunique']

aggs['weekend'] = ['sum', 'mean', 'nunique']

aggs['year'] = ['nunique', 'mean']

aggs['category_1'] = ['sum', 'mean', 'min', 'nunique']

aggs['category_2'] = ['sum', 'mean', 'min', 'nunique']

aggs['category_3'] = ['sum', 'mean', 'min', 'nunique']

aggs['card_id'] = ['size', 'count']

aggs['price'] = ['min','mean','std','max']





for col in ['category_2','category_3']:

    new_merchant_trans_df[col+'_mean'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('mean')

    new_merchant_trans_df[col+'_min'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('min')

    new_merchant_trans_df[col+'_max'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('max')

    new_merchant_trans_df[col+'_sum'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('sum')

    new_merchant_trans_df[col+'_std'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']



new_columns = create_new_columns('new_hist',aggs)

new_merchant_trans_group_df = new_merchant_trans_df.groupby('card_id').agg(aggs)

new_merchant_trans_group_df.columns = new_columns

new_merchant_trans_group_df.reset_index(drop=False,inplace=True)

new_merchant_trans_group_df['new_hist_purchase_date_diff'] = (new_merchant_trans_group_df['new_hist_purchase_date_max'] - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days

new_merchant_trans_group_df['new_hist_purchase_date_average'] = new_merchant_trans_group_df['new_hist_purchase_date_diff']/new_merchant_trans_group_df['new_hist_card_id_size']

new_merchant_trans_group_df['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_max']).dt.days

new_merchant_trans_group_df['new_hist_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days

#merge with train, test

train_df = train_df.merge(new_merchant_trans_group_df,on='card_id',how='left')

test_df = test_df.merge(new_merchant_trans_group_df,on='card_id',how='left')
#clean-up memory

del new_merchant_trans_group_df; gc.collect()

del historical_trans_df; gc.collect()

del new_merchant_trans_df; gc.collect()
train_df['outliers'] = 0

train_df.loc[train_df['target'] < -30, 'outliers'] = 1

outls = train_df['outliers'].value_counts()

print("Outliers: {}".format(outls))



## process both train and test

for df in [train_df, test_df]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['dayofyear'] = df['first_active_month'].dt.dayofyear

    df['quarter'] = df['first_active_month'].dt.quarter

    df['is_month_start'] = df['first_active_month'].dt.is_month_start

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_last_buy'] = (df['new_hist_purchase_date_max'] - df['first_active_month']).dt.days

    

    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\

                     'new_hist_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

        

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['card_id_cnt_total'] = df['new_hist_card_id_count']+df['hist_card_id_count']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

    df['purchase_amount_mean'] = df['new_hist_purchase_amount_mean']+df['hist_purchase_amount_mean']

    df['purchase_amount_max'] = df['new_hist_purchase_amount_max']+df['hist_purchase_amount_max']

    

    df['days_feature1'] = df['elapsed_time'] * df['feature_1']

    df['days_feature2'] = df['elapsed_time'] * df['feature_2']

    df['days_feature3'] = df['elapsed_time'] * df['feature_3']



    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']

    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']

    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']



for f in ['feature_1','feature_2','feature_3']:

    order_label = train_df.groupby([f])['outliers'].mean()

    train_df[f] = train_df[f].map(order_label)

    test_df[f] = test_df[f].map(order_label)

    

for df in [train_df,test_df]:

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']

    df['feature_mean'] = df['feature_sum']/3

    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)

    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)

    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    

##

train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = train_df['target']

del train_df['target']
##model params

param = {'num_leaves': 51,

         'min_data_in_leaf': 35, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.008,

         "boosting": "gbdt",

         "feature_fraction": 0.85,

         "bagging_freq": 1,

         "bagging_fraction": 0.82,

         "bagging_seed": 42,

         "metric": 'rmse',

         "lambda_l1": 0.11,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2019}



#prepare fit model with cross-validation

folds = KFold(n_splits=9, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()

#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    logger.info(strLog)

    

strRMSE = "".format(np.sqrt(mean_squared_error(oof, target)))

print(strRMSE)
##plot the feature importance

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)