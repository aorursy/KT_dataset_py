# Import the necessary libraries

import numpy as np

import pandas as pd

import nltk

import matplotlib.pyplot as plt

import datetime

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import time

import sys

import lightgbm as lgb
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

new_transactions = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv',parse_dates=['purchase_date']))

old_transactions = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv',parse_dates=['purchase_date']))

train = reduce_mem_usage(pd.read_csv('../input/train.csv',parse_dates=["first_active_month"]))

test = reduce_mem_usage(pd.read_csv('../input/test.csv',parse_dates=["first_active_month"]))
new_transactions.head()
old_transactions.head()
train.head()
test.head()
#Finding columns with null values

old_transactions.isna().sum()
#Replacing null values with the most frequent values in the column.

old_transactions['category_2'].fillna(1.0,inplace = True)

old_transactions['category_3'].fillna('B',inplace = True)



new_transactions['category_2'].fillna(1.0,inplace = True)

new_transactions['category_3'].fillna('B',inplace = True)
#Changing Categorical Columns to Boolean

old_transactions['authorized_flag'] = old_transactions['authorized_flag'].map({'Y':1, 'N':0}).astype(bool)

old_transactions['category_1'] = old_transactions['category_1'].map({'Y':1, 'N':0}).astype(bool)

old_transactions['category_3'] = old_transactions['category_3'].map({'A':0, 'B':1, 'C':2}).astype('category')



new_transactions['authorized_flag'] = new_transactions['authorized_flag'].map({'Y':1, 'N':0}).astype(bool)

new_transactions['category_1'] = new_transactions['category_1'].map({'Y':1, 'N':0}).astype(bool)

new_transactions['category_3'] = new_transactions['category_3'].map({'A':0, 'B':1, 'C':2}).astype('category')

old_transactions.head()
new_transactions.head()
#Handling the Outliers

train['outliers'] = 0

train.loc[train['target'] < -30, 'outliers'] = 1

outls = train['outliers'].value_counts()

print("Outliers:\n{}".format(outls))
#define function to name columns for aggregation

def create_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
#Adding a few features using purchase_amount, purchase_date

aggs={}

aggs['purchase_amount'] = ['sum','max','min','mean','var','std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['card_id'] = ['size', 'count']
for col in ['category_2','category_3']:

    old_transactions[col+'_mean'] = old_transactions.groupby([col])['purchase_amount'].transform('mean')

    old_transactions[col+'_min'] = old_transactions.groupby([col])['purchase_amount'].transform('min')

    old_transactions[col+'_max'] = old_transactions.groupby([col])['purchase_amount'].transform('max')

    old_transactions[col+'_sum'] = old_transactions.groupby([col])['purchase_amount'].transform('sum')

    old_transactions[col+'_std'] = old_transactions.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']    
new_columns = create_new_columns('old',aggs)

historical_trans_group_df = old_transactions.groupby('card_id').agg(aggs)

historical_trans_group_df.columns = new_columns

historical_trans_group_df.reset_index(drop=False,inplace=True)

historical_trans_group_df['old_purchase_date_diff'] = (historical_trans_group_df['old_purchase_date_max'] - historical_trans_group_df['old_purchase_date_min']).dt.days

historical_trans_group_df['old_purchase_date_average'] = historical_trans_group_df['old_purchase_date_diff']/historical_trans_group_df['old_card_id_size']

historical_trans_group_df['old_purchase_date_uptonow'] = (datetime.datetime.today() - historical_trans_group_df['old_purchase_date_max']).dt.days

historical_trans_group_df['old_purchase_date_uptomin'] = (datetime.datetime.today() - historical_trans_group_df['old_purchase_date_min']).dt.days

aggs['purchase_amount'] = ['sum','max','min','mean','var','std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['card_id'] = ['size', 'count']
for col in ['category_2','category_3']:

    new_transactions[col+'_mean'] = new_transactions.groupby([col])['purchase_amount'].transform('mean')

    new_transactions[col+'_min'] = new_transactions.groupby([col])['purchase_amount'].transform('min')

    new_transactions[col+'_max'] = new_transactions.groupby([col])['purchase_amount'].transform('max')

    new_transactions[col+'_sum'] = new_transactions.groupby([col])['purchase_amount'].transform('sum')

    new_transactions[col+'_std'] = new_transactions.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']
new_columns = create_new_columns('new',aggs)

new_merchant_trans_group_df = new_transactions.groupby('card_id').agg(aggs)

new_merchant_trans_group_df.columns = new_columns

new_merchant_trans_group_df.reset_index(drop=False,inplace=True)

new_merchant_trans_group_df['new_purchase_date_diff'] = (new_merchant_trans_group_df['new_purchase_date_max'] - new_merchant_trans_group_df['new_purchase_date_min']).dt.days

new_merchant_trans_group_df['new_purchase_date_average'] = new_merchant_trans_group_df['new_purchase_date_diff']/new_merchant_trans_group_df['new_card_id_size']

new_merchant_trans_group_df['new_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_purchase_date_max']).dt.days

new_merchant_trans_group_df['new_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_purchase_date_min']).dt.days

#Creating Dummy Variables 

old_transactions = pd.get_dummies(old_transactions, columns=['category_2', 'category_3'])

new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
#Adding a few features created using purchase_date.

for df in [old_transactions, new_transactions]:

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['dayofyear'] = df['purchase_date'].dt.dayofyear

    df['quarter'] = df['purchase_date'].dt.quarter

    df['is_month_start'] = df['purchase_date'].dt.is_month_start

    df['purchase_month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
#Adding a few features created using first_active_month.

for df in [train, test]:

    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['dayofyear'] = df['first_active_month'].dt.dayofyear

    df['quarter'] = df['first_active_month'].dt.quarter

    df['is_month_start'] = df['first_active_month'].dt.is_month_start

    df['month'] = df['first_active_month'].dt.month

    df['weekend'] = (df.first_active_month.dt.weekday >=5).astype(int)

    df['hour'] = df['first_active_month'].dt.hour
old_transactions.head()
#Adding Aggregate Columns.

def aggregate_transactions(history):

    

    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).astype(np.int64) * 1e-9

    agg_func = {

    'year':['sum','mean','nunique'],

    'weekend':['sum','mean','nunique'],

    'dayofweek':['min','max','mean','nunique'],

    'weekofyear':['min','max','mean','nunique'],

    'hour':['min','max','mean','nunique'],

    'category_1': ['sum', 'mean'],

    'category_2_1.0': ['mean'],

    'category_2_2.0': ['mean'],

    'category_2_3.0': ['mean'],

    'category_2_4.0': ['mean'],

    'category_2_5.0': ['mean'],

    'category_3_0': ['mean'],

    'category_3_1': ['mean'],

    'category_3_2': ['mean'],

    'merchant_id': ['nunique'],

    'merchant_category_id': ['nunique'],

    'state_id': ['nunique'],

    'city_id': ['nunique'],

    'subsector_id': ['nunique'],

    'installments': ['sum', 'mean', 'max', 'min', 'std'],

    'purchase_month': ['mean', 'max', 'min', 'std'],

    'month_lag': ['mean', 'max', 'min', 'std'],

    'month_diff': ['mean']

    }

    

    agg_history = history.groupby(['card_id']).agg(agg_func)

    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]

    agg_history.reset_index(inplace=True)

    

    df = (history.groupby('card_id').size().reset_index(name='transactions_count'))

    

    agg_history = pd.merge(df, agg_history, on='card_id', how='left')

    

    return agg_history
def aggregate_per_month(history):

    grouped = history.groupby(['card_id', 'month_lag'])



    agg_func = {

            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],

            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],

            }



    intermediate_group = grouped.agg(agg_func)

    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]

    intermediate_group.reset_index(inplace=True)



    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])

    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]

    final_group.reset_index(inplace=True)

    

    return final_group
old = aggregate_transactions(old_transactions)

old.columns = ['old_' + c if c != 'card_id' else c for c in old.columns]

old[:5]
new = aggregate_transactions(new_transactions)

new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]

new[:5]
final_group_old =  aggregate_per_month(old_transactions) 

final_group_old[:5]
final_group_new =  aggregate_per_month(new_transactions) 

final_group_new[:5]
#merge all created dataframes with train and test

print("...")

train = train.merge(historical_trans_group_df,on='card_id',how='left')

print("...")

test = test.merge(historical_trans_group_df,on='card_id',how='left')



print("...")

train = train.merge(new_merchant_trans_group_df,on='card_id',how='left')

print("...")

test = test.merge(new_merchant_trans_group_df,on='card_id',how='left')



print("...")

train = train.merge(old, on='card_id', how='left')

print("...")

test = test.merge(old, on='card_id', how='left')



print("...")

train = train.merge(new, on='card_id', how='left')

print("...")

test = test.merge(new, on='card_id', how='left')



print("...")

train = train.merge(final_group_old, on='card_id', how='left')

print("...")

test = test.merge(final_group_old, on='card_id', how='left')



print("...")

train = train.merge(final_group_new, on='card_id', how='left')

print("...")

test = test.merge(final_group_new, on='card_id', how='left')
#Adding a few more features

for df in [train, test]:

    for f in ['old_purchase_date_max','old_purchase_date_min','new_purchase_date_max','new_purchase_date_min']:

        df[f] = pd.to_datetime(df[f])

    df['old_first_buy'] = (df['old_purchase_date_min'] - df['first_active_month']).dt.days

    df['old_last_buy'] = (df['old_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    for f in ['old_purchase_date_max','old_purchase_date_min','new_purchase_date_max','new_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size']+df['old_card_id_size']

    df['card_id_cnt_total'] = df['new_card_id_count']+df['old_card_id_count']

    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['old_purchase_amount_sum']

    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['old_purchase_amount_mean']

    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['old_purchase_amount_max']

#Storing the new train and test externally

#test.to_csv('test.csv')

#train.to_csv('train.csv')

target = train['target']

del train['target']
features = [c for c in train.columns if c not in ['card_id','target', 'first_active_month','outliers']]

cat_features = ['feature_2', 'feature_3']
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
#Applying KFolds

folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

start = time.time()

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features],label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][features],label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
#Applying RepeatedKFolds

from sklearn.model_selection import RepeatedKFold

folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4950)

oof_2 = np.zeros(len(train))

predictions_2 = np.zeros(len(test))

feature_importance_df_2 = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=cat_features)

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=cat_features)



    num_round = 10000

    clf_r = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)

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



folds_stack = RepeatedKFold(n_splits=5, n_repeats=1, random_state=4590)

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
sub_df = pd.DataFrame({"card_id":test["card_id"].values})

sub_df["target"] = predictions_3

sub_df.to_csv("submit_tour_de_force.csv", index=False)