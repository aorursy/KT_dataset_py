# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
import gc
import os

from sklearn.model_selection import KFold

import time
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt 
import seaborn as sns 
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
print("shape of train : ",train.shape)
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("shape of test : ",test.shape)

ht = pd.read_csv("../input/historical_transactions.csv")
import datetime

for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['start_year'] = df['first_active_month'].dt.year
    df['start_month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

ytrain = train['target']
del train['target']
# binarize the categorical variables where it makes sense
ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})
ht['category_1'] = ht['category_1'].map({'Y':1, 'N':0})

ht['category_2x1'] = (ht['category_2'] == 1) + 0
ht['category_2x2'] = (ht['category_2'] == 2) + 0
ht['category_2x3'] = (ht['category_2'] == 3) + 0
ht['category_2x4'] = (ht['category_2'] == 4) + 0
ht['category_2x5'] = (ht['category_2'] == 5) + 0
ht['category_3A'] = (ht['category_3'].astype(str) == 'A') + 0
ht['category_3B'] = (ht['category_3'].astype(str) == 'B') + 0
ht['category_3C'] = (ht['category_3'].astype(str) == 'C') + 0
def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],        
        'city_id': ['nunique'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],

        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)

del ht
gc.collect()
new_merchant = pd.read_csv("../input/new_merchant_transactions.csv")
new_merchant.head(5)

new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})

new_merchant['category_1'] = new_merchant['category_1'].map({'Y':1, 'N':0})
new_merchant['category_3A'] = (new_merchant['category_3'].astype(str) == 'A') + 0
new_merchant['category_3B'] = (new_merchant['category_3'].astype(str) == 'B') + 0
new_merchant['category_3C'] = (new_merchant['category_3'].astype(str) == 'C') + 0

new_merchant['category_2x1'] = (new_merchant['category_2'] == 1) + 0
new_merchant['category_2x2'] = (new_merchant['category_2'] == 2) + 0
new_merchant['category_2x3'] = (new_merchant['category_2'] == 3) + 0
new_merchant['category_2x4'] = (new_merchant['category_2'] == 4) + 0
new_merchant['category_2x5'] = (new_merchant['category_2'] == 5) + 0
new_merchant['purchase_date'] = pd.DatetimeIndex(new_merchant['purchase_date']).\
                                      astype(np.int64) * 1e-9
def aggregate_new_transactions(new_trans):    
    
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2': ['nunique'],
        'category_3A': ['sum'],
        'category_3B': ['sum'],
        'category_3C': ['sum'],     
        'category_2x1': ['sum','mean'],
        'category_2x2': ['sum','mean'],
        'category_2x3': ['sum','mean'],
        'category_2x4': ['sum','mean'],
        'category_2x5': ['sum','mean'],        

        'city_id': ['nunique'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_lag': ['min', 'max'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']        
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)

del new_merchant
print(train.shape)
print(test.shape)

xtrain = pd.merge(train, new_trans, on='card_id', how='left')
xtest = pd.merge(test, new_trans, on='card_id', how='left')

del new_trans

print(xtrain.shape)
print(xtest.shape)

xtrain = pd.merge(xtrain, history, on='card_id', how='left')
xtest = pd.merge(xtest, history, on='card_id', how='left')

del history

print(xtrain.shape)
print(xtest.shape)

xtrain.head(3)
xtrain.drop('first_active_month', axis = 1, inplace = True)
xtest.drop('first_active_month', axis = 1, inplace = True)
categorical_feats = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(xtrain[col].values.astype('str')) + list(xtest[col].values.astype('str')))
    xtrain[col] = lbl.transform(list(xtrain[col].values.astype('str')))
    xtest[col] = lbl.transform(list(xtest[col].values.astype('str')))
df_all = pd.concat([xtrain, xtest])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = xtrain.shape[0]

xtrain = df_all[:len_train]
xtest = df_all[len_train:]
# prepare for modeling
id_train = xtrain['card_id'].copy(); xtrain.drop('card_id', axis = 1, inplace = True)
id_test = xtest['card_id'].copy(); xtest.drop('card_id', axis = 1, inplace = True)


nfolds = 10
folds = KFold(n_splits= nfolds, shuffle=True, random_state=15)
param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': 10,
         'learning_rate': 0.005,
         "min_child_samples": 100,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}
feature_importance_df = np.zeros((xtrain.shape[1], nfolds))
mvalid = np.zeros(len(xtrain))
mfull = np.zeros(len(xtest))


start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain.values, ytrain.values)):
    print('----')
    print("fold n°{}".format(fold_))
    
    x0,y0 = xtrain.iloc[trn_idx], ytrain[trn_idx]
    x1,y1 = xtrain.iloc[val_idx], ytrain[val_idx]
    
    trn_data = lgb.Dataset(x0, label= y0); val_data = lgb.Dataset(x1, label= y1)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=500, early_stopping_rounds = 150)
    mvalid[val_idx] = clf.predict(x1, num_iteration=clf.best_iteration)
    
    feature_importance_df[:, fold_] = clf.feature_importance()
    
    mfull += clf.predict(xtest, num_iteration=clf.best_iteration) / folds.n_splits
ximp = pd.DataFrame()
ximp['feature'] = xtrain.columns
ximp['importance'] = feature_importance_df.mean(axis = 1)

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=ximp.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
xsub = pd.DataFrame()
xsub['card_id']  = id_test
xsub['target'] = mfull
xsub.to_csv('sub_lgb.csv', index = False)
