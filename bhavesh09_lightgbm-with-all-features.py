# Essential libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:.6f}'.format
import os
import datetime
# Sklearn support libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
# LGBM
import lightgbm as lgb
# Plotting libraries
from plotnine import *
import matplotlib.pyplot as plt
import plotnine
from mizani.breaks import date_breaks
#os.chdir(r"C:\Users\bhavesh\Documents\Kaggle\Elo Merchant Category Recommendation")
train = pd.read_csv('../input/train.csv', parse_dates =['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates =['first_active_month'])
hist_trans = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])
new_trans = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
merchants = pd.read_csv('../input/merchants.csv')
dfs = [train, test, hist_trans, new_trans, merchants]
for d in dfs:
    print(d.shape)
# Imputing null value with most common value
test['first_active_month'] = pd.to_datetime(test['first_active_month'].fillna('2017-09-01 00:00:00'))
plotnine.options.figure_size = (14, 6)
ggplot(train, aes(x='target')) +\
geom_histogram(bins=30, fill='blue', color='black') +\
scale_x_continuous(breaks=range(-40, 20, 10))
train['dayDiffActiveMonth'] = (train['first_active_month'] - pd.to_datetime('2018-02-01 00:00:00')).dt.days
test['dayDiffActiveMonth'] = (test['first_active_month'] - pd.to_datetime('2018-02-01 00:00:00')).dt.days
hist_trans.authorized_flag = (hist_trans.authorized_flag=='Y').astype('int')
new_trans.authorized_flag = (new_trans.authorized_flag=='Y').astype('int')
hist_trans_summ = hist_trans.groupby('card_id')['authorized_flag'].agg(['count', 'sum', 'mean'])
hist_trans_summ = hist_trans_summ.reset_index()
hist_trans_summ.columns = ['card_id','total_transactions','successful_transactions','successful_transactions_prop']
hist_trans_summ['failed_transactions'] = hist_trans_summ.total_transactions - hist_trans_summ.successful_transactions
hist_trans['failed_trans_dt_rnk'] = hist_trans[hist_trans.authorized_flag==0].groupby('card_id')['purchase_date'].rank(ascending=False)
hist_trans[['last_falied_trans_lag','last_failed_trans_amount']] = hist_trans[hist_trans.failed_trans_dt_rnk==1][['month_lag','purchase_amount']]
hist_trans = hist_trans.drop('failed_trans_dt_rnk', axis=1)
hist_trans['last_falied_trans_lag'] = hist_trans['last_falied_trans_lag'].fillna(1)
hist_trans['last_failed_trans_amount']=hist_trans['last_failed_trans_amount'].fillna(0) # may need to rethink on this null imputation
train = train.merge(hist_trans_summ, how='left', on='card_id')
test = test.merge(hist_trans_summ, how='left', on='card_id')
del(hist_trans_summ)
hist_trans_summ = hist_trans[hist_trans.authorized_flag==0].groupby('card_id')['purchase_amount'].agg(['mean','max','sum','std']).reset_index()
hist_trans_summ.columns = ['card_id','failed_trans_amt_mean','failed_trans_amt_max','failed_trans_amt_sum','failed_trans_amt_std']
train = train.merge(hist_trans_summ, how='left', on='card_id')
test = test.merge(hist_trans_summ, how='left', on='card_id')
hist_trans.category_1 = (hist_trans.category_1=='Y').astype('int')
new_trans.category_1 = (new_trans.category_1=='Y').astype('int')
for c in ['category_1','installments','month_lag','purchase_amount']:
    hist_transt_amt_summ = hist_trans.groupby(['card_id']).agg({c : ['count', 'sum', 'mean', 'max', 'std']}).reset_index()
    hist_transt_amt_summ.columns = ['card_id', 'hist_'+ c +'_cnt', 'hist_'+ c +'_sum', 
                                    'hist_'+ c +'_mean',  'hist_'+ c +'_max', 'hist_'+ c +'_std']
    train = train.merge(hist_transt_amt_summ, how='left', on='card_id')
    test = test.merge(hist_transt_amt_summ, how='left', on='card_id')
    del(hist_transt_amt_summ)
    
    new_transt_amt_summ = new_trans.groupby(['card_id']).agg({c : ['count', 'sum', 'mean', 'max', 'std']}).reset_index()
    new_transt_amt_summ.columns = ['card_id', 'new_'+ c +'_cnt', 'new_'+ c +'_sum', 
                                    'new_'+ c +'_mean',  'new_'+ c +'_max', 'new_'+ c +'_std']
    train = train.merge(new_transt_amt_summ, how='left', on='card_id')
    test = test.merge(new_transt_amt_summ, how='left', on='card_id')
    del(new_transt_amt_summ)
train = train.fillna(0)
test = test.fillna(0)
X_train = train.drop(['target', 'first_active_month', 'card_id'], axis=1, inplace=False)
y_train = train.target
X_test = test.drop([ 'first_active_month', 'card_id'], axis=1, inplace=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape, X_test.shape)
train_columns = X_train.columns
lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(train_columns))
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=list(train_columns))
params = {
    'objective': 'regression_l2',
    'metric': { 'rmse'},
    'num_leaves': 2000,
    'learning_rate': 0.1,
    'feature_fraction': 1,
    'verbose': 0,
    'max_depth' : 6,
    'min_data_in_leaf' : 8
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50) #3.76446
test_lgbm_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration)
submission = pd.DataFrame(test_lgbm_pred, index=X_test.index)
submission.columns=['target']
submission = pd.concat([test.card_id, submission], axis=1)
submission.to_csv('submission_lgbm.csv', index=False)
lgb_fi = pd.DataFrame(gbm.feature_importance(), index=train_columns).reset_index()
lgb_fi.columns = ['column','score']
