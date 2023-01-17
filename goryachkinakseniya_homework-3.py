# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import datetime

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_clients = pd.read_csv('../input/x5-uplift-valid/data/clients2.csv', index_col='client_id',parse_dates=['first_issue_date','first_redeem_date'])

df_train = pd.read_csv('../input/x5-uplift-valid/data/train.csv', index_col='client_id')

df_test = pd.read_csv('../input/x5-uplift-valid/data/test.csv', index_col='client_id')

df_products = pd.read_csv('../input/x5-uplift-valid/data/products.csv', index_col='product_id')

df_test_purch = pd.read_csv('../input/x5-uplift-valid/test_purch/test_purch.csv',parse_dates=['transaction_datetime'])

df_train_purch = pd.read_csv('../input/x5-uplift-valid/train_purch/train_purch.csv',parse_dates=['transaction_datetime'])
df_test_purch['date'] = df_test_purch['transaction_datetime'].apply(pd.to_datetime)

df_train_purch['date'] = df_train_purch['transaction_datetime'].apply(pd.to_datetime)
last_cols = ['regular_points_received', 'purchase_sum']

last_train_month = df_train_purch[df_train_purch['transaction_datetime'] > '2019-02-18'].groupby(['client_id','transaction_id'])[last_cols].last()

last_test_month = df_test_purch[df_test_purch['transaction_datetime'] > '2019-02-18'].groupby(['client_id','transaction_id'])[last_cols].last()
features =  pd.concat([last_train_month.groupby('client_id')['purchase_sum'].count(),

                       last_train_month.groupby('client_id').sum()],axis = 1)



features_test =  pd.concat([last_test_month.groupby('client_id')['purchase_sum'].count(),

                       last_test_month.groupby('client_id').sum()],axis = 1)



features.columns = ['last_month_trans_count', 'regular_points_received_sum_last_month', 'purchase_sum_last_month']


merged_train = pd.concat([df_train,df_clients,features],axis = 1,sort = True)

merged_train['first_issue_date'] = merged_train['first_issue_date'].astype(int)/10**9

merged_train['first_redeem_date'] = merged_train['first_redeem_date'].astype(int)/10**9
merged_train.pop('client_id.1')

merged_train.pop('age')

merged_train.pop('gender')
treatment = merged_train[merged_train['treatment_flg'] == 1].drop('treatment_flg',axis = 1)

treatment_x = treatment.drop('target',axis = 1)

treatment_y = treatment['target']

control = merged_train[merged_train['treatment_flg'] == 0].drop('treatment_flg',axis = 1)

control_x = control.drop('target',axis = 1)

control_y = control['target']
import lightgbm as lgbm

params = {'learning_rate':0.03,'max_depth':4,'num_leaves':20,

             'min_data_in_leaf':3, 'application':'binary',

             'subsample':0.8, 'colsample_bytree': 0.8,

             'reg_alpha':0.01,'data_random_seed':42,'metric':'binary_logloss',

             'max_bin':416,'bagging_freq':3,'reg_lambda':0.01,'num_leaves':20             

    }

matrix = lgbm.Dataset(treatment_x, label=treatment_y)

cv_result = lgbm.cv(params, matrix, num_boost_round=5000,nfold=5, stratified=True, 

                              early_stopping_rounds=50, seed=42, verbose_eval=50)
treatment_model = lgbm.LGBMClassifier(n_estimators = len(cv_result['binary_logloss-mean']),**params)

treatment_model.fit(treatment_x,treatment_y)

control_model = lgbm.LGBMClassifier(n_estimators = len(cv_result['binary_logloss-mean']),**params)

control_model.fit(control_x,control_y)
df_test['target'] = 1

merged_test = pd.concat([df_test,df_clients,features_test],axis = 1,sort = True)

merged_test = merged_test[~merged_test['target'].isnull()].copy()
merged_test['first_issue_date'] = merged_test['first_issue_date'].astype(int)/10**9

merged_test['first_redeem_date'] = merged_test['first_redeem_date'].astype(int)/10**9
merged_test.pop('client_id.1')

merged_test.pop('age')

merged_test.pop('gender')
test_x = merged_test.drop('target',axis = 1)
preds_pos = treatment_model.predict_proba(test_x)[:,1]

preds_neg = control_model.predict_proba(test_x)[:,1]

pred = abs(preds_pos-preds_neg)

df_submission = pd.DataFrame({'client_id':test_x.index.values,'pred': pred})

df_submission.to_csv('submission.csv',index = False)