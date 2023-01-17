import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics

import warnings
import datetime
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))
#Loading Train and Test Data
df_train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
df_test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("{} observations and {} features in train set.".format(df_train.shape[0],df_train.shape[1]))
print("{} observations and {} features in test set.".format(df_test.shape[0],df_test.shape[1]))
df_train.head()
df_train["month"] = df_train["first_active_month"].dt.month
df_test["month"] = df_test["first_active_month"].dt.month
df_train["year"] = df_train["first_active_month"].dt.year
df_test["year"] = df_test["first_active_month"].dt.year
df_train['elapsed_time'] = (datetime.date(2018, 2, 1) - df_train['first_active_month'].dt.date).dt.days
df_test['elapsed_time'] = (datetime.date(2018, 2, 1) - df_test['first_active_month'].dt.date).dt.days
df_train.head()
df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])
df_test = pd.get_dummies(df_test, columns=['feature_1', 'feature_2'])
df_train.head()
df_hist_trans = pd.read_csv("../input/historical_transactions.csv")
df_hist_trans.head()
df_hist_trans = pd.get_dummies(df_hist_trans, columns=['category_2', 'category_3'])
df_hist_trans['authorized_flag'] = df_hist_trans['authorized_flag'].map({'Y': 1, 'N': 0})
df_hist_trans['category_1'] = df_hist_trans['category_1'].map({'Y': 1, 'N': 0})
df_hist_trans.head()
def aggregate_transactions(trans, prefix):  
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean', 'median'],
        'category_1': ['mean', 'median'],
        'category_2_1.0': ['mean', 'median'],
        'category_2_2.0': ['mean', 'median'],
        'category_2_3.0': ['mean', 'median'],
        'category_2_4.0': ['mean', 'median'],
        'category_2_5.0': ['mean', 'median'],
        'category_3_A': ['mean', 'median'],
        'category_3_B': ['mean', 'median'],
        'category_3_C': ['mean', 'median'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'median'],
        'installments': ['sum', 'mean', 'max', 'min', 'std', 'median'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
    }
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))
    
    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')
    
    return agg_trans
import gc
merch_hist = aggregate_transactions(df_hist_trans, prefix='hist_')
del df_hist_trans
gc.collect()
df_train = pd.merge(df_train, merch_hist, on='card_id',how='left')
df_test = pd.merge(df_test, merch_hist, on='card_id',how='left')
del merch_hist
gc.collect()
df_train.head()
df_new_trans = pd.read_csv("../input/new_merchant_transactions.csv")
df_new_trans.head()
df_new_trans = pd.get_dummies(df_new_trans, columns=['category_2', 'category_3'])
df_new_trans['authorized_flag'] = df_new_trans['authorized_flag'].map({'Y': 1, 'N': 0})
df_new_trans['category_1'] = df_new_trans['category_1'].map({'Y': 1, 'N': 0})
df_new_trans.head()
merch_new = aggregate_transactions(df_new_trans, prefix='new_')
del df_new_trans
gc.collect()
df_train = pd.merge(df_train, merch_new, on='card_id',how='left')
df_test = pd.merge(df_test, merch_new, on='card_id',how='left')
del merch_new
gc.collect()
df_train.head()
# TO DO add merchants
target = df_train['target']
drops = ['card_id', 'first_active_month', 'target']
use_cols = [c for c in df_train.columns if c not in drops]
features = list(df_train[use_cols].columns)
df_train[features].head()
print(df_train[features].shape)
print(df_test[features].shape) # Validation set
import contextlib
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train[features],
                                                    target, test_size=0.2)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

del(df_train)
del(X_train)
del(X_test)
dholdout = xgb.DMatrix(df_test[features])
target_holdout = df_test["card_id"].values

del(df_test)
# # https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters/notebook
# def xgb_evaluate(max_depth, 
#                  gamma,
#                  min_child_weight,
#                  max_delta_step,
#                  subsample,
#                  colsample_bytree):
    
#     global RMSEbest
#     global ITERbest
    
#     params = {'eval_metric': 'rmse',
#               'nthread' : 4,
#               'silent' : True,
#               'max_depth': int(max_depth),
#               'subsample': max(min(subsample, 1), 0),
#               'eta': 0.1,
#               'gamma': gamma,
#               'colsample_bytree': max(min(colsample_bytree, 1), 0),   
#               'min_child_weight': min_child_weight ,
#               'max_delta_step':int(max_delta_step),
#               'seed' : 16
#              }
    
#     folds = 5
#     cv_score = 0
    
#     print("\n Search parameters (%d-fold validation):\n %s" % (folds, params), file=log_file )
#     log_file.flush()
    
#     # Used around 1000 boosting rounds in the full model
#     cv_result = xgb.cv(params, 
#                        dtrain, 
#                        num_boost_round=2000, # can be increased 1k ~ 20k
#                        nfold=folds,
# #                       verbose_eval = 10,
#                        early_stopping_rounds = 100,
#                        metrics = 'rmse',
#                        show_stdv = True
#                       )    
    
#     val_score = cv_result['test-rmse-mean'].iloc[-1]
#     train_score = cv_result['train-rmse-mean'].iloc[-1]
#     print(' Stopped after %d iterations with train-rmse = %f val-rmse = %f ( diff = %f ) ' % ( len(cv_result), train_score, val_score, (train_score - val_score) ) )
#     if ( val_score > RMSEbest ):
#         RMSEbest = val_score
#         ITERbest = len(cv_result)

#     # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
#     return -1.0 * val_score
# log_file = open('Elo-RMSE-5fold-XGB-run-01-v1-full.log', 'a')
# RMSEbest = -1.
# ITERbest = 0

# xgb_bo = BayesianOptimization(xgb_evaluate, {
#                                     'max_depth': (2, 12),
#                                      'gamma': (0.001, 10.0),
#                                      'min_child_weight': (0, 20),
#                                      'max_delta_step': (0, 10),
#                                      'subsample': (0.4, 1.0),
#                                      'colsample_bytree' :(0.4, 1.0)})
# xgb_bo.explore({
#               'max_depth':            [3, 8, 3, 8, 8, 3, 8, 3],
#               'gamma':                [0.5, 8, 0.2, 9, 0.5, 8, 0.2, 9],
#               'min_child_weight':     [0.2, 0.2, 0.2, 0.2, 12, 12, 12, 12],
#               'max_delta_step':       [1, 2, 2, 1, 2, 1, 1, 2],
#               'subsample':            [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
#               'colsample_bytree':     [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
#               })
# %%time

# # Use the expected improvement acquisition function to handle negative numbers
# # Optimally needs quite a few more initiation points 15-20 and number of iterations 25-50

# print('-'*130)
# print('-'*130, file=log_file)
# log_file.flush()

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore')
#     #xgb_bo.maximize(init_points=1, n_iter=1, acq='ei', xi=0.0)
#     xgb_bo.maximize(init_points=10, n_iter=50, acq='ei', xi=0.01)
# print('-'*130)
# print('Final Results')
# print('Maximum XGBOOST value: %f' % xgb_bo.res['max']['max_val'])
# print('Best XGBOOST parameters: ', xgb_bo.res['max']['max_params'])
# print('-'*130, file=log_file)
# print('Final Result:', file=log_file)
# print('Maximum XGBOOST value: %f' % xgb_bo.res['max']['max_val'], file=log_file)
# print('Best XGBOOST parameters: ', xgb_bo.res['max']['max_params'], file=log_file)
# log_file.flush()
# log_file.close()

# history_df = pd.DataFrame(xgb_bo.res['all']['params'])
# history_df2 = pd.DataFrame(xgb_bo.res['all']['values'])
# history_df = pd.concat((history_df, history_df2), axis=1)
# history_df.rename(columns = { 0 : 'rmse'}, inplace=True)
# history_df['rmse'] = np.abs(history_df['rmse'])
# history_df.to_csv('Elo-RMSE-5fold-XGB-run-01-v1-grid.csv')
# pd.read_csv('Elo-RMSE-5fold-XGB-run-01-v1-grid.csv')
params1 = {'max_depth': 12.0, 
          'gamma': 3.1390339637040787, 
          'min_child_weight': 0.0, 
          'max_delta_step': 10.0, 
          'subsample': 1.0, 
          'colsample_bytree': 1.0}

params2 = {'colsample_bytree': 0.4,
          'gamma': 7.438375302732893,
          'max_delta_step': 3.0732617140069647,
          'max_depth': 9.926188389437563,
          'min_child_weight': 20.0,
          'subsample': 1.0}

params1['max_depth'] = int(params1['max_depth'])
params2['max_depth'] = int(params2['max_depth'])
%%time 

# Train a new model with the best parameters from the search
model1 = xgb.train(params1, dtrain, num_boost_round=500)
model2 = xgb.train(params2, dtrain, num_boost_round=500)
# # Predict on testing and training set
# y_pred = model2.predict(dtest)
# y_train_pred = model2.predict(dtrain)

# # Report testing and training RMSE
# print('Test error:', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('Train error:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
# # feature importance
# fig =  plt.figure(figsize = (12,8))
# axes = fig.add_subplot(111)
# xgb.plot_importance(model2,ax = axes,height =0.5)
# sns.despine()
# plt.tight_layout()
predictions1 = model1.predict(dholdout)
predictions2 = model2.predict(dholdout)
df_sub = pd.DataFrame({'card_id': target_holdout, 'target': predictions1})
df_sub.to_csv('p1.csv', index=False)
df_sub = pd.DataFrame({'card_id': target_holdout, 'target': predictions2})
df_sub.to_csv('p2.csv', index=False)
