import pandas as pd
import numpy as  np

import lightgbm as lgb

import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['figure.figsize'] = 16, 8
data = pd.read_pickle("../input/future-sales-predict/data.pkl")
correlation = data.corr()
X_train = data[data['date_block_num']<33].drop(['date_block_num', 'item_cnt_month'], axis=1)
y_train = data[data['date_block_num']<33]['item_cnt_month'].values
X_val = data[data['date_block_num']==33].drop(['date_block_num', 'item_cnt_month'], axis=1)
y_val = data[data['date_block_num']==33]['item_cnt_month'].values
X_test = data[data['date_block_num']==34].drop(['date_block_num', 'item_cnt_month'], axis=1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape
train = lgb.Dataset(X_train, label=y_train)
valid = lgb.Dataset(X_val, label=y_val)
param = {
         'objective':'regression',
         'metric': 'rmse',
         'learning_rate':0.001,
         'max_depth':10,
         'bagging_fraction':0.8,
         'feature_fraction':0.8,
         'bagging_frequency': 6,
         'verbosity':-1,
         'random_state': 42}
evals_result = {}
model = lgb.train(params=param, train_set=train, valid_sets=[train, valid], num_boost_round=1000,
                  evals_result = evals_result,
                  early_stopping_rounds=10,
                  verbose_eval=100)
lgb.plot_metric(evals_result, metric='rmse')
lgb.plot_importance(model, max_num_features=10, importance_type='gain')
y_preds = model.predict(X_test, model.best_iteration)
y_preds = y_preds.clip(0,20)
submission = pd.DataFrame()
submission['ID'] = range(X_test.shape[0])
submission['item_cnt_month'] = y_preds
submission.to_csv("submit.csv", index=False)