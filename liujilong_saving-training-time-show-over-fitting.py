import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os
from sklearn.metrics import roc_auc_score
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint8',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
}
train_df = pd.read_csv("../input/train.csv", parse_dates=['click_time'], nrows=1000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
val_df = train_df[900000:]
train_df = train_df[:900000]

target = 'is_attributed'
predictors= ['app','os', 'channel', 'hour']
categorical = ['app', 'os', 'channel', 'hour']

print("preparing validation datasets")
xgtrain = lgb.Dataset(train_df[predictors].values,
                      label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
xgvalid = lgb.Dataset(val_df[predictors].values,
                      label=val_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
}
evals_results = {}

bst1 = lgb.train(lgb_params,
                 xgtrain,
                 valid_sets=[xgvalid],
                 valid_names=['valid'],
                 evals_result=evals_results,
                 num_boost_round=1000,
                 early_stopping_rounds=50,
                 verbose_eval=10)
def lgb_auc(bst, train_df, predictors, target, num_iter=None):
    if num_iter is None:
        num_iter = bst.best_iteration
    pred = bst.predict(train_df[predictors], num_iteration=num_iter)
    return roc_auc_score(train_df[target], pred)
train_auc = lgb_auc(bst1, train_df, predictors, target)
val_auc = lgb_auc(bst1, val_df, predictors, target)
print(f">>>>>> train_auc {train_auc}  val_auc  {val_auc}")
%matplotlib inline
points = 20
best_iter = bst1.best_iteration
cur_iter = bst1.current_iteration()
step_len = int(best_iter / points)
iters = range(0, cur_iter, step_len)
train_auc = [lgb_auc(bst1, train_df, predictors, target, ite)
             for ite in iters]
val_auc = [lgb_auc(bst1, val_df, predictors, target, ite)
             for ite in iters]
import matplotlib.pyplot as plt
plt.plot(iters, train_auc, 'r', label="train_auc")
plt.plot(iters, val_auc, 'g', label="valid_auc")
plt.legend(bbox_to_anchor=(0.7, 0.3), loc=2, borderaxespad=0.)
plt.show()