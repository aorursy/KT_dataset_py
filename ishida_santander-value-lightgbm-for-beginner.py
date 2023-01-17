import numpy as np
import pandas as pd
import os
import gc
import time
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
from contextlib import contextmanager
print(os.listdir("../input"))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Data Import
num_rows = None
train_df = pd.read_csv('../input/train.csv', nrows= num_rows)
test_df = pd.read_csv('../input/test.csv', nrows= num_rows)
df = pd.concat([train_df, test_df], axis=0)
sub_id = test_df.ID
print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))
del train_df, test_df
gc.collect()
X_train = df[df['target'].notnull()].drop(["ID", "target"], axis=1)
y_train = df[df['target'].notnull()].target
X_test = df[df['target'].isnull()].drop(["ID"], axis=1)
y_train = np.log1p(df[df['target'].notnull()].target)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "boosting_type":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "random_seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")
# feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])
sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test
sub_lgb = sub_lgb.set_index(sub_id)
sub_lgb.to_csv('sub_lgb.csv', encoding='utf-8-sig')