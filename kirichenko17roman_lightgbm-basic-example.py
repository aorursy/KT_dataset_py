import numpy as np 

import pandas as pd 
train = pd.read_csv("/kaggle/input/corruption/coruppt_train.csv")

test = pd.read_csv("/kaggle/input/corruption/coruppt_test.csv")
y = train['is_corrupted'].values

del train['is_corrupted']

x = train.select_dtypes(exclude = ['object']).values
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, ParameterGrid

from sklearn.metrics import (roc_curve, auc, accuracy_score)



gridParams = {

    'lambda_l1': [1],

    'lambda_l2': [0],

    'n_estimators': [200],

    'num_leaves': [100],

    'feature_fraction': [0.6],

    'bagging_fraction': [0.9]

}



mdl = lgb.LGBMClassifier(

    learning_rate = 0.02,

    boosting_type = 'gbdt',

    objective = 'binary',

    task = 'train',

    metric = 'auc',

    metric_freq = 1,

    is_training_metric = True,

    max_bin = 255,

    bagging_freq = 5,

    n_jobs = -1

)



scoring = {'AUC': 'roc_auc'}



grid = GridSearchCV(mdl, gridParams, verbose=1, cv=5, scoring=scoring, n_jobs=-1, refit='AUC')



grid.fit(x, y)



print('Best parameters found by grid search are:', grid.best_params_)

print('Best score found by grid search is:', grid.best_score_)
grid.best_estimator_.fit(x, y)
ss = pd.read_csv("/kaggle/input/corruption/sample_submission.csv")



ss['is_corrupted'] = 1-grid.predict_proba(test.select_dtypes(exclude = ['object']).values)
ss.head()
ss.to_csv("subms.csv", index=False)