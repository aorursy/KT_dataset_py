import gc

import time

import warnings



warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

import seaborn as sns 

import xgboost as xgb

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier



import os

print(os.listdir("../input")) 
seed=2019

np.random.seed(seed)

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.shape
train_df.T.dtypes
missing_df = pd.DataFrame(train_df.isnull().sum(axis=0).reset_index())

missing_df.T
sns.countplot(data=test_df, x=train_df.target)
train_df.head()
X = train_df.iloc[:, 2:]

y = train_df.target
ids = test_df.id.to_frame()

test_df.drop('id', inplace=True, axis=1)
sc = StandardScaler(seed)

sc.fit(X)

X = sc.transform(X)

test_df = sc.transform(test_df)
params = {

    'tree_learner': 'serial',

#     'min_data_in_leaf': 2,

    

#     'objective': 'binary',

    'objective': 'binary',

    'learning_rate': 0.00000001,

    'num_leaves':2,  # Lower value for better accuracy

    'bagging_freq': 5,

    'bagging_fraction': 0.63,

    'boost_from_average': 'false',

    'boosting': 'gbdt',

    'feature_fraction': 0.4,

    'min_gain_to_split': 0.70,

    'max_depth': -1,

    'metric': 'auc',

    'max_bin' :4,

    'verbosity': 1,

#     'lambda_l2': 0.02,

#     'is_unbalance':True 

}

 
num_round = 2000000000

# Cross-validation

folds = StratifiedKFold(n_splits=21, shuffle=True, random_state=seed)

lstCV = folds.split(X, y)

# Train and Test Predication Vector

train_pred = np.zeros(len(X))

test_pred = np.zeros(len(test_df))
# Traning LightGBM  with the help of StratifiedKFold

for fold_, (trn_idx, val_idx) in enumerate(lstCV):

    print('Fold', fold_, 'started at', time.ctime())

    trn_data = lgb.Dataset(X[trn_idx], label=y[trn_idx])

    val_data = lgb.Dataset(X[val_idx], label=y[val_idx])

    clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=5000,

                    early_stopping_rounds=6000)

    train_pred[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    test_pred += clf.predict(test_df, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.10f}".format(roc_auc_score(y, train_pred)))
print("CV score: {:<8.10f}".format(roc_auc_score(y, train_pred)))
submission_lgb = pd.DataFrame({

    "id": ids["id"],

    "target": test_pred

})

submission_lgb.to_csv('submission.csv', index=False)