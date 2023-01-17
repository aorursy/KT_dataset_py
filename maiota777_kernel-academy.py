# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series
#データの読み込み

#日付でパースした場合

df_train = pd.read_csv("../input/train.csv", index_col=0, parse_dates=['issue_d'])

df_test = pd.read_csv("../input/test.csv", index_col=0, parse_dates=['issue_d'])

#df_train = pd.read_csv("../input/train.csv", index_col=0)

#df_test = pd.read_csv("../input/test.csv", index_col=0)

df_test
#日付を文字列に変換

df_train['issue_d'] = df_train['issue_d'].astype(str)

df_test['issue_d'] = df_test['issue_d'].astype(str)
y_train = df_train.loan_condition #loan_conditionがターゲット

X_train = df_train.drop(['loan_condition'], axis=1)  #loan_conditionを外す

X_test = df_test
cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        print(col, X_train[col].nunique()) #ユニーク数
#カテゴリ処理

import category_encoders as ce

cats.remove('emp_title') #emp_titleを外す

cats

oe = ce.OrdinalEncoder(cols=cats, return_df=False) 
X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])

X_train.head()
#NaNを扱えないため補間

#X_train.fillna('0', inplace=True)

#X_test.fillna('0', inplace=True)

X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)

X_train
#K分割交差検定

from sklearn.model_selection import KFold



# 5分割交差検定

kf = KFold(n_splits=5, random_state=71, shuffle=True)
for train_ix, test_ix in kf.split(X_train, y_train):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_vala, y_vala = X_train.iloc[test_ix], y_train.iloc[test_ix]
# 使わない特徴量を外す

X_train_.drop(['emp_title', 'application_type'], axis=1, inplace=True)

X_vala.drop(['emp_title', 'application_type'], axis=1, inplace=True)

X_test.drop(['emp_title',  'application_type'], axis=1, inplace=True)
#グリッドサーチを行う

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier
skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
param_grid = {'learning_rate': np.logspace(-3, -1, 3), 

'max_depth':  np.linspace(5,12,4,dtype = int), 

'colsample_bytree': np.linspace(0.5, 1.0, 3), 'random_state': [71]}
fit_params = {"early_stopping_rounds": 20, "eval_metric": 'auc', "eval_set": [(X_vala, y_vala)]}
import lightgbm as lgb

from lightgbm import LGBMClassifier
clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71, importance_type='split', learning_rate=0.05, max_depth=-1, 

min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=9999, n_jobs=-1, 

num_leaves=31, objective=None, random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

subsample=0.9, subsample_for_bin=200000, subsample_freq=0)
gs = GridSearchCV(clf, param_grid, scoring='roc_auc', fit_params=fit_params, n_jobs=-1, cv=skf, verbose=True)
gs.fit(X_train_, y_train_)
#clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_vala, y_vala)])
#予測をする

#pred = clf.predict_proba(X_test.values)[:,1]

pred = gs.predict_proba(X_test.values)[:,1]
# sample_submission.csvのファイルを、submissionとして読みこむ

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

submission['loan_condition'] = pred # loan_conditionを上書き
submission
submission.to_csv('./submission0604_00.csv')