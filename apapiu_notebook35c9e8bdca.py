# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
train  = pd.read_csv("../input/train.csv")
train.shape
train.head(10)
train.isnull().sum().sum() #no Null values
train.dtypes
cat_feats = train.columns[train.dtypes == "object"]

train_cat = train[cat_feats]
#unique levels for each categorical feature:

train_cat.apply(lambda x: pd.unique(x).shape[0]).sort_values(ascending = False).head(10)
pd.Series(train.loss).hist()
pd.Series(np.log1p(train.loss).hist())
train.loc[:,"cont1":"cont4"].hist()

pylab.rcParams['figure.figsize'] = (10, 10)
train.loc[:,"cont5":"cont10"].hist()
train["log_target"] = np.log1p(train.loss)
sns.pairplot(data = train[["log_target", "cont1", "cont2", "cont3", "cont4"]].sample(1000))
sns.pairplot(hue = "cat3",

             data = train[["log_target", "cont9", "cont10", "cont11", "cont12", "cat3"]].sample(1000))
sns.pairplot(data = train[["log_target", "cont5", "cont6", "cont7", "cont8"]].sample(1000))
good_cols = np.setdiff1d(train.loc[:,"cat1":"cont14"].columns, 

             ["cat116", "cat110","cat109","cat113", "cat112"])
train_cat_large = train[["cat116", "cat110","cat109","cat113", "cat112"]]
X_int_cat = train_cat_large.apply(lambda x: pd.factorize(x)[0])
X = train[good_cols]
X = pd.get_dummies(X)
(X.shape, X_int_cat.shape)
X = pd.concat((X, X_int_cat), 1)
X.shape
from xgboost import XGBRegressor

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso
X_tr, X_val, y_tr, y_val = train_test_split(X, train["log_target"])
X_tr.shape
model = XGBRegressor(n_estimators=60, max_depth=6)
model.fit(X_tr, y_tr)
preds = model.predict(X_val)
pd.Series(preds).hist()
mean_absolute_error(np.expm1(preds), np.expm1(y_val)) 
xgb_params = {

    'seed': 0,

    'learning_rate': 0.3,

    'max_depth': 6,

}

dtrain = xgb.DMatrix(X_tr, label=y_tr)
res = xgb.cv(xgb_params, dtrain, num_boost_round=80, nfold=2, seed=0,

             early_stopping_rounds=10, verbose_eval=1, show_stdv=True)