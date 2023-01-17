# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style('whitegrid')

import time

import lightgbm as lgb

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv('../input/train.csv')
train.shape
train.head()
test_df = pd.read_csv('../input/test.csv')
X = train.drop(["ID_code", "target"], axis=1)

Y = train["target"]

X_test = test_df.drop(["ID_code"], axis=1)
X_test.head()
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y

n_fold = 15

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
params = {'num_leaves': 13,

         'min_data_in_leaf': 80,

          'min_sum_hessian_in_leaf': 10.0,

         'objective': 'binary',

          'boost_from_average': False,

         'max_depth': -1,

         'learning_rate': 0.0083,

         'boost': 'gbdt',

         'bagging_freq': 5,

         'tree_learner': "serial",

         'bagging_fraction': 0.335,

         'feature_fraction': 0.041,

         #'reg_alpha': 1.738,

         #'reg_lambda': 4.99,

         'metric': 'auc',

         #'min_gain_to_split': 0.01077313523861969,

         #'min_child_weight': 19.428902804238373,

         'num_threads': 8}
prediction = np.zeros(len(X_test))

oof = np.zeros(len(X))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,Y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]

    

    X_tr, y_tr = augment(X_train.values, y_train.values)

    X_tr = pd.DataFrame(X_tr)

    

    train_data = lgb.Dataset(X_tr, label=y_tr)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    model = lgb.train(params,train_data,num_boost_round=1000000,

                    valid_sets = [train_data, valid_data],verbose_eval=3000,early_stopping_rounds = 4000)

    oof[valid_index] = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)

            

    #y_pred_valid = model.predict(X_valid)

    prediction += model.predict(X_test, num_iteration=model.best_iteration)/15

print("CV score: {:<8.5f}".format(roc_auc_score(Y, oof)))
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = prediction

sub.to_csv("submission.csv", index=False)