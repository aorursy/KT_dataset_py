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
import pandas as pd

import numpy as np

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
train.shape
train.var_126.plot(kind="hist")
#Merge test and train

merged = pd.concat([train, test])

#Saving the list of original features in a new list `original_features`.

original_features = merged.columns

merged.shape


from matplotlib import *

import matplotlib.pyplot as plt
train.target.value_counts().plot(kind="bar")
#well ,we'll go with smote if we get less accuracy
idx = features = merged.columns.values[0:200]

for df in [merged]:

    df['sum'] = df[idx].sum(axis=1)  

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    
merged["new"]=merged["var_108"]*merged["var_63"]


merged["log_73"]=np.log(merged["var_73"]+10)
merged["log_1"]=np.log(merged["var_1"]+17)
merged["log_1*73"]=merged["log_73"]*merged["log_1"]
merged["var_2_sq"]=merged["var_2"]*merged["var_2"]
#merged.isnull().sum()
train = merged.iloc[:len(train)]

X = train

train.head()
test = merged.iloc[len(train):]

test.head()
test2=test.drop(["target","ID_code"],1)
features=train.drop(["ID_code","target"],1)

target=train["target"]
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.33,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 12,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
num_round = 100000

# check random state 44000

folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=100)

oof = np.zeros(len(features))

predictions = np.zeros(len(test2))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(features.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(features.iloc[trn_idx], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(features.iloc[val_idx], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 2500)

    oof[val_idx] = clf.predict(features.iloc[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(test2, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
submission = pd.DataFrame({"ID_code": test.ID_code.values})

submission["target"] = predictions

submission.to_csv("submission.csv", index=False)
#smote

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
#test1=test.drop(labels="ID_code",axis=1)
submission.head()