# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import xgboost

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_train.head()
y = df_train.label.values

X = df_train.drop('label', axis=1).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)




dtrain = xgboost.DMatrix(X_train, label=y_train)

dval = xgboost.DMatrix(X_val, label=y_val)



params = {

    'max_depth': 5,                 # the maximum depth of each tree

    'eta': 0.5,                     # the training step for each iteration

    'eta_decay': 0.9,

    'min_eta': 0.05,

    'silent': 0,                    # logging mode - quiet

    'objective': 'multi:softmax',   # multiclass classification using the softmax objective

    'num_class': 10,                 # the number of classes that exist in this datset

    'early_stopping_rounds': 10,

    'n_estimators': 5000

}  



def eval_acc(preds, dtrain):

    labels = dtrain.get_label()

    return 'acc', metrics.accuracy_score(labels, preds)



results = {}

model = xgboost.train(params, dtrain, num_boost_round=100, evals=[(dval, 'val')], evals_result=results, feval=eval_acc)
df_test = pd.read_csv('../input/test.csv')

df_test.head()
x_test = df_test.values

dtest = xgboost.DMatrix(x_test)
preds = model.predict(dtest)

print(preds.shape)

preds[:2]
df_sub = pd.read_csv('../input/sample_submission.csv')

df_sub.head()
df_sub.Label = preds.astype(np.int32)

df_sub.head()
df_sub.to_csv('submission.csv', index=False)