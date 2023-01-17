# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print (train_df.shape, test_df.shape)

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



sc = StandardScaler()

X_std = sc.fit_transform(train_df.values[:, 1:])

y = train_df.values[:, 0]



test_std = sc.fit_transform(test_df.values)



print (X_std.shape, y.shape)

print (test_std.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.1)

print (X_train.shape, y_train.shape)

print (X_valid.shape, y_valid.shape)
param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective", "multi:softmax"), ("eval_metric", "merror"), ("alpha", 8), ("lambda", 2), ("num_class", 10)]

n_rounds = 600

early_stopping = 50

    

d_train = xgb.DMatrix(X_train, label=y_train)

d_val = xgb.DMatrix(X_valid, label=y_valid)

eval_list = [(d_train, "train"), (d_val, "validation")]

bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
d_test = xgb.DMatrix(data=test_std)

y_pred = bst.predict(d_test)