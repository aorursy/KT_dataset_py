# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import xgboost as xgb

from sklearn import datasets

from sklearn.cross_validation import train_test_split
# Load data

boston = datasets.load_boston()

X_train, X_test, y_train, y_test = train_test_split(

       boston.data, boston.target, test_size=0.33, random_state=42)
dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)

watchlist = [(dtrain, "train"), (dvalid, "eval")]

params = {"objective": "reg:linear",

          "eval_metric": "rmse",

          "eta": 0.01,

          "max_depth": 6,

          "silent": 1,

          "nthread": 1}

num_boost_round = 100

gbm = xgb.train(params, dtrain, num_boost_round,

                evals=watchlist, verbose_eval=True)

y_pred = gbm.predict(dvalid)
np.sqrt(np.mean((y_pred - y_test) ** 2))