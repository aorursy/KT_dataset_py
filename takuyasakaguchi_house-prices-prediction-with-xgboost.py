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


import pandas as pd

import numpy as np

from sklearn.cross_validation import KFold

from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score

import xgboost as xgb

import random

import seaborn as sns

random.seed(2)
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_test = pd.read_csv('../input/test.csv')
df_test.head()
plt.style.use('seaborn-talk')
# make a template to visualize prediction accuracy.

x = np.linspace(1,8)

y = x

plt.plot(x,y)

plt.xlim(-1,9)

plt.ylim(-1,9)
df_test['SalePrice'] = -1
df = pd.concat([df_train, df_test])
df = df.drop('Id', axis = 1)
s = df.dtypes

object_columns= s[s.values == 'object'].index.values
for item in object_columns:

    df[item] = df[item].astype('category')
df = pd.get_dummies(df)
columns = df.columns.values
df = df.fillna(0)
train = df[df.SalePrice >0]
test = df[df.SalePrice == -1]
y_train = train.SalePrice

y_test = test.SalePrice

x_train = train.drop("SalePrice", axis =1)

x_test = test.drop("SalePrice", axis =1)

y_train = y_train.values

y_test = y_test.values

x_train = x_train.values

x_test = x_test.values
import sklearn.preprocessing as pp

scaler = pp.StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.cross_validation import train_test_split

from sklearn import metrics
Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, random_state=0)
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
random_state = 0
dtrain = xgb.DMatrix(Xtrain, ytrain)

dvalid = xgb.DMatrix(Xtest, ytest)



dtest = xgb.DMatrix(x_test, y_test)

d_all_train = xgb.DMatrix(x_train, y_train)
num_boost_round = 2000
eta = 0.01

max_depth = 7

subsample = 0.4

colsample_bytree = 0.4





print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

params = {

        "objective": "reg:linear",

        "booster" : "gbtree",

        "eval_metric": "rmse",

        "eta": eta,

        "tree_method": 'exact',

        "max_depth": max_depth,

        "subsample": subsample,

        "colsample_bytree": colsample_bytree,

        "silent": 1,

        "seed": 0

}



early_stopping_rounds = 15

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
ypred = gbm.predict(dvalid)

print(rmse(ytest, ypred))

plt.scatter((ytest)/100000, (ypred)/100000)

plt.xlabel('prediction')

plt.ylabel('actual')



plt.plot(x,y)

plt.xlim(-1,9)

plt.ylim(-1,9)
rmse(ytest, ypred)
num_boost_round = 1106

gbm = xgb.train(params, d_all_train, num_boost_round)

prediction = gbm.predict(dtest)
prediction
df_submission = pd.read_csv('../input/sample_submission.csv')
df_submission.SalePrice = prediction
df_submission.to_csv('xgboost_regression.csv', index = False)