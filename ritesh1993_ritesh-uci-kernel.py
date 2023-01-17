# Importing packages

import plotly_express as px

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
sns.set(style="ticks")

g = sns.countplot(x="target", data=data, palette="bwr")

sns.despine()

g.figure.set_size_inches(6,4)

plt.show()
train = data.copy()

train_y = data['target']

del train['target']
# Dividing the data into train test split

X, Xcv, y, ycv = train_test_split(train,train_y,test_size = 0.2,random_state=0)
from sklearn.metrics import accuracy_score

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials



def objective(space):

    # Instantiate the classifier

    clf = xgb.XGBClassifier(n_estimators =1000,colsample_bytree=space['colsample_bytree'],

                           learning_rate = .3,

                            max_depth = int(space['max_depth']),

                            min_child_weight = space['min_child_weight'],

                            subsample = space['subsample'],

                           gamma = space['gamma'],

                           reg_lambda = space['reg_lambda'])

    

    eval_set  = [( X, y), ( Xcv, ycv)]

    # Fit the classsifier

    clf.fit(X, y,

            eval_set=eval_set, eval_metric="rmse",

            early_stopping_rounds=10,verbose=False)

    

    # Predict on Cross Validation data

    pred = clf.predict(Xcv)

    accuracy = accuracy_score(ycv, pred>0.5)

    return {'loss': -accuracy, 'status': STATUS_OK }
#Creating the hyper parameter Space for the XGBoost classifier



space ={'max_depth': hp.quniform("x_max_depth", 4, 16, 1),

        'min_child_weight': hp.quniform ('x_min_child', 1, 10, 1),

        'subsample': hp.uniform ('x_subsample', 0.7, 1),

        'gamma' : hp.uniform ('x_gamma', 0.1,0.5),

        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.7,1),

        'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1)

    }
trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=100,

            trials=trials)

print(best)