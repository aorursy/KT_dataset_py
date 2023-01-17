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
train = pd.read_csv('/kaggle/input/allstate-purchase-prediction-challenge/train.csv.zip')

test = pd.read_csv('/kaggle/input/allstate-purchase-prediction-challenge/test_v2.csv.zip')

train.shape
train.isnull().sum()
pd.set_option('display.max_columns', None)

train
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

##########################  Psudo Code  ####################



#     time ===> extract hours

#     state, location ===> drop for now

#     car_value ===> create dummy variables

#     risk_factor ===> NaN predict with mean

#     all missing values predict with mean for now

train_cl = train.copy()

# time extract hour

train_cl['time'] = train["time"].str[0:2].astype('int32')

# state, location

train_cl = train_cl.drop(['state','location'],axis=1)

# car value to dummy variables

train_cl = pd.get_dummies(train_cl, columns=['car_value'])

# columns contain missing values are

# car_value, 1531

# risk_factor, 240418

# C_previous, 18711

# duration_previous, 18711

# impute missing values with mean for now

train_cl = train_cl.fillna(train_cl.mean())
from sklearn.model_selection import train_test_split

from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb



X = train_cl.drop(["record_type","customer_ID"],axis=1)

y = train_cl['record_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

X_train.shape
help(lgb.LGBMClassifier)
estimator = lgb.LGBMClassifier(learning_rate = 0.1, metric = 'l1', 

                        n_estimators = 10000)





param_grid = {

    'num_leaves ': [x for x in range(10,50,1)]

}

gridsearch = GridSearchCV(estimator, param_grid)



gridsearch.fit(X_train, y_train,

        eval_set = [(X_test, y_test)],

        eval_metric = ['auc', 'binary_logloss'],

        early_stopping_rounds = 5)


print('Best parameters found by grid search are:', gridsearch.best_params_)