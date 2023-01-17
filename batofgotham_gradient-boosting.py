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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

df_test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')

df_train.shape, df_test.shape
df_train
train_revenue = df_train.pop('revenue')
df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)
df_train['Open Date'] = df_train['Open Date'].str.split('/').apply(lambda x : x[2])
df_test['Open Date'] = df_test['Open Date'].str.split('/').apply(lambda x : x[2])
df_train.shape, df_test.shape
df_train.drop(columns=["Id"],inplace=True)

df_test_index = df_test.pop('Id')
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore')
df_train_ohe = ohe.fit_transform(df_train)

df_train_ohe = df_train_ohe.todense()
df_test_ohe = ohe.transform(df_test)

df_test_ohe = df_test_ohe.toarray()
df_train_ohe.shape
df_test_ohe.shape
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
param = {

    "n_estimators": range(10,20,2),

    "learning_rate": [0.0001,0.001,0.01,0.1],

    "loss" : ['ls', 'lad', 'huber', 'quantile'],

    "min_samples_split": range(10,15,2),

    "min_samples_leaf": range(10,15,2),

    "max_depth": range(10,20,2),

    "alpha": [0,0.1,0.3,0.5,0.7,0.9]

}
#gbr = GradientBoostingRegressor(random_state=17)
#gbr_gcv = GridSearchCV(gbr,param,'neg_mean_squared_error',cv=5)

#gbr_gcv.fit(df_train_ohe,train_revenue)
#gbr_gcv.best_estimator_
gbr = GradientBoostingRegressor(alpha=0.1, ccp_alpha=0.0, criterion='friedman_mse',

                          init=None, learning_rate=0.1, loss='ls', max_depth=10,

                          max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=10, min_samples_split=10,

                          min_weight_fraction_leaf=0.0, n_estimators=10,

                          n_iter_no_change=None, presort='deprecated',

                          random_state=17, subsample=1.0, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)
gbr.fit(df_train_ohe,train_revenue)
train_revenue_predict = gbr.predict(df_train_ohe)

test_revenue = gbr.predict(df_test_ohe)
from sklearn.metrics import mean_squared_error



mse = mean_squared_error(train_revenue_predict,train_revenue)

rmse = np.sqrt(mse)

print(rmse)
df_submit = pd.DataFrame({'Id': df_test_index, 'Prediction': test_revenue})
df_submit.to_csv('submit.csv',index=False) 

df_submit.head()