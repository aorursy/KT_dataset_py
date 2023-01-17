# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import preprocessing

from sklearn import model_selection

from sklearn import metrics

import xgboost

import time

import lightgbm

from catboost import CatBoostClassifier
df = pd.read_csv('/kaggle/input/telecom-churn/telecom_churn.csv')
df.head()
d = {'Yes' : 1, 'No' : 0}



df['International plan'] = df['International plan'].map(d)

df['Voice mail plan'] = df['Voice mail plan'].map(d)
df['Churn'] = df['Churn'].astype('int64')
X = df.drop(['Churn', 'State', 'Area code'], axis=1)

y = df['Churn']
X_train, X_test, y_train, y_test = model_selection.train_test_split(

    X, y, stratify=y,  test_size=0.33, random_state=42

)
list_time_xg = []

list_error_xg = []

list_time_cat = []

list_error_cat = []

list_time_lght = []

list_error_lght = []
for i in range(1, 6):

    #Xgboost

    start_time = time.time()

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

              'learning_rate': 0.01, 'loss': 'ls'}

    xgb = xgboost.XGBRegressor(**params)

    xgb.fit(X_train, y_train)

    all_time = time.time() - start_time

    list_time_xg.append(all_time) 

    mse = metrics.mean_squared_error(y_test, xgb.predict(X_test))

    list_error_xg.append(mse) 

    #Catboost

    start_time = time.time()

    cat = CatBoostClassifier(iterations=2,

                               learning_rate=1,

                               depth=2)

    cat.fit(X_train, y_train)

    all_time = time.time() - start_time

    list_time_cat.append(all_time) 

    mse = metrics.mean_squared_error(y_test, cat.predict(X_test))

    list_error_cat.append(mse) 

    #lightgbm

    start_time = time.time()

    parameters = {

        'application': 'binary',

        'objective': 'binary',

        'metric': 'auc',

        'is_unbalance': 'true',

        'boosting': 'gbdt',

        'num_leaves': 31,

        'feature_fraction': 0.5,

        'bagging_fraction': 0.5,

        'bagging_freq': 20,

        'learning_rate': 0.05,

        'verbose': 0

    }



    lgbm = lightgbm.LGBMClassifier(**parameters)

    lgbm.fit(X_train, y_train)

    all_time = time.time() - start_time

    list_time_lght.append(all_time) 

    mse = metrics.mean_squared_error(y_test, lgbm.predict(X_test))

    list_error_lght.append(mse) 

    
list_boost = ['xgboost', 'catboost', 'lightgbm']

list_time = [np.mean(list_time_xg), np.mean(list_time_cat), np.mean(list_time_lght)]

list_error = [np.mean(list_error_xg), np.mean(list_error_cat), np.mean(list_error_lght)]
data = {'model' : list_boost,

        'time' : list_time,

       'error': list_error}

df_res = pd.DataFrame(data)
df_res