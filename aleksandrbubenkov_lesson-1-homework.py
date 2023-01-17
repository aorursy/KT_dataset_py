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
from sklearn import model_selection

from sklearn import metrics

import catboost

import xgboost

import lightgbm

import time
df = pd.read_csv('/kaggle/input/telecom-churn-datasets/churn-bigml-80.csv')

df.head()
df = df.drop(columns=['State', 'Area code'])
df['International plan'] = df['International plan'].astype('str').replace({'Yes': 1, 'No': 0})

df['Voice mail plan'] = df['Voice mail plan'].astype('str').replace({'Yes': 1, 'No': 0})

df['Churn'] = df['Churn'].astype('str').replace({'False': 0, 'True': 1})

df.head()
X, y = df.drop(columns=['Churn']), df['Churn']



X_train, X_test, y_train, y_test = model_selection.train_test_split(

    X, y, stratify=y, test_size=0.33, random_state=42)



params = [list(np.arange(100, 600, 100)), list(np.arange(0.05, 0, -0.01))]
#XGBoost

XGB = {'Model': 'XGBoost', 'n_estimators': [], 'learning_rate': [], 'Time': [], 'MSE': []}



for n, l in zip(params[0], params[1]):

    start_time = time.time()

    xgb = xgboost.XGBClassifier(n_estimators=n, learning_rate=l)

    xgb.fit(X_train, y_train)

    total_time = time.time() - start_time

    mse = metrics.mean_squared_error(y_test, xgb.predict(X_test))

    XGB['n_estimators'] += [n]

    XGB['learning_rate'] += [l]

    XGB['Time'] += [total_time]

    XGB['MSE'] += [mse]

XGB = pd.DataFrame(XGB)

XGB
#CatBoost

Cat = {'Model': 'CatBoost', 'n_estimators': [], 'learning_rate': [], 'Time': [], 'MSE': []}



for n, l in zip(params[0], params[1]):

    start_time = time.time()

    cat = catboost.CatBoostClassifier(n_estimators=n, learning_rate=l)

    cat.fit(X_train, y_train)

    total_time = time.time() - start_time

    mse = metrics.mean_squared_error(y_test, cat.predict(X_test))

    Cat['n_estimators'] += [n]

    Cat['learning_rate'] += [l]

    Cat['Time'] += [total_time]

    Cat['MSE'] += [mse]

Cat = pd.DataFrame(Cat)
Cat
#LightGBM

LGBM = {'Model': 'LightGBM', 'n_estimators': [], 'learning_rate': [], 'Time': [], 'MSE': []}



for n, l in zip(params[0], params[1]):

    start_time = time.time()

    lgbm = lightgbm.LGBMClassifier(n_estimators=n, learning_rate=l)

    lgbm.fit(X_train, y_train)

    total_time = time.time() - start_time

    mse = metrics.mean_squared_error(y_test, lgbm.predict(X_test))

    LGBM['n_estimators'] += [n]

    LGBM['learning_rate'] += [l]

    LGBM['Time'] += [total_time]

    LGBM['MSE'] += [mse]

LGBM = pd.DataFrame(LGBM)
LGBM
final = pd.concat([XGB, Cat, LGBM])

final