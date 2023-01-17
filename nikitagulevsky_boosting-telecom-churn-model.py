# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost as cb

from sklearn import tree

from sklearn import ensemble

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets

from sklearn import utils

from sklearn import metrics

import xgboost

import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv ('/kaggle/input/telecom-churn/telecom_churn.csv')
df
Y = df['Churn']
X = df.drop(['Churn'], axis = 1)
class MyGradientBoostingRegressor(object): ### Создаем класс, который поможет нам посчитать абонент ушел или не ушел

    

    def __init__(self, n_estimators=100, max_depth=4, min_samples_split=2, learning_rate=0.01):

        self.n_estimators = n_estimators

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.learning_rate = learning_rate

        self.DecisionTreeRegressorArr = []

        self.train_score_ = []

        

    def fit(self, X, y):

        params = {'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split}

        e = y

        pred = np.zeros(len(y))

        for i in range(self.n_estimators):

            dt = tree.DecisionTreeRegressor(**params)

            dt.fit(X, e)

            pred_current = dt.predict(X)

            pred += pred_current

            self.train_score_.append(metrics.mean_squared_error(y, pred))

            e = 2 * self.learning_rate * (y - pred)

            self.DecisionTreeRegressorArr.append(dt)

    

    def predict(self, X):

        pred = np.zeros(len(X))

        for dt in self.DecisionTreeRegressorArr:

            pred += dt.predict(X)

        return pred

    

    def staged_predict(self, X):

        pred = np.zeros(len(X))

        for dt in self.DecisionTreeRegressorArr:

            pred += dt.predict(X)

            yield pred
np.random.seed(42)



X, y = utils.shuffle(X,Y, random_state=2)

X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]

X_test, y_test = X[offset:], y[offset:]



# Fit gradient boosting

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.02}

mygb = MyGradientBoostingRegressor(**params)



mygb.fit(X_train, y_train)

mse = metrics.mean_squared_error(y_test, mygb.predict(X_test))

print("Gradient Boosting MSE: %.4f" % mse)
metrics.r2_score(y_test, mygb.predict(X_test))
plt.plot(mygb.train_score_)

plt.show()
%%time

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

xgb = xgboost.XGBRegressor(**params)

xgb.fit(X_train, y_train)

print(xgb)

mse = metrics.mean_squared_error(y_test, xgb.predict(X_test))

print("xgboost MSE: %.4f" % mse)
from catboost import CatBoostClassifier
SEED = 1
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)

%%time

params = {'loss_function':'Logloss', # objective function

          'eval_metric':'AUC', # metric

          'verbose': 500, # output to stdout info about training process every 500 iterations

          'random_seed': SEED

         }

cbc_1 = CatBoostClassifier(**params)

cbc_1.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)

          eval_set=(X_valid, y_valid), # data to validate on

          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score

          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)

         );
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
%%time

hyper_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': ['l2', 'auc'],

    'learning_rate': 0.005,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.7,

    'bagging_freq': 10,

    'verbose': 0,

    "max_depth": 8,

    "num_leaves": 128,  

    "max_bin": 512,

    "num_iterations": 3000,

    "n_estimators": 100

}

gbm = lgb.LGBMRegressor(**hyper_params)

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='l1',

        early_stopping_rounds=1000)