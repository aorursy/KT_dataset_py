# Data file

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import VotingRegressor
train = pd.read_csv('/kaggle/input/santander-value-prediction-challenge/train.csv', header=0)

train.head(10)
test = pd.read_csv('/kaggle/input/santander-value-prediction-challenge/test.csv', header=0)

test.head(10)
submission = pd.read_csv('/kaggle/input/santander-value-prediction-challenge/sample_submission.csv', header=0)

submission.head(10)
target_col = 'target'

drop_col = ['ID', 'target']



train_feature = train.drop(columns=drop_col)

train_target = train['target']

test_feature = test.drop(columns='ID')

submission_id = test['ID'].values



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)
# RandomForest==============



rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestRegressor')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



rf_prediction = rf.predict(test_feature)

rf_prediction