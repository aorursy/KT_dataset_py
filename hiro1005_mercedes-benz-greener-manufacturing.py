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
train = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip', header=0)

train.head(10)
test = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip', header=0)

test.head(10)
submission = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip', header=0)

submission.head(10)
train_mid = train.copy()

train_mid['train_or_test'] = 'train'



test_mid = test.copy()

test_mid['train_or_test'] = 'test'



test_mid['y'] = 9



alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True) 



print('The size of the train data:' + str(train.shape))

print('The size of the test data:' + str(test.shape))

print('The size of the submission data:' + str(submission.shape))

print('The size of the alldata data:' + str(alldata.shape))
hot_X0 = pd.get_dummies(alldata['X0'])

hot_X1 = pd.get_dummies(alldata['X1'])

hot_X2 = pd.get_dummies(alldata['X2'])

hot_X3 = pd.get_dummies(alldata['X3'])

hot_X4 = pd.get_dummies(alldata['X4'])

hot_X5 = pd.get_dummies(alldata['X5'])

hot_X6 = pd.get_dummies(alldata['X6'])

hot_X8 = pd.get_dummies(alldata['X8'])
alldata = pd.concat([alldata, hot_X0, hot_X1, hot_X2, hot_X3, hot_X4, hot_X5, hot_X6, hot_X8], axis=1)



del alldata['X0']

del alldata['X1']

del alldata['X2']

del alldata['X3']

del alldata['X4']

del alldata['X5']

del alldata['X6']

del alldata['X8']



train = alldata.query('train_or_test == "train"')

test = alldata.query('train_or_test == "test"')



target_col = 'y'

drop_col = ['ID', 'y', 'train_or_test']



train_feature = train.drop(columns=drop_col)

train_target = train[target_col]

test_feature = test.drop(columns=drop_col)

submission_id = test['ID'].values



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)
import warnings

warnings.filterwarnings('ignore')



# RandomForest==============



rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestRegressor')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



# k-近傍法（k-NN）==============



knn = KNeighborsRegressor()

knn.fit(X_train, y_train)

print('='*20)

print('KNeighborsRegressor')

print(f'accuracy of train set: {knn.score(X_train, y_train)}')

print(f'accuracy of test set: {knn.score(X_test, y_test)}')



# 決定木==============



decisiontree = DecisionTreeRegressor(max_depth=3, random_state=0)

decisiontree.fit(X_train, y_train)

print('='*20)

print('DecisionTreeRegressor')

print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')

print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')



# LinearRegression (線形回帰)==============



lr = LinearRegression()

lr.fit(X_train, y_train)

print('='*20)

print('LinearRegression')

print(f'accuracy of train set: {lr.score(X_train, y_train)}')

print(f'accuracy of test set: {lr.score(X_test, y_test)}')