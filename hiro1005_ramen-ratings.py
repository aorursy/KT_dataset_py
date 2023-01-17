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
ramen = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv', header=0)
ramen
brand = pd.get_dummies(ramen['Brand'])
style = pd.get_dummies(ramen['Style'])
country = pd.get_dummies(ramen['Country'])

ramen.sort_values(by=['Stars'], ascending=False).head(50)

del ramen['Review #']
del ramen['Variety']
del ramen['Top Ten']
del ramen['Brand']
del ramen['Style']
del ramen['Country']

ramen = ramen.drop(ramen.index[[122, 993, 32]])
total_ramen = pd.concat([ramen, brand, style, country], axis=1)

train_feature = total_ramen.drop(columns='Stars')
train_target = total_ramen['Stars']

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
import warnings
warnings.filterwarnings('ignore')

# RandomForest==============

rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestRegressor')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')
# SVR（Support Vector Regression）==============

svr = SVR(verbose=True)
svr.fit(X_train, y_train)
print('='*20)
print('SVR')
print(f'accuracy of train set: {svr.score(X_train, y_train)}')
print(f'accuracy of test set: {svr.score(X_test, y_test)}')

# LinearSVR==============

lsvr = LinearSVR(verbose=True, random_state=0)
lsvr.fit(X_train, y_train)
print('='*20)
print('LinearSVR')
print(f'accuracy of train set: {lsvr.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvr.score(X_test, y_test)}')

# SGDRegressor==============

sgd = SGDRegressor(verbose=0, random_state=0)
sgd.fit(X_train, y_train)
print('='*20)
print('SGDRegressor')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')