# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import VotingClassifier, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import tree
import xgboost as xgb
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
houseDataTrain = pd.read_csv("../input/train.csv")
houseDataTest = pd.read_csv("../input/test.csv")
houseDataSampleSubmission = pd.read_csv("../input/sample_submission.csv")
houseDataTrain.head(10)
print(houseDataTrain.shape)
houseDataTest.head(5)
print(houseDataTest.shape)
houseDataSampleSubmission.head(5)
houseDataTrain.describe()
houseDataTrain.isnull().sum().sum()
nullColumns = houseDataTrain.columns[houseDataTrain.isnull().any()]
print(nullColumns)
for item in nullColumns:
    if houseDataTrain[item].dtype == 'float64':
        houseDataTrain[item].fillna((houseDataTrain[item].mean()), inplace = True)
    elif houseDataTrain[item].dtype == 'O':
        houseDataTrain[item].fillna(houseDataTrain[item].value_counts().index[0], inplace = True)
houseDataTrain.isnull().sum().sum()
houseDataTrain[nullColumns].head(10)
objectColumns = houseDataTrain.select_dtypes(['object']).columns
objectColumns
for item in objectColumns:
    houseDataTrain[item] = houseDataTrain[item].astype('category')
categoryColumns = houseDataTrain.select_dtypes(['category']).columns
houseDataTrain[categoryColumns] = houseDataTrain[categoryColumns].apply(lambda x: x.cat.codes)
houseDataTrain.head(10)
y = houseDataTrain['SalePrice']
x = houseDataTrain.drop(['Id', 'SalePrice'], axis = 1)
print(x.shape)
print(y.shape)
x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size = 0.25, random_state = 42)
print(x_train.shape)
print(x_cv.shape)
linearRegression = LinearRegression()
linearRegression.fit(x_train, y_train)
linearRegression.score(x_train, y_train)
prediction_linearRegression = linearRegression.predict(x_cv)
mean_squared_error(y_cv, prediction_linearRegression)
print(linearRegression.score(x_cv, y_cv))
print(r2_score(y_cv, prediction_linearRegression))
ridgeRegression = Ridge()
ridgeRegression.fit(x_train, y_train)
ridgeRegression.score(x_train, y_train)
prediction_ridgeRegression = ridgeRegression.predict(x_cv)
mean_squared_error(y_cv, prediction_ridgeRegression)
r2_score(y_cv, prediction_ridgeRegression)
lassoRegression = Lasso(alpha = 1, max_iter = 5000)
lassoRegression.fit(x_train, y_train)
lassoRegression.score(x_train, y_train)
prediction_lassoRegression = lassoRegression.predict(x_cv)
mean_squared_error(y_cv, prediction_lassoRegression)
r2_score(y_cv, prediction_lassoRegression)
elasticNet = ElasticNet(alpha = 1, l1_ratio = 0.9, max_iter = 5000, normalize = False)
elasticNet.fit(x_train, y_train)
elasticNet.score(x_train, y_train)
prediction_elasticNet = elasticNet.predict(x_cv)
mean_squared_error(y_cv, prediction_elasticNet)
r2_score(y_cv, prediction_elasticNet)
votingClassifier = VotingClassifier(estimators = [('Linear Regression', linearRegression), ('Ridge Regression', ridgeRegression), ('Lasso Regression', lassoRegression), ('Elastic Net Regression', elasticNet)], voting = 'hard')
baggingRegressor = BaggingRegressor(tree.DecisionTreeRegressor(random_state = 1))
baggingRegressor.fit(x_train, y_train)
baggingRegressor.score(x_train, y_train)
prediction_baggingRegressor = baggingRegressor.predict(x_cv)
mean_squared_error(y_cv, prediction_baggingRegressor)
r2_score(y_cv, prediction_baggingRegressor)
randomForestRegressor = RandomForestRegressor(n_estimators = 30)
randomForestRegressor.fit(x_train, y_train)
randomForestRegressor.score(x_train, y_train)
prediction_randomForest = randomForestRegressor.predict(x_cv)
mean_squared_error(y_cv, prediction_randomForest)
r2_score(y_cv, prediction_randomForest)
adaBoostRegressor = AdaBoostRegressor(n_estimators = 60)
adaBoostRegressor.fit(x_train, y_train)
adaBoostRegressor.score(x_train, y_train)
prediction_adaBoost = adaBoostRegressor.predict(x_cv)
mean_squared_error(y_cv, prediction_adaBoost)
r2_score(y_cv, prediction_adaBoost)
gradientBoostingRegressor = GradientBoostingRegressor(max_depth = 4)
gradientBoostingRegressor.fit(x_train, y_train)
gradientBoostingRegressor.score(x_train, y_train)
prediction_gradientBoost = gradientBoostingRegressor.predict(x_cv)
mean_squared_error(y_cv, prediction_gradientBoost)
r2_score(y_cv, prediction_gradientBoost)
xgBoost = xgb.XGBRegressor(max_depth = 4, learning_rate = 0.1, n_estimators = 500)
xgBoost.fit(x_train, y_train)
xgBoost.score(x_train, y_train)
prediction_xgBoost = xgBoost.predict(x_cv)
mean_squared_error(y_cv, prediction_xgBoost)
r2_score(y_cv, prediction_xgBoost)
params = {'learning_rate': 0.1}
train_data = lgb.Dataset(x_train, label = y_train)
lgbRegressor = lgb.train(params, train_data, 100)
prediction_lgbRegressor = lgbRegressor.predict(x_cv)
mean_squared_error(y_cv, prediction_lgbRegressor)
r2_score(y_cv, prediction_lgbRegressor)