# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/moneyball"))
print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/moneyball/test.csv')
print(train.head(2))
print(test.head(2))
print(train.columns)
train.describe()
print(train.isnull().sum())
print(train.info())
print(test.isnull().sum())
print(test.info())
plt.figure(figsize=(14,12))
foo = sns.heatmap(train.drop('ID',axis=1).corr(), vmax=0.6, square=True, annot=True)
fig, (axarr1, axarr2, axarr3) = plt.subplots(3, 3, figsize=(12, 12))
# sns.kdeplot(train.W, ax=axarr1[0])
sns.kdeplot(train.W, train.R, ax=axarr1[0])
axarr1[0].set_title("W vs R")
sns.kdeplot(train.W, train.RA, ax=axarr1[1]) 
axarr1[1].set_title("W vs RA")
sns.kdeplot(train.W, train.H, ax=axarr1[2])
axarr1[2].set_title("W vs H")
sns.kdeplot(train.W, train.HR, ax=axarr2[0])
axarr2[0].set_title("W vs HR")
sns.kdeplot(train.W, train['1B'], ax=axarr2[1])
axarr2[1].set_title("W vs 1B")
sns.kdeplot(train.W, train['2B'], ax=axarr2[2])
axarr2[2].set_title("W vs 2B")
sns.kdeplot(train.W, train['3B'], ax=axarr3[0])
axarr3[0].set_title("W vs 3B")
sns.kdeplot(train.W, train.BB, ax=axarr3[1])
axarr3[1].set_title("W vs BB")
sns.kdeplot(train.W, train.SO, ax=axarr3[2])
axarr3[2].set_title("W vs SO")
train.columns
cols = ['G', 'R', 'AB', 'H', 'BB','BBHBP', 'Outs', 'RA', 'BA', 'OPS', '3B', 'SO', 'CS']
ttrain, ttest = train_test_split(train, test_size = 0.2)
train_x = ttrain[cols]
#rain_x = ttrain
train_target = ttrain.W

test_x = ttest[cols]
#est_x = ttest
test_target = ttest.W
# feature extraction
train_cols = SelectKBest(score_func=chi2, k=15)
fit = train_cols.fit(ttrain.drop(['W', 'ID'], axis=1), ttrain.W)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(ttrain.drop(['W', 'ID'], axis=1))
# Summarize selected features
print(features[0:5, :])
print(fit.get_support())

dummy = ttrain.drop(['W', 'ID'], axis=1)
dummy.reset_index(drop=True, inplace=True)
dummy.iloc[:, fit.get_support()]
dummy_columns = ttrain.drop(['W', 'ID'], axis=1).columns
#print(dummy[fit.get_support()])
columns_selectkBest = dummy_columns[fit.get_support()]
train_x_selectKBest = ttrain[columns_selectkBest]
test_x_selectKBest = ttest[columns_selectkBest]
regressor = linear_model.LinearRegression()
rfe = RFE(regressor, 20)
fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
print("Num Features: {}".format(fit.n_features_)) 
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking: {}".format(fit.ranking_))

columns_rfe = dummy_columns[fit.get_support()]
train_x_rfe = ttrain[columns_rfe]
test_x_rfe = ttest[columns_rfe]
print(train_x_rfe.columns)
print(test_x_rfe.columns)
regressor = linear_model.LinearRegression()
regressor.fit(train_x, train_target)
pred =  regressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
print('Coefficients: \n', regressor.coef_)
regressor_selectKBest = linear_model.LinearRegression()
regressor_selectKBest.fit(train_x_selectKBest, train_target)
pred = regressor_selectKBest.predict(test_x_selectKBest)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
print('Coefficients: \n', regressor.coef_)
regressor_rfe = linear_model.LinearRegression()
regressor_rfe.fit(train_x_rfe, train_target)
pred = regressor_rfe.predict(test_x_rfe)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
print(len(ttrain.columns))
regressor = linear_model.LinearRegression()
r2_score_dict = dict()
for i in range(2, 24):
    rfe = RFE(regressor, i)
    fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
    columns_rfe = dummy_columns[fit.get_support()]
    train_x_rfe = ttrain[columns_rfe]
    test_x_rfe = ttest[columns_rfe]
    regressor_rfe = linear_model.LinearRegression()
    regressor_rfe.fit(train_x_rfe, train_target)
    pred = regressor_rfe.predict(test_x_rfe)
    r2_score_dict[i] = r2_score(test_target, pred)
# print("Num Features: {}".format(fit.n_features_)) 
# print("Selected Features: {}".format(fit.support_))
# print("Feature Ranking: {}".format(fit.ranking_))
# print(train_x_rfe.columns)
# print(test_x_rfe.columns)
print(max(r2_score_dict, key=r2_score_dict.get))

regressor = linear_model.LinearRegression()
rfe = RFE(regressor, 22)
fit = rfe.fit(ttrain.drop(['ID', 'W'], axis=1), ttrain.W)
columns_rfe = dummy_columns[fit.get_support()]
train_x_rfe = ttrain[columns_rfe]
test_x_rfe = ttest[columns_rfe]
regressor_rfe2 = linear_model.LinearRegression()
regressor_rfe2.fit(train_x_rfe, train_target)
pred = regressor_rfe2.predict(test_x_rfe)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(train_x,  train_target)
print(reg.alpha_)

ridgeRegressor = linear_model.Ridge(alpha=0.1)
ridgeRegressor.fit(train_x, train_target)
pred = ridgeRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
lassoRegressor = linear_model.Lasso(alpha=0.1)
lassoRegressor.fit(train_x, train_target)
pred = lassoRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
elasticRegressor = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
elasticRegressor.fit(train_x, train_target)
pred = elasticRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
lassoLarsRegressor = linear_model.LassoLars(alpha=0.1)
lassoLarsRegressor.fit(train_x, train_target)
pred = lassoLarsRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
bayesianRidge = linear_model.BayesianRidge()
bayesianRidge.fit(train_x, train_target)
pred = bayesianRidge.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
ardRegression = linear_model.ARDRegression(compute_score=True)
ardRegression.fit(train_x, train_target)
pred = ardRegression.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
huberRegressor = linear_model.HuberRegressor()
huberRegressor.fit(train_x, train_target)
pred = huberRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
import decimal
def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)
epsilon = drange(1, 10, 0.1)
dict_r2_score = dict()
for e in epsilon:
    huber = linear_model.HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                           epsilon=e)
    huber.fit(train_x, train_target)
    p = huber.predict(test_x)
    dict_r2_score[e] = r2_score(test_target, p)
max_id = max(dict_r2_score, key=dict_r2_score.get)
print(max_id, dict_r2_score[max_id])
huberRegressor = linear_model.HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, 
                                            epsilon=5.8)
huberRegressor.fit(train_x, train_target)
pred = huberRegressor.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
randomForest = ensemble.RandomForestRegressor(n_estimators=100)
rf_params = {'max_depth': range(5, 8),
             'max_features': range(5, 10)}
cv = GridSearchCV(cv=5, param_grid=rf_params, estimator=randomForest, n_jobs=-1, scoring='neg_mean_absolute_error')
cv.fit(train_x, train_target)
print(-cv.best_score_)
print(cv.best_params_)
randomForest = ensemble.RandomForestRegressor(n_estimators=100, max_depth=7, max_features=9, n_jobs=-1)
randomForest.fit(train_x, train_target)
pred = randomForest.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
randomForest2 = ensemble.RandomForestRegressor(n_estimators=100)
rf_params = {'max_depth': range(5, 15),
             'max_features': range(5, 23)}
cv = GridSearchCV(cv=5, param_grid=rf_params, estimator=randomForest, n_jobs=-1, scoring='neg_mean_absolute_error')
cv.fit(train_x, train_target)
print(-cv.best_score_)
print(cv.best_params_)
randomForest2 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=14, max_features=22, n_jobs=-1)
randomForest2.fit(train_x, train_target)
pred = randomForest2.predict(test_x)
print(mean_squared_error(test_target, pred))
print(r2_score(test_target, pred))
# pred_submit = ridgeRegressor.predict(test)
# pred_submit = huberRegressor.predict(test)
pred_submit = regressor_rfe2.predict(test[columns_rfe])
submit = pd.DataFrame(pred_submit, index=test['ID'], columns=['W'])
submit = submit.round()
submit.W = submit.W.astype('int64')
submit.to_csv('regressor_rfe2.csv', index_label='ID')
