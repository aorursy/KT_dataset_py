# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



# string label to categorical values

from sklearn.preprocessing import LabelEncoder



# cross validation

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



# limit warning

import warnings  

warnings.filterwarnings('ignore')
# import data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# convert objects (strings) into numbers

print(train['SaleCondition'].unique())

for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))



print(train['SaleCondition'].unique())
# keep ID for submission

train_ID = train['Id']

test_ID = test['Id']



# split data for training

y_train = train['SalePrice']

X_train = train.drop(['Id','SalePrice'], axis=1)

X_test = test.drop('Id', axis=1)



# dealing with missing data

Xmat = pd.concat([X_train, X_test])
# before nan removals

msno.matrix(df=Xmat, figsize=(12, 8), color=(0.5,0,0))
# after nan removals

Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)

Xmat = Xmat.fillna(Xmat.median())

msno.matrix(df=Xmat, figsize=(12, 8), color=(0.5,0,0))
# add a new feature 'total sqfootage'

Xmat['TotalSF'] = Xmat['TotalBsmtSF'] + Xmat['1stFlrSF'] + Xmat['2ndFlrSF']
# normality check for the target

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax = sns.distplot(y_train)
# log-transform (log(1 + x)) the dependent variable for normality

y_train = np.log1p(y_train)



fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax = sns.distplot(y_train)
# restore original form

X_train = Xmat[:X_train.shape[0]]

X_test = Xmat[X_train.shape[0]:]



# interaction terms

X_train["Interaction"] = X_train["TotalSF"] * X_train["OverallQual"]

X_test["Interaction"] = X_test["TotalSF"] * X_test["OverallQual"]



X_train.head()
X_test.head()
print(y_train[:10])
# LightGBM

import lightgbm as lgb
# example hyperparameters for light GBM

lgbParams = {"num_leaves": 41, # model complexity

         "min_data_in_leaf": 10, # minimal number of data in one leaf

         "objective":'regression', # regression, binary, ...

         "metric": 'rmse', # mae, auc, binary_logloss, ...

         "num_iterations": 5000, # number of boosting iterations 

         "max_depth": -1, # max depth for tree model

         "max_bin": 255, # max number of bins that features are bucketed

         "learning_rate": 0.005, # learning rate

         "num_iterations": 500, # number of boosting iterations

         "boosting": "gbdt", # dart, goss, ...

         "feature_fraction": 0.9, # selected proportion of feature on each iteration

         "bagging_fraction": 0.9, # selected proportion of data without resampling

         "bagging_freq": 3, # perform bagging at every k iteration

         "lambda_l1": 0.1, # L1 regularization

         "verbosity": -1, # <0, fatal; 0: error, 1: info, >1: debug

         "random_state": 42, # random state

        }
## Visualize feature importance



# make a LightGBM dataset

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

d_train = lgb.Dataset(trainX, trainY)

d_eval = lgb.Dataset(testX, testY, reference=d_train)



# model training

LGBmodel = lgb.train(lgbParams, d_train, valid_sets=d_eval, early_stopping_rounds=100, verbose_eval=1000)



# feature importance

importance = LGBmodel.feature_importance()

ranking = np.argsort(-importance)

fig, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=importance[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()
# KFold cross-validation

n_folds = 5

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)



# placeholders

y_pred_lgb = np.zeros(np.shape(X_train)[0])

predictions_lgb = np.zeros(np.shape(X_test)[0])

for train_idx, test_idx in kf.split(X_train):

    # train, test split

    trainX, testX = X_train.iloc[train_idx], X_train.iloc[test_idx]

    trainY, testY = y_train.iloc[train_idx], y_train.iloc[test_idx]

    

    # make a LightGBM dataset

    d_train = lgb.Dataset(trainX, trainY)

    d_eval = lgb.Dataset(testX, testY, reference=d_train)

    

    # model training

    LGBmodel = lgb.train(lgbParams, d_train, valid_sets=d_eval, early_stopping_rounds=100, verbose_eval=1000)

    

    # cross validation score

    y_pred_lgb[test_idx] = LGBmodel.predict(testX, num_iteration=LGBmodel.best_iteration) 

    

    # prediction on test data

    predictions_lgb += LGBmodel.predict(X_test) / n_folds
# CV score

cvscore = np.sqrt(((y_pred_lgb - y_train.values) ** 2).mean()) # Root mean squared error

print("CV score = " + str(cvscore))
import xgboost as xgb
# example hyperparameters for XGB

xgbParams = {'learning_rate': 0.005,

             'max_depth': 10,

             'sabsample': 0.9,

             'colsample_bytree': 0.9,

             'n_estimators': 1000,

             'objective': 'reg:linear',

             'eval_metric': 'rmse',

             'alpha': 0.1,

             'verbosity': 0,

            }
# make a LightGBM dataset

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

d_train = xgb.DMatrix(trainX, trainY)

d_eval = xgb.DMatrix(testX, testY)

watchlist = [(d_eval, 'eval'), (d_train, 'train')]

n_rounds = 5000



# model training

XGBmodel = xgb.train(xgbParams, d_train, n_rounds, watchlist, early_stopping_rounds=100, verbose_eval=1000)



# feature importance

importance = np.asarray(list(XGBmodel.get_score(importance_type='gain').values()))

ranking = np.argsort(-importance)

fig, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=importance[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()
# placeholders

y_pred_xgb = np.zeros(np.shape(X_train)[0])

predictions_xgb = np.zeros(np.shape(X_test)[0])

for train_idx, test_idx in kf.split(X_train):

    # train, test split

    trainX, testX = X_train.iloc[train_idx], X_train.iloc[test_idx]

    trainY, testY = y_train.iloc[train_idx], y_train.iloc[test_idx]

    

    # make a LightGBM dataset

    d_train = xgb.DMatrix(trainX, trainY)

    d_eval = xgb.DMatrix(testX, testY)

    watchlist = [(d_eval, 'eval'), (d_train, 'train')]

    

    # model training

    XGBmodel = xgb.train(xgbParams, d_train, n_rounds, watchlist, early_stopping_rounds=100, verbose_eval=1000)

    

    # cross validation score

    y_pred_xgb[test_idx] = XGBmodel.predict(xgb.DMatrix(testX)) 

    

    # prediction on test data

    predictions_xgb += XGBmodel.predict(xgb.DMatrix(X_test)) / n_folds
# CV score

cvscore = np.sqrt(((y_pred_xgb - y_train.values) ** 2).mean()) # Root mean squared error

print("CV score = " + str(cvscore))
from catboost import CatBoostRegressor
Catmodel = CatBoostRegressor(iterations=2000, 

                          learning_rate = 0.005,

                          use_best_model = True,

                          eval_metric = 'RMSE',

                          loss_function = 'RMSE',

                          boosting_type = 'Ordered', # or Plain

                          verbose = 0)                                     
# train, test split

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.33, random_state=42)



# model training

Catmodel.fit(trainX, trainY, eval_set=[(trainX, trainY)], early_stopping_rounds=100)



# feature importance

importance = Catmodel.get_feature_importance()

ranking = np.argsort(-importance)

fig, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=importance[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()
# placeholders

y_pred_cat = np.zeros(np.shape(X_train)[0])

predictions_cat = np.zeros(np.shape(X_test)[0])

for train_idx, test_idx in kf.split(X_train):

    # train, test split

    trainX, testX = X_train.iloc[train_idx], X_train.iloc[test_idx]

    trainY, testY = y_train.iloc[train_idx], y_train.iloc[test_idx]

    

    # model training

    Catmodel.fit(trainX, trainY, eval_set=[(testX, testY)], early_stopping_rounds=100)

    

    # cross validation score

    y_pred_cat[test_idx] = Catmodel.predict(testX) 

    

    # prediction on test data

    predictions_cat += Catmodel.predict(X_test) / n_folds
# CV score

cvscore = np.sqrt(((y_pred_cat - y_train.values) ** 2).mean()) # Root mean squared error

print("CV score (CatBoost) = " + str(cvscore))
X1 = pd.DataFrame({'LGB': y_pred_lgb, 'XGB': y_pred_xgb, 'CAT': y_pred_cat})

X1.head(7)
# linear model

from sklearn.linear_model import BayesianRidge



# fitting

linearModel1 = BayesianRidge()

linearModel1.fit(X1, y_train)



# prediction

X2 = pd.DataFrame({'LGB': predictions_lgb, 'XGB': predictions_xgb, 'CAT': predictions_cat})

predictions1 = linearModel1.predict(X2)
# Neural network (multi-layer perceptron)

from sklearn.neural_network import MLPRegressor

MLPmodel = MLPRegressor(random_state=1220, activation='relu', solver='sgd', learning_rate='adaptive', tol=1e-06, hidden_layer_sizes=(250, ))



# Support vector machine (NuSVR)

from sklearn.svm import NuSVR

SVRmodel = NuSVR(kernel='rbf', degree=4, gamma='auto', nu=0.59, coef0=0.053, tol=1e-6)



# z-scoring: not necessary for Gradient boosting models 

X_train = (X_train - np.mean(X_train)) / np.std(X_train)

X_test = (X_test - np.mean(X_test)) / np.std(X_test)



# placeholders

y_pred_nn = np.zeros(np.shape(X_train)[0])

predictions_nn = np.zeros(np.shape(X_test)[0])

y_pred_svr = np.zeros(np.shape(X_train)[0])

predictions_svr = np.zeros(np.shape(X_test)[0])

for train_idx, test_idx in kf.split(X_train):

    # train, test split

    trainX, testX = X_train.iloc[train_idx], X_train.iloc[test_idx]

    trainY, testY = y_train.iloc[train_idx], y_train.iloc[test_idx]

    

    # model training

    MLPmodel.fit(trainX, trainY)

    SVRmodel.fit(trainX, trainY)

    

    # cross validation score

    y_pred_nn[test_idx] = MLPmodel.predict(testX) 

    y_pred_svr[test_idx] = SVRmodel.predict(testX) 

    

    # prediction on test data

    predictions_nn += MLPmodel.predict(X_test) / n_folds

    predictions_svr += SVRmodel.predict(X_test) / n_folds
# CV score

cvscore = np.sqrt(((y_pred_nn - y_train.values) ** 2).mean()) # Root mean squared error

print("CV score (Neural network) = " + str(cvscore))

cvscore = np.sqrt(((y_pred_svr - y_train.values) ** 2).mean()) # Root mean squared error

print("CV score (SVR) = " + str(cvscore))
# second Stacking ================================

X3 = pd.DataFrame({'MLP': y_pred_nn, 'SVR': y_pred_svr})



linearModel2 = BayesianRidge()

linearModel2.fit(X3, y_train)



# prediction

X4 = pd.DataFrame({'MLP': predictions_nn, 'SVR': predictions_svr})

predictions2 = linearModel2.predict(X4)



# final stacking (Gradient boosts + NN + SVR) ==========

X5 = pd.DataFrame({'LGB+XGB+CAT': linearModel1.predict(X1), 'NN+SVR': linearModel2.predict(X3)})



linearModel3 = BayesianRidge()

linearModel3.fit(X5, y_train)



# prediction

X4 = pd.DataFrame({'LGB+XGB+CAT': predictions1, 'NN+SVR': predictions2})

predictions = linearModel3.predict(X4)
# Don't forget that we log-transform our target. So we need to get it back!

predictions = np.expm1(predictions)



# submission

submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": predictions

})

submission.to_csv('houseprice_stacked.csv', index=False)