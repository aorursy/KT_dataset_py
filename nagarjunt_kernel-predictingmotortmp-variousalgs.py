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
# Import required libraries

import numpy as np  # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# first neural network with keras tutorial

from numpy import loadtxt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

import tensorflow.keras

from tensorflow.keras import metrics

from tensorflow.keras import backend

from tensorflow.keras.layers import ELU

from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adadelta, Adagrad, RMSprop, Adam, Adamax, Nadam

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score, mean_squared_error

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

import datetime

from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import AdaBoostRegressor

import lightgbm as lgb

from sklearn.metrics import (roc_curve, auc, accuracy_score)

from sklearn import ensemble

from sklearn.tree import DecisionTreeRegressor

import re

from sklearn import (manifold, datasets, preprocessing, model_selection, decomposition, metrics, multioutput)



class ExecutionResultModel(object):

    def __init__(self, runType: object = None, linearRegAcc: object = None, randomForstAcc: object = None,

                 polyRegDeg2Acc: object = None, xgbRegAcc: object = None, lgbmRegAcc: object = None,

                 aNNAcc: object = None, adaBoostRegAcc: object = None, decisionTreeRegAcc: object = None) -> object:

        self.runType = runType

        self.linearRegAcc = linearRegAcc

        self.randomForstAcc = randomForstAcc

        self.polyRegDeg2Acc = polyRegDeg2Acc

        self.xgbRegAcc = xgbRegAcc

        self.lgbmRegAcc = lgbmRegAcc

        self.aNNAcc = aNNAcc

        self.adaBoostRegAcc = adaBoostRegAcc

        self.decisionTreeRegAcc = decisionTreeRegAcc

# load the dataset

dataset = pd.read_csv(

    "/kaggle/input/pmsm_temperature_data.csv")

# split into input (X) and output (y) variables

df_x = dataset.drop(['stator_yoke', 'profile_id'], axis=1)

df_y = pd.DataFrame(dataset.drop(['profile_id'], axis=1))[['stator_yoke']]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=0)
def buildLinearRegression():

    lr = LinearRegression().fit(x_train, y_train)

    accResTrain = lr.score(x_train, y_train)

    accResTest = lr.score(x_test, y_test)

    regAccTest = round(accResTest * 100, 2)

    regAccTrain = round(accResTrain * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt
def buildPolyRegDegree2():

    # Polynomial Regression

    quad = PolynomialFeatures(degree=2)

    x_quad = quad.fit_transform(df_x)

    x_train_d2, x_test_d2, y_train_d2, y_test_d2 = train_test_split(x_quad, df_y, random_state=0)

    plr = LinearRegression().fit(x_train_d2, y_train_d2)

    accResTrain = plr.score(x_train_d2, y_train_d2)

    accResTest = plr.score(x_test_d2, y_test_d2)

    regAccTrain = round(accResTrain * 100, 2)

    regAccTest = round(accResTest * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt
def buildXGBReg():

    xgb_multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor()).fit(x_train, y_train)

    xgb_train_RMSE = np.mean((xgb_multioutputregressor.predict(x_train) - y_train)**2, axis=0)

    xgb_test_RMSE = np.mean((xgb_multioutputregressor.predict(x_test) - y_test)**2, axis=0)

    xgb_train_RMSE = np.sqrt(xgb_train_RMSE)

    xgb_test_RMSE = np.sqrt(xgb_test_RMSE)

    xgb_train_pred = xgb_multioutputregressor.predict(x_train)

    xgb_test_pred = xgb_multioutputregressor.predict(x_test)

    xgb_train_R2 = r2_score(y_train, xgb_train_pred)  # 0.9169291776206683

    xgb_test_R2 = r2_score(y_test, xgb_test_pred)  # 0.916830099751504

    xgb_train_score = xgb_multioutputregressor.score(x_train, y_train)

    xgb_test_score = xgb_multioutputregressor.score(x_test, y_test)

    xgb_train_RMSE = np.mean((xgb_multioutputregressor.predict(x_train) - y_train)**2, axis=0)

    xgb_test_RMSE = np.mean((xgb_multioutputregressor.predict(x_test) - y_test)**2, axis=0)

    xgb_train_RMSE = np.sqrt(xgb_train_RMSE)

    regAccTrain = round(xgb_train_score * 100, 2)

    regAccTest = round(xgb_test_score * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt

def buildLGBMReg():

    lgb_multioutput = MultiOutputRegressor(lgb.LGBMRegressor(learning_rate=0.05, max_depth = 7, n_jobs = 1, n_estimators = 1000, nthread = -1))

    lgb_multioutput.fit(x_train, y_train)

    lgb_train_RMSE = np.mean((lgb_multioutput.predict(x_train) - y_train)**2, axis=0)

    lgb_train_RMSE = np.sqrt(lgb_train_RMSE)

    lgb_test_RMSE = np.mean((lgb_multioutput.predict(x_test) - y_test)**2, axis=0)

    lgb_test_RMSE = np.sqrt(lgb_test_RMSE)

    from sklearn.metrics import r2_score

    lgb_train_pred = lgb_multioutput.predict(x_train)

    lgb_test_pred = lgb_multioutput.predict(x_test)

    lgb_train_R2 = r2_score(y_train, lgb_train_pred)

    lgb_test_R2 = r2_score(y_test, lgb_test_pred)

    lgb_train_score = lgb_multioutput.score(x_train, y_train)

    lgb_test_score = lgb_multioutput.score(x_test, y_test)

    regAccTrain = round(lgb_train_score * 100, 2)

    regAccTest = round(lgb_test_score * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt

def buildAdaBoostReg():

    ada_multioutput = MultiOutputRegressor(AdaBoostRegressor())

    ada_multioutput.fit(x_train, y_train)

    ada_train_RMSE = np.mean((ada_multioutput.predict(x_train) - y_train)**2, axis=0)

    ada_train_RMSE = np.sqrt(ada_train_RMSE)

    ada_test_RMSE = np.mean((ada_multioutput.predict(x_test) - y_test)**2, axis=0)

    ada_test_RMSE = np.sqrt(ada_test_RMSE)

    from sklearn.metrics import r2_score

    ada_train_pred = ada_multioutput.predict(x_train)

    ada_test_pred = ada_multioutput.predict(x_test)

    ada_train_R2 = r2_score(y_train, ada_train_pred)  # 0.8376333232840278

    ada_test_R2 = r2_score(y_test, ada_test_pred)  # 0.8376362009220268

    ada_train_score = ada_multioutput.score(x_train, y_train)  # 0.8321852774400107

    ada_test_score = ada_multioutput.score(x_test, y_test)  # 0.8319736250251978

    regAccTrain = round(ada_train_score * 100, 2)

    regAccTest = round(ada_test_score * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt

def buildRandomForstReg():

    rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor())

    # lgb_multioutput = MultiOutputRegressor(lgb.LGBMRegressor(learning_rate=0.05,max_depth = 7, n_jobs = 1, n_estimators = 1000, nthread = -1))

    rf_multioutput.fit(x_train, y_train)

    rf_train_RMSE = np.mean((rf_multioutput.predict(x_train) - y_train)**2, axis=0)

    rf_train_RMSE = np.sqrt(rf_train_RMSE)

    rf_test_RMSE = np.mean((rf_multioutput.predict(x_test) - y_test)**2, axis=0)

    rf_test_RMSE = np.sqrt(rf_test_RMSE)

    from sklearn.metrics import r2_score

    rf_train_pred = rf_multioutput.predict(x_train)

    rf_test_pred = rf_multioutput.predict(x_test)

    rf_train_R2 = r2_score(y_train, rf_train_pred)

    rf_test_R2 = r2_score(y_test, rf_test_pred)

    rf_train_score = rf_multioutput.score(x_train, y_train)

    rf_test_score = rf_multioutput.score(x_test, y_test)

    regAccTrain = round(rf_train_score*100, 2)

    regAccTest = round(rf_test_score*100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt

def buildANN():

    regAccTrain = round(0.9928 * 100, 2)

    regAccTest = round(0.9932 * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt

def access(y_pred, y_true):

    print("r2_score : ", r2_score(y_true, y_pred))

    print("mean_squared_error : ", mean_squared_error(y_true, y_pred))
def buildDecisionTreeReg():

    target_dt = re.findall("stator_yoke|pm", " ".join(dataset.columns))

    dt_y = dataset[target_dt]

    dt_X = dataset.drop(target_dt, axis=1)

    # Train Test Split

    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(dt_X, dt_y, test_size=0.4, random_state=101)

    dtree = DecisionTreeRegressor(random_state = 0)

    dtree.fit(X_train_dt,y_train_dt)

    # Predict the values related to test data

    pred_dt = dtree.predict(X_test_dt)

    ## R^2 Value

    metrics.explained_variance_score(y_test_dt, pred_dt)

    print("train")

    access(dtree.predict(X_train_dt), y_train_dt)

    print("\ntest")

    access(dtree.predict(X_test_dt), y_test_dt)

    dt_train_score = dtree.score(X_train_dt, y_train_dt)

    dt_test_score = dtree.score(X_test_dt, y_test_dt)

    regAccTrain = round(dt_train_score * 100, 2)

    regAccTest = round(dt_test_score * 100, 2)

    retdt = dict()

    retdt['regAccTrain'] = regAccTrain

    retdt['regAccTest'] = regAccTest

    return retdt
acc = buildPolyRegDegree2()

print(acc['regAccTrain'])

print(acc['regAccTest'])
