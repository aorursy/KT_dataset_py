import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
xTrain_oh = pd.read_csv("/kaggle/input/cs334-project-preprocess/xTrain_renfe_oh.csv")

xTest_oh = pd.read_csv("/kaggle/input/cs334-project-preprocess/xTest_renfe_oh.csv")

yTrain_oh = pd.read_csv("/kaggle/input/cs334-project-preprocess/yTrain_renfe_oh.csv")

yTest_oh = pd.read_csv("/kaggle/input/cs334-project-preprocess/yTest_renfe_oh.csv")
from sklearn.decomposition import NMF



def run_nmf(xTrain_oh, xTest_oh):

    approxTol=1

    maxK = np.max(xTrain_oh.shape[1])

    lastErr = np.finfo(float).max

    m = 0

    for k in range(1, maxK, 1):

        nmf = NMF(n_components=k)

        nmf.fit(xTrain_oh)

        err = nmf.reconstruction_err_

        if lastErr - err < approxTol or err < approxTol:

            m = k

            break

        lastErr = err

    print("NMF: number of columns is ", k)

    train_nmf = pd.DataFrame(nmf.transform(xTrain_oh))

    test_nmf = pd.DataFrame(nmf.transform(xTest_oh))

    

    train_nmf.to_csv("train_nmf.csv", index=False)

    test_nmf.to_csv("test_nmf.csv", index=False)

    return train_nmf, test_nmf

    

train_nmf, test_nmf = run_nmf(xTrain_oh, xTest_oh)

# These are commands used for reading the saved dataset. 

#train_nmf = pd.read_csv("../input/output-data/train_nmf.csv")

#test_nmf = pd.read_csv("../input/output-data/test_nmf.csv")
from sklearn.decomposition import PCA

def run_pca(xTrain, xTest):

    # set the shape to be the max

    pca = PCA(n_components=xTrain.shape[1])

    pca.fit(xTrain)

    # calculate number of components to get to 98%

    exp_var = pca.explained_variance_ratio_

    tot_var = np.cumsum(exp_var)

    k = np.argmax(tot_var > 0.98) + 1

    # refit it to this value

    print("PCA: number of PC is ", k)

    pca = PCA(n_components=k)

    train_pca = pca.fit_transform(xTrain)

    test_pca = pca.transform(xTest)

    

    train_pca = pd.DataFrame(train_pca)

    test_pca = pd.DataFrame(test_pca)

    

    train_pca.to_csv("train_pca.csv", index=False)

    test_pca.to_csv("test_pca.csv", index=False)

    

    return train_pca, test_pca



train_pca, test_pca = run_pca(xTrain_oh, xTest_oh)

# These are commands used for reading the saved dataset.

#train_pca = pd.read_csv("../input/output-data/train_pca.csv")

#test_pca = pd.read_csv("../input/output-data/test_pca.csv")
from sklearn.neighbors import KNeighborsRegressor as knn

#from skmultilearn.adapt import MLkNN as knn

def knn_acc(xTrain, yTrain, xTest, yTest):

    

    ###

    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}

    model = GridSearchCV(knn(), param_grid, cv=5)

    model = model.fit(xTrain, yTrain['price'])

    print("For KNN:")

    print("Best parameters found by grid search:")

    print(model.best_params_)

    print("Best CV score:")

    print(model.best_score_)

    

    preds = model.predict(xTest)

    error = sqrt(mean_squared_error(yTest['price'],preds))

    trainScore = model.score(xTrain, yTrain['price'])

    testScore = model.score(xTest, yTest['price'])

    r2 = r2_score(yTest['price'], preds)



    print()

    print('trainScore of knn: ', trainScore)

    print('testScore of knn: ', testScore)

    print('RMSE of test: ', error)

    print('R-square of test: ', r2)

    print()

    

    

knn_acc(xTrain_oh, yTrain_oh, xTest_oh, yTest_oh)

knn_acc(train_nmf, yTrain_oh, test_nmf, yTest_oh)

knn_acc(train_pca, yTrain_oh, test_pca, yTest_oh)
from sklearn.svm import SVR

def svr_acc(xTrain, yTrain, xTest, yTest):

    svr = SVR()

    svr.fit(xTrain, yTrain)

    preds = svr.predict(xTest)

    error = sqrt(mean_squared_error(yTest['price'],preds))

    r2 = r2_score(yTest['price'], preds)

    

    print('For SVR:')

    print('TrainScore: ', svr.score(xTrain, yTrain['price']))

    print('TestScore: ', svr.score(xTest, yTest['price']))

    print('RMSE of test: ', error)

    print('R-square of test: ', r2)

    print()

    



svr_acc(xTrain_oh, yTrain_oh, xTest_oh, yTest_oh)

svr_acc(train_nmf, yTrain_oh, test_nmf, yTest_oh)

svr_acc(train_pca, yTrain_oh, test_pca, yTest_oh)
from sklearn.tree import DecisionTreeRegressor as dtc

def dt_acc(xTrain, yTrain, xTest, yTest):

    

    ###

    param_grid = {'criterion': ['mse', 'mae'],

                 'max_depth': [x for x in range(5, 10)],

                 'min_samples_leaf': [x for x in range(2, 5)]}

    model = GridSearchCV(dtc(), param_grid, cv=5)

    model = model.fit(xTrain, yTrain['price'])

    print("For Decision Tree:")

    print("Best parameters found by grid search:")

    print(model.best_params_)

    print("Best CV score:")

    print(model.best_score_)

    

    preds = model.predict(xTest)

    error = sqrt(mean_squared_error(yTest['price'],preds))

    trainScore = model.score(xTrain, yTrain['price'])

    testScore = model.score(xTest, yTest['price'])

    r2 = r2_score(yTest['price'], preds)

    

    print()

    print('trainScore of dt: ', trainScore)

    print('testScore of dt: ', testScore)

    print('RMSE of test: ', error)

    print('R-square of test: ', r2)

    print()

    



dt_acc(xTrain_oh, yTrain_oh, xTest_oh, yTest_oh)

dt_acc(train_nmf, yTrain_oh, test_nmf, yTest_oh)

dt_acc(train_pca, yTrain_oh, test_pca, yTest_oh)
from sklearn.ensemble import RandomForestRegressor

def rf_acc(xTrain, yTrain, xTest, yTest):

    

    ###

    param_grid = {'n_estimators': [x for x in range(50, 100, 10)]}

    model = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)

    model = model.fit(xTrain, yTrain['price'])

    print("For Random Forest:")

    print("Best parameters found by grid search:")

    print(model.best_params_)

    print("Best CV score:")

    print(model.best_score_)

    

    preds = model.predict(xTest)

    error = sqrt(mean_squared_error(yTest['price'],preds))

    trainScore = model.score(xTrain, yTrain['price'])

    testScore = model.score(xTest, yTest['price'])

    r2 = r2_score(yTest['price'], preds)

    

    print()

    print('trainScore of rf: ', trainScore)

    print('testScore of rf: ', testScore)

    print('RMSE of test: ', error)

    print('R-square of test: ', r2)

    print()



rf_acc(xTrain_oh, yTrain_oh, xTest_oh, yTest_oh)

rf_acc(train_nmf, yTrain_oh, test_nmf, yTest_oh)

rf_acc(train_pca, yTrain_oh, test_pca, yTest_oh)
from xgboost import XGBRegressor

def xgb_acc(xTrain, yTrain, xTest, yTest):

    

    ###

    param_grid = {'colsample_bytree':[0.4,0.6,0.8],

                  'min_child_weight':[1.5,6,10],

                  'learning_rate':[0.1,0.2,0.5],

                  'max_depth':[3,5,7],

                  'n_estimators':[20,50,80]}

    model = GridSearchCV(XGBRegressor(), param_grid, cv=5)

    model = model.fit(xTrain, yTrain['price'])

    print("For XGBoost:")

    print("Best parameters found by grid search:")

    print(model.best_params_)

    print("Best CV score:")

    print(model.best_score_)

    

    preds = model.predict(xTest)

    error = sqrt(mean_squared_error(yTest['price'],preds))

    trainScore = model.score(xTrain, yTrain['price'])

    testScore = model.score(xTest, yTest['price'])

    r2 = r2_score(yTest['price'], preds)

    

    print()

    print('trainScore of XGB: ', trainScore)

    print('testScore of XGB: ', testScore)

    print('RMSE of test: ', error)

    print('R-square of test: ', r2)

    print()



xgb_acc(xTrain_oh, yTrain_oh, xTest_oh, yTest_oh)

xgb_acc(train_nmf, yTrain_oh, test_nmf, yTest_oh)

xgb_acc(train_pca, yTrain_oh, test_pca, yTest_oh)