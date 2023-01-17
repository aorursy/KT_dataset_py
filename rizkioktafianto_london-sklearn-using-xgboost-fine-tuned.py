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
train = "../input/data-science-london-scikit-learn/train.csv"
test = "../input/data-science-london-scikit-learn/test.csv"
labels = "../input/data-science-london-scikit-learn/trainLabels.csv"

x_train = pd.read_csv(train, header=None)
x_test = pd.read_csv(test, header=None)
y_train = pd.read_csv(labels, header=None)
from sklearn.model_selection import train_test_split

xTrain, xVal, yTrain, yVal = train_test_split(x_train, y_train, random_state=123, test_size=0.2)
x_test
# standarization

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
xTrain_norm = ss.fit_transform(xTrain)
xTest_norm = ss.fit_transform(x_test)
xVal_norm = ss.fit_transform(xVal)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

XGB = XGBClassifier()

#benchmark
XGB.fit(xTrain_norm, yTrain.values.ravel())
prediction = XGB.predict(xVal_norm)
print("XGB Benchmark Accuracy: ", accuracy_score(yVal, prediction))
print(XGB.get_xgb_params())
# Hyperparameter Tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV
import xgboost as xgb

def modelfit(alg, xTrain, yTrain, xVal, yVal, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(xTrain, label=yTrain.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(xTrain, yTrain,eval_metric='auc')
        
    #Predict training set:
    val_predictions = alg.predict(xVal)
    val_predprob = alg.predict_proba(xVal)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(yVal, val_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(yVal, val_predprob))
# first iteration change in max_depth, min_child_weight, gamma, subsample, colsample_bytree, scale_pos_weight

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, xTrain_norm, yTrain, xVal_norm, yVal)
# iteration to max_depth and min_child_weigth

param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,10,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(xTrain_norm, yTrain)
gsearch1.best_params_, gsearch1.best_score_
# tune gamma
# fixed parameter max_depth and min_child_weight


param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(xTrain_norm,yTrain)
gsearch2.best_params_, gsearch2.best_score_
# Benchmark Again

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, xTrain_norm, yTrain, xVal_norm, yVal)
# tune subsample and colsample_bytree

param_test3 = {
 'subsample':[i/10.0 for i in range(5,10)],
 'colsample_bytree':[i/10.0 for i in range(5,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(xTrain_norm,yTrain)
gsearch3.best_params_, gsearch3.best_score_
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
 min_child_weight=1, gamma=0.0, subsample=0.9, colsample_bytree=0.9,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(xTrain_norm,yTrain)
gsearch4.best_params_, gsearch4.best_score_
param_test5 = {
 'reg_alpha':[1e-6, 5e-6, 7.5e-6, 1e-5, 5e-5]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
 min_child_weight=1, gamma=0.0, subsample=0.9, colsample_bytree=0.9,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(xTrain_norm,yTrain)
gsearch5.best_params_, gsearch5.best_score_
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.9,
 reg_alpha=0,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, xTrain_norm, yTrain, xVal_norm, yVal)
xgb3 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.9,
 reg_alpha=0,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, xTrain_norm, yTrain, xVal_norm, yVal)
predictions = xgb3.predict(xTest_norm)
predictions = pd.DataFrame({"Id":x_test.index+1,
                            "Solution":predictions})
predictions.to_csv("predictions_london.csv", index=False)

predictions
# Bagging Classifier

