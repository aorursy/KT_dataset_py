import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_breast_cancer,load_diabetes,load_wine

from sklearn.metrics import auc,accuracy_score,confusion_matrix,mean_squared_error,classification_report

from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split,KFold
def display_scores(scores):

    print("mean",np.mean(score),'\n',"std",np.std(scores))
diabetes = load_diabetes()
X = diabetes.data

y = diabetes.target
xgb_model = xgb.XGBRegressor(onjective = "reg:linear")

xgb_model.fit(X,y)

pred = xgb_model.predict(X)

mse = mean_squared_error(pred,y)

np.sqrt(mse)
cancer = load_breast_cancer()

X = cancer.data

y = cancer.target

xgb_model = xgb.XGBClassifier(objective = 'binary:logistic')

xgb_model.fit(X,y)

pred = xgb_model.predict(X)

print(classification_report(pred,y))
wine = load_wine()

X = wine.data

y = wine.target

xgb_model = xgb.XGBClassifier(objective = 'multi:softprob')

xgb_model.fit(X,y)

pred = xgb_model.predict(X)

print(classification_report(pred,y))
diabetes = load_diabetes()

X = diabetes.data

y = diabetes.target

kfold = KFold(n_splits = 5,shuffle = True)

scores = []

for train_index,test_index in kfold.split(X):

    X_train,X_test = X[train_index],X[test_index]

    y_train,y_test = y[train_index],y[test_index]

    xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror')

    xgb_model.fit(X_train,y_train)

    y_pred = xgb_model.predict(X_test)

    scores.append(mean_squared_error(y_test,y_pred))
scores
diabetes = load_diabetes()

X = diabetes.data

y = diabetes.target

scores = []

kfold = KFold(n_splits = 56,shuffle = True)

for train_index,test_index in kfold.split(X):

  X_train,X_test = X[train_index],X[test_index]

  y_train,y_test  = y[train_index],y[test_index]

  xgb_classifier = xgb.XGBRegressor(objective = 'reg:squarederror')

  xgb_classifier.fit(X_train,y_train)

  pred = xgb_classifier.predict(X_test)

  scores.append(mean_squared_error(pred,y_test))

xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror')

scores = cross_val_score(xgb_model,X,y,scoring = 'neg_mean_squared_error',cv = 6)

#A node is split only when the resulting split gives a positive reduction in the loss function. 

#Gamma specifies the minimum loss reduction required to make a split.
from random import randint

xgb_model = xgb.XGBRegressor()

params = {

    'colsample_bytree':[0.3,0.7],

    #fraction of columns to be randomly sampled for each tree

    'gamma':[0.1,0.05],

    # a node is split only when the resulting split gives a positive reduction in the loss functiong

    #gamma specifies the minimum loss reduction required to make the split

    'learning_rate':[0.03,0.3],

    'max_depth':[2,6],

    'n_estimators':[100,150],

    #fraction of samples randomly sampled for each tree

    'subsample':[0.6,0.4]    

}
search = RandomizedSearchCV(xgb_model,

                            param_distributions = params,

                            n_iter = 200,

                            cv = 3,

                            verbose = 29,

                            n_jobs = -1,

                            return_train_score = True

                           )
search.fit(X,y)
search.cv_results_
search.best_score_
search.best_params_
search.best_estimator_
cancer = load_breast_cancer()

X = cancer.data

y = cancer.target

xgb_model = xgb.XGBClassifier(objective = 'binary:logistic',

                              evaluation_metric = 'auc',

                             )

X_train,X_test,y_train,y_test = train_test_split(X,y)
xgb_model.fit(X_train,y_train,

              eval_set = [(X_test,y_test)],

              early_stopping_rounds = 5

             )

pred = xgb_model.predict(X_test)

print(classification_report(y_test,pred))
xgb_model = xgb.XGBClassifier(objective = 'binary:logistic',evaluation_metric = 'auc')

X_train,X_test,y_train,y_test = train_test_split(X,y)

xgb_model.fit(X_train,y_train,early_stopping_rounds = 5,eval_set = [(X_test,y_test)])
xgb_model.best_score
xgb_model.best_iteration
xgb_model.best_ntree_limit
xgb_model.get_params()
xgb_model.load_model()