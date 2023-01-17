#Loading libraries

import pandas as pd

import numpy as np

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

from sklearn import model_selection, preprocessing, metrics

import matplotlib.pyplot as plt

import os
#inputting parameters

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/train.csv")
#train test split

X_train,X_test,y_train,y_test = train_test_split(train.drop(["target","ID_code"],axis=1),train["target"],test_size=0.3,random_state=14)
#grid of parameters

gridParams = {

    'learning_rate': [0.05],

    'num_leaves': [90,200],

    'boosting_type' : ['gbdt'],

    'objective' : ['binary'],

    'max_depth' : [5,6,7,8],

    'random_state' : [501], 

    'colsample_bytree' : [0.5,0.7],

    'subsample' : [0.5,0.7],

    'min_split_gain' : [0.01],

    'min_data_in_leaf':[10],

    'metric':['auc']

    }
#modelling

clf = lgb.LGBMRegressor()

grid = RandomizedSearchCV(clf,gridParams,verbose=1,cv=10,n_jobs = -1,n_iter=10)

grid.fit(X_train,y_train)
#best parameters

grid.best_params_
#Prediction

y_pred = grid.predict(X_test)
#auc calculation

metrics.roc_auc_score(y_test,y_pred)
#Feature importance for top 50 predictors

predictors = [x for x in X_train.columns]

feat_imp = pd.Series(grid.best_estimator_.feature_importances_, predictors).sort_values(ascending=False)

feat_imp = feat_imp[0:50]

plt.rcParams['figure.figsize'] = 20, 5

feat_imp.plot(kind='bar', title='Feature Importance')

plt.ylabel('Feature Importance Score')
#submission

predictions = grid.predict(test.drop(["ID_code","target"],axis=1))

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)