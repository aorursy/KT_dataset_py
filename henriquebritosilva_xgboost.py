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

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

import sys

!{sys.executable} -m pip install xgboost

import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

%matplotlib inline

import matplotlib.pyplot as plt



X = pd.read_csv("/kaggle/input/_visagio-hackathon_/database_fires.csv")

#database.head()

#Y = pd.read_csv("/kaggle/input/_visagio-hackathon_/respostas.csv")

#resposta.head()

#print(len(database))

#print("X", X.shape)

#print("y", y.shape)

Y = X.fires

colunas = ["estacao","precipitacao","temp_max","temp_min","insolacao","evaporacao_piche","temp_comp_med","umidade_rel_med","vel_vento_med","altitude"]

X = X.loc[:][colunas]

#X.head()
seed = 10

X_training, X_test, y_training, y_test = train_test_split(X, y, random_state=seed, test_size=0.25, stratify=y)

print("Test set X", X_test.shape)

print("Test set y", y_test.shape)

#X_training.head()
X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, random_state=seed,test_size=0.33, stratify=y_training)

print("Train set X", X_train.shape)

print("Train set y", y_train.shape)

print("Validation set X", X_val.shape)

print("Validation set y", y_val.shape)
# XGBoost with Cross Validation

# import function for grid search from sklearn

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

# define the possible values of each parameters to be tested

params = {'learning_rate': [0.1, 0.2, 0.3],'alpha': [5, 10, 15],'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 1.0],'max_depth': [3, 4, 5]}

# create model object with XGBClassifier

xgb_model_cv_gs = xgb.XGBClassifier(objective="binary:logistic", random_state=seed,

eval_metric="auc", n_estimators=10)

# create kfold object with StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# create grid search object with GridSearchCV

grid_search = GridSearchCV(xgb_model_cv_gs, param_grid=params, scoring='roc_auc',cv=skf.split(X_training, y_training))

# train the model with grid search

grid_search.fit(X_training, y_training)

# print best hyperparameters combination

print('\n Best hyperparameters:')

print(grid_search.best_params_)

# get cv_results

cv_results = pd.DataFrame(grid_search.cv_results_)

# print average accuracy on validation sets

print("Average accuracy on validation set: {:.3f} +/- {:.3f}".format(cv_results[cv_results.rank_test_score == 1].mean_test_score.values[0],cv_results[cv_results.rank_test_score == 1].std_test_score.values[0]))

# set the best option for the parameters

xgb_model_cv_gs.set_params(learning_rate = grid_search.best_params_['learning_rate'],

alpha = grid_search.best_params_['alpha'],

colsample_bytree = grid_search.best_params_['colsample_bytree'],

max_depth = grid_search.best_params_['max_depth'])

# train a model using the best parameters

xgb_model_cv_gs.fit(X_training, y_training)

# plot variable importance

xgb.plot_importance(xgb_model_cv_gs)

plt.show()

# XGBoost with Cross Validation

# import function for grid search from sklearn

from sklearn.model_selection import GridSearchCV

# define the possible values of each parameters to be tested

params = {'learning_rate': [0.1, 0.2, 0.3],

'alpha': [5, 10, 15],

'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 1.0],

'max_depth': [3, 4, 5]}

# create model object with XGBClassifier

xgb_model_cv_gs = xgb.XGBClassifier(objective="binary:logistic", random_state=seed,

eval_metric="auc", n_estimators=10)

# create kfold object with StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# create grid search object with GridSearchCV

grid_search = GridSearchCV(xgb_model_cv_gs, param_grid=params, scoring='roc_auc',

cv=skf.split(X_training, y_training))

# train the model with grid search

grid_search.fit(X_training, y_training)

# print best hyperparameters combination

print('\n Best hyperparameters:')

print(grid_search.best_params_)

# get cv_results

cv_results = pd.DataFrame(grid_search.cv_results_)

# print average accuracy on validation sets

print("Average accuracy on validation set: {:.3f} +/- {:.3f}".format(cv_results[cv_results.rank_test_score == 1].mean_test_score.values[0],cv_results[cv_results.rank_test_score == 1].std_test_score.values[0]))

# set the best option for the parameters

xgb_model_cv_gs.set_params(learning_rate = grid_search.best_params_['learning_rate'],

alpha = grid_search.best_params_['alpha'],

colsample_bytree = grid_search.best_params_['colsample_bytree'],

max_depth = grid_search.best_params_['max_depth'])

# train a model using the best parameters

xgb_model_cv_gs.fit(X_training, y_training)

# plot variable importance

xgb.plot_importance(xgb_model_cv_gs)

plt.show()
