# importing general purpose libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import math

import random

import warnings

from sklearn import datasets



# importing model selection and evaluation libraries



# train-test-validation dataset creation

from sklearn.model_selection import train_test_split



# data normalization

from sklearn.preprocessing import MinMaxScaler, StandardScaler



# Pipeline

from sklearn.pipeline import Pipeline



# feature selection

from mlxtend.feature_selection import SequentialFeatureSelector

from mlxtend.plotting import plot_sequential_feature_selection



# hyperparameter tuning

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# crossvalidation

from sklearn.model_selection import cross_val_score, KFold



# accuracy testing

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error



# Importing models



# linear models

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.linear_model import BayesianRidge



# non-parametric models

from sklearn.neighbors import KNeighborsRegressor



# Decision tree

from sklearn.tree import DecisionTreeRegressor



# Support vectr machine

from sklearn.svm import SVR



# ensemble models



# bagging

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor



# tree based boosting

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor



# stacking

from mlxtend.regressor import StackingRegressor

def create_pipeline(norm, model):

    if norm == 1:

        scale = StandardScaler()

        pipe = Pipeline([('norm', scale), ('reg', model)])

    elif norm == 2:

        scale = MinMaxScaler()

        pipe = Pipeline([('norm', scale), ('reg', model)])

    else:

        pipe = Pipeline([('reg', model)])

    return pipe
def select_features(model, X_train, Y_train, selection,

                    score_criteria, see_details, norm=0):

    pipe = create_pipeline(norm, model)

    sfs = SequentialFeatureSelector(pipe,

                                    forward=selection,

                                    k_features='best',

                                    scoring=score_criteria,

                                    verbose=see_details)

    sfs = sfs.fit(X_train, Y_train)

    return list(sfs.k_feature_idx_)
def run_model(model, param_grid, X_train, Y_train,

              X, Y, score_criteria, folds,

              see_details, norm=0):

    pipe = create_pipeline(norm, model)

    model_grid = GridSearchCV(pipe,

                              param_grid,

                              cv=folds,

                              scoring=score_criteria,

                              verbose=see_details)

    model_grid.fit(X_train, Y_train)



    return model_grid.best_estimator_
def get_model_eval(model, X_train, Y_train, X_test, Y_test):

    return pd.Series([model, mean_squared_error(Y_train, model.predict(X_train)),

                      mean_squared_error(Y_test, model.predict(X_test)),

                      (abs(model.predict(X_train) - Y_train) / Y_train).mean(),

                      (abs(model.predict(X_test) - Y_test) / Y_test).mean()])
PARAM_DICT = {

              LinearRegression: {'reg__copy_X': [True, False],

                                 'reg__fit_intercept': [True, False],

                                 'reg__n_jobs': [10, 20]},

              Ridge: {'reg__alpha': [0.1, 1, 100],

                      'reg__copy_X': [True, False],

                      'reg__fit_intercept': [True, False],

                      'reg__tol': [0.1, 1],

                      'reg__solver': ['auto', 'svd', 'cholesky', 'lsqr',

                                      'sparse_cg', 'sag', 'saga']},

              Lasso: {'reg__alpha': [0.1, 1, 100],

                      'reg__copy_X': [True, False],

                      'reg__fit_intercept': [True, False],

                      'reg__tol': [0.1, 1]},



              KNeighborsRegressor: {'reg__n_neighbors': [5, 30, 100]},

              BayesianRidge: {'reg__alpha_1': [10**-6, 10**-3],

                              'reg__alpha_2': [10**-6, 10**-3],

                              'reg__copy_X': [True, False],

                              'reg__fit_intercept': [True, False],

                              'reg__lambda_1': [10**-6, 10**-3],

                              'reg__lambda_2': [10**-6, 10**-3],

                              'reg__n_iter': [300, 500, 1000],

                              'reg__tol': [0.001, 0.01, 0.1]},



              DecisionTreeRegressor: {'reg__max_depth': [5, 10, 20],

                                      'reg__max_features': [0.3, 0.7, 1.0],

                                      'reg__max_leaf_nodes': [10, 50, 100],

                                      'reg__splitter': ['best', 'random']},



              BaggingRegressor: {

                                 'reg__bootstrap': [True, False],

                                 'reg__bootstrap_features': [True, False],

                                 'reg__max_features': [0.3, 0.7, 1.0],

                                 'reg__max_samples': [0.3, 0.7, 1.0],

                                 'reg__n_estimators': [10, 50, 100]},

              RandomForestRegressor: {'reg__bootstrap': [True, False],

                                      'reg__max_depth': [5, 10, 20],

                                      'reg__max_features': [0.3, 0.7, 1.0],

                                      'reg__max_leaf_nodes': [10, 50, 100],

                                      'reg__min_impurity_decrease': [0, 0.1, 0.2],

                                      'reg__n_estimators': [10, 50, 100]},



              SVR: {'reg__C': [10**-3, 1, 1000],

                    'reg__kernel': ['linear', 'poly', 'rbf'],

                    'reg__shrinking': [True, False]},



              GradientBoostingRegressor: {'reg__learning_rate': [0.1, 0.2, 0.5],

                                          'reg__loss': ['ls', 'lad', 'huber', 'quantile'],

                                          'reg__max_depth': [10, 20, 50],

                                          'reg__max_features': [0.5, 0.8, 1.0],

                                          'reg__max_leaf_nodes': [10, 50, 100],

                                          'reg__min_impurity_decrease': [0, 0.1, 0.2],

                                          'reg__min_samples_leaf': [5, 10, 20],

                                          'reg__min_samples_split': [5, 10, 20],

                                          'reg__n_estimators': [10, 50, 100]},

              XGBRegressor: {'reg__booster': ['gbtree', 'gblinear', 'dart'],

                             'reg__learning_rate': [0.2, 0.5, 0.8],

                             'reg__max_depth': [5, 10, 20],

                             'reg__n_estimators': [10, 50, 100],

                             'reg__reg_alpha': [0.1, 1, 10],

                             'reg__reg_lambda': [0.1, 1, 10],

                             'reg__subsample': [0.3, 0.5, 0.8]},



              }
# --------------------------------------------------------------------------

# USER CONTROL PANEL, CHANGE THE VARIABLES, MODEL FORMS ETC. HERE



# Read data here, define X (features) and Y (Target variable)

data = datasets.load_diabetes()

X = pd.DataFrame(data['data'])

X.columns = data['feature_names']

Y = data['target']



# Specify size of test data (%)

size = 0.3



# Set random seed for sampling consistency

random.seed(100)



# Set type of normalization you want to perform

# 0 - No Normalization, 1 - Min-max scaling, 2 - Zscore scaling

norm = 0



# Mention all model forms you want to run - Model Objects

to_run = [LinearRegression,

          Ridge,

          Lasso,

          KNeighborsRegressor,

          DecisionTreeRegressor,

          BaggingRegressor,

          SVR]



# Specify number of crossvalidation folds

folds = 3



# Specify model selection criteria

# Possible values are:

# ‘explained_variance’

# ‘neg_mean_absolute_error’

# ‘neg_mean_squared_error’

# ‘neg_mean_squared_log_error’

# ‘neg_median_absolute_error’

# ‘r2’

score_criteria = 'neg_mean_absolute_error'



# Specify details of terminal output you'd like to see

# 0 - No output, 1 - All details, 2 - Progress bar

# Outputs might vary based on individual functions

see_details = 1



# --------------------------------------------------------------------------
# Model execution part, resuts will be stored in the dataframe 'results'

# Best model can be selected based on these criteria



results = pd.DataFrame(columns=['ModelForm', 'TrainRMSE', 'TestRMSE',

                                'TrainMAPE', 'TestMAPE'])



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size)



for model in to_run:

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        best_feat = select_features(model(), X_train, Y_train, True,

                                    score_criteria, see_details, norm)

        model = run_model(model(), PARAM_DICT[model],

                          X_train.iloc[:, best_feat],

                          Y_train,

                          X.iloc[:, best_feat], Y,

                          score_criteria, folds, see_details, norm)

        stats = get_model_eval(model, X_train.iloc[:, best_feat], Y_train,

                               X_test.iloc[:, best_feat], Y_test)

        stats.index = results.columns

        results = results.append(stats, ignore_index=True)



print(results)
