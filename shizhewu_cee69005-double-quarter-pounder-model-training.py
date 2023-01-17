import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import warnings

warnings.filterwarnings('ignore')

import lightgbm as lgb

import gc

import holidays

from sklearn import model_selection,linear_model,neighbors

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/cee69005-double-quarter-pounder-preprocessing/final_data_table.csv',index_col=[0])

data.drop("site_id", axis = 1, inplace = True)

data.drop("building_id", axis = 1, inplace = True)
data
X_LGBM = data

y_LGBM = np.log1p(X_LGBM.meter_reading)

X_half_1 = X_LGBM[:int(X_LGBM.shape[0] / 2)]

X_half_2 = X_LGBM[int(X_LGBM.shape[0] / 2):]

y_half_1 = y_LGBM[:int(X_LGBM.shape[0] / 2)]

y_half_2 = y_LGBM[int(X_LGBM.shape[0] / 2):]

categorical_features = [ "meter"]

d_half_1 = lgb.Dataset(X_half_1, label = y_half_1, categorical_feature = categorical_features, free_raw_data = False)

d_half_2 = lgb.Dataset(X_half_2, label = y_half_2, categorical_feature = categorical_features, free_raw_data = False)

watchlist_1 = [d_half_1, d_half_2]

watchlist_2 = [d_half_2, d_half_1]

params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 40,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse"

}

print("Building model with first half and validating on second half:")

model_half_1 = lgb.train(params, train_set = d_half_1, num_boost_round = 1000, valid_sets = watchlist_1, verbose_eval = 200, early_stopping_rounds = 200)

print("Building model with second half and validating on first half:")

model_half_2 = lgb.train(params, train_set = d_half_2, num_boost_round = 1000, valid_sets = watchlist_2, verbose_eval = 200, early_stopping_rounds = 200)
X_ridge = data

y_ridge = np.log1p(X_ridge.meter_reading)

n_folds = 5 

n_alphas = 20

kfoldcv = model_selection.KFold(n_splits = n_folds,shuffle = True,random_state = 42)

alphas = np.logspace(-3, 2, n_alphas)

ridge = linear_model.Ridge() 

param_grid = {'alpha':alphas} 

gscv = model_selection.GridSearchCV(ridge,param_grid,scoring = 'neg_root_mean_squared_error',cv = kfoldcv,iid = False,return_train_score = False)

gscv.fit(X_ridge,y_ridge)

print(gscv.best_params_)

print(-gscv.best_score_)
X_knn = data

y_knn = np.log1p(X_knn.meter_reading)

n_folds = 5 

n_neighbors_set = np.arange(1,11)

kfoldcv = model_selection.KFold(n_splits = n_folds,shuffle = True,random_state = 42)

neig = neighbors.KNeighborsRegressor()

param_grid = {'n_neighbors':n_neighbors_set} 

gscv = model_selection.GridSearchCV(neig,param_grid,scoring = 'neg_root_mean_squared_error',cv = kfoldcv,iid = False,return_train_score = False)

gscv.fit(X_knn,y_knn)

print(gscv.best_params_)

print(-gscv.best_score_)