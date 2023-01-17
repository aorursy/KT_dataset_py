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
%%time

import matplotlib.pyplot as plt



from tqdm import tqdm_notebook, tqdm_gui



from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import LinearRegression, LassoCV

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

pd.options.display.precision = 15



%matplotlib inline

import lightgbm as lgb

from xgboost.sklearn import XGBRegressor

import xgboost as xgb

import time

import datetime

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats



import tsfresh

from tsfresh import extract_features

from tsfresh import select_features

from tsfresh.utilities.dataframe_functions import impute



print('MODULES IMPORTED')
X_train_scaled = pd.read_csv('../input/train_features.csv')

y_train = pd.read_csv('../input/y_train.csv')

X_test_scaled = pd.read_csv('../input/test_features.csv')
print(X_train_scaled.head())

print(y_train.head())

print(X_test_scaled.head())
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_train, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

                    verbose=10000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'lasso':

            model = LassoCV(**params, n_jobs = -1)

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)

            print(f'Fold {fold_n}. MAE: {score:.4f}.')

            print('')

            y_pred = model.predict(X_test)

            

        if model_type == 'xgb':

            model = XGBRegressor(**params, n_estimators = 50000, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

                    verbose=10000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

    

        

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_absolute_error(y_valid, y_pred_valid))



        prediction += y_pred    

        

        

        if True:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            if model_type == 'lasso':

                fold_importance["importance"] = model.coef_

            else :

                fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

            

        



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if plot_feature_importance:

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title(model_type+' Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
params = {'num_leaves': 128,

          'min_data_in_leaf': 79,

          'objective': 'huber',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": 0.8126672064208567,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501

         }

oof_lgb, prediction_lgb, feature_importance_lgb = train_model(params=params, model_type='lgb', plot_feature_importance=True)
xgb_params = {'eta': 0.03,

              'max_depth': 9,

              'subsample': 0.85,

              'objective': 'reg:linear',

              'eval_metric': 'mae',

              'silent': True,

              'nthread': 4}

oof_xgb, prediction_xgb, feature_importance_xgb = train_model(X=X_train_scaled, X_test=X_test_scaled, params=xgb_params, model_type='xgb', plot_feature_importance=True)
params = {'cv':5}

oof_lasso, prediction_lasso, feature_importance_lasso = train_model(X=X_train_scaled, X_test=X_test_scaled, params=params, model_type='lasso', plot_feature_importance=True)
feature_importance_lgb.groupby(["feature"])["importance"].mean().to_csv("lgb_importance.csv")

feature_importance_xgb.groupby(["feature"])["importance"].mean().to_csv("xgb_importance.csv")

feature_importance_lasso.groupby(["feature"])["importance"].mean().to_csv("lasso_importance.csv")