import numpy as np

import pandas as pd

import os

from numba import jit

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV

from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn import neighbors

from sklearn.metrics import mean_absolute_error

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor

from sklearn.preprocessing import StandardScaler

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

from sklearn.kernel_ridge import KernelRidge

from sklearn.neighbors import NearestNeighbors

import librosa, librosa.display

import builtins

from sklearn.ensemble import RandomForestRegressor

import eli5

import shap

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

from sklearn import metrics

from IPython.display import HTML

import json

import altair as alt

from collections import Counter
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head()
plt.hist(train[train.columns[10]]);

plt.title(f'Distribution of {train.columns[10]}');
plt.hist(train.mean(1));

plt.title('Distribution of mean values of train columns');
plt.hist(train.std(1));

plt.title('Distribution of standard deviations of train columns');
train['target'].value_counts()
X = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id'], axis=1)

y = train['target']

n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
@jit

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict

    
params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 5,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 0.8

         }

result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb',

                                                                                  eval_metric='auc', plot_feature_importance=True, verbose=50, n_estimators=200)
sub = pd.read_csv("../input/sample_submission.csv")

sub['target'] = result_dict_lgb['prediction'][:, 1]

sub.to_csv("submission.csv", index=False)

sub.head()
train['wheezy-copper-turtle-magic'].nunique(), test['wheezy-copper-turtle-magic'].nunique()
sorted(train['wheezy-copper-turtle-magic'].unique()) == sorted(test['wheezy-copper-turtle-magic'].unique())
plt.hist(train.loc[train['target'] == 0, 'wheezy-copper-turtle-magic'], color='r', bins=512, label='0');

plt.hist(train.loc[train['target'] == 1, 'wheezy-copper-turtle-magic'], color='g', bins=512, label='1');

plt.hist(test['wheezy-copper-turtle-magic'], color='b', bins=512, label='1');

plt.title('Distribution of wheezy-copper-turtle-magic');

plt.legend()
col_part_names = [col.split('-') for col in train.columns]

col_part_names = [i for j in col_part_names for i in j]

Counter(col_part_names).most_common(10)
some_cols = [i[0] for i in Counter(col_part_names).most_common() if i[1] > 4]
test['target'] = -1

len_train = train.shape[0]

scaler = StandardScaler()

all_data = pd.concat([train, test], axis=0, sort=False, ignore_index=True).reset_index(drop=True)

for c in some_cols:

    more_such_cols = [col for col in all_data.columns if c in col]

    all_data[f'{c}_mean'] = all_data[more_such_cols].mean(1)

    all_data[f'{c}_min'] = all_data[more_such_cols].min(1)

    all_data[f'{c}_max'] = all_data[more_such_cols].max(1)

    all_data[f'{c}_std'] = all_data[more_such_cols].std(1)

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

all_data[cols] = scaler.fit_transform(all_data[cols])

#frequencies

all_data['wheezy-copper-turtle-magic_count'] = all_data.groupby(['wheezy-copper-turtle-magic'])['id'].transform('count')
train = all_data[:len_train].reset_index(drop=True)

test = all_data[len_train:].reset_index(drop=True)
train = pd.concat([train, pd.get_dummies(train['wheezy-copper-turtle-magic'], prefix='wctm')], axis=1)

test = pd.concat([test, pd.get_dummies(test['wheezy-copper-turtle-magic'], prefix='wctm')], axis=1)
X = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id', 'target'], axis=1)

y = train['target']

n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
params = {'num_leaves': 1024,

          'min_child_samples': 10,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 5,

          "subsample": 1.0,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 1.0,

          'min_sum_hessian_in_leaf': 10,

          'num_threads': -1

         }





result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb',

                                                                                  eval_metric='auc', plot_feature_importance=True, verbose=500, n_estimators=10000)
sub = pd.read_csv("../input/sample_submission.csv")

sub['target'] = result_dict_lgb['prediction'][:, 1]

sub.to_csv("submission_1.csv", index=False)

sub.head()