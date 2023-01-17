import gc

import os

import time

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from catboost import CatBoostRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

from typing import TypeVar, List, Dict, Tuple

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')



# Refs: https://www.kaggle.com/byfone/basic-feature-feat-catboost

# Refs: https://www.kaggle.com/kernels/scriptcontent/13873316/download
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/LANL/"

else:

    PATH="../input/"

os.listdir(PATH)



train_X = pd.read_csv(os.path.join(PATH,'../input/all-features/train_X.csv'))

train_y = pd.read_csv(os.path.join(PATH,'../input/all-features/train_y.csv'))

train_X.shape, train_y.shape
submission = pd.read_csv(os.path.join(PATH,'LANL-Earthquake-Prediction/sample_submission.csv'), index_col='seg_id')

test_X = pd.read_csv(os.path.join(PATH,'../input/all-features/test_X.csv'))

submission.shape, test_X.shape
"""Use cross-validation (e.g., KFold)"""

def fold_generator(x, y, groups=None, num_folds=10, shuffle=True, seed=2019):

    folds = KFold(num_folds, shuffle=shuffle, random_state=seed)

    for train_index, test_index in folds.split(x, y, groups):

        yield train_index, test_index
"""Search the hyperparameters and return the estimator with the best parameters"""

def search_cv(x, y, model, grid, max_iter=None, num_folds=10, shuffle=True):

    t0 = time.time()

    

    cv = fold_generator(x, y, num_folds=num_folds)

    if max_iter is None:

        # Exhaustive search over specified parameter values for an estimator (model)

        search = GridSearchCV(model, grid, cv=cv,

                              scoring='neg_mean_absolute_error')

    else:

        # Randomized search on hyper parameters with 

        # The number of parameter settings that are tried is given by n_iter (not all parameter values are tried out)

        search = RandomizedSearchCV(model, grid, n_iter=max_iter, cv=cv,

                                    scoring='neg_mean_absolute_error')

    search.fit(x, y, silent=True)

    

    t0 = time.time() - t0

    print("Best CV score: {:.4f}, time: {:.1f}s".format(-search.best_score_, t0))

    print(search.best_params_)

    return search.best_estimator_
"""Train, make predictions & plot results"""

def make_predictions(x, y, model, num_folds=10, shuffle=True, test=None, plot=True):

    if test is not None:

        sub_prediction = np.zeros(test.shape[0])

        

    oof_prediction = np.zeros(x.shape[0])

    # use cross-validation (10-fold cross-validation)

    for tr_idx, val_idx in fold_generator(x, y, num_folds=num_folds):

        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])

        oof_prediction[val_idx] = model.predict(x.iloc[val_idx])



        if test is not None:

            sub_prediction += model.predict(test) / num_folds

    

    if plot:

        plot_predictions(y, oof_prediction)

    if test is None:

        return oof_prediction

    else:

        return oof_prediction, sub_prediction
def plot_predictions(y, oof_predictions):

    """Plot out-of-fold predictions vs actual values."""

    fig, axis = plt.subplots(1, 2, figsize=(14, 6))

    ax1, ax2 = axis

    ax1.set_xlabel('actual')

    ax1.set_ylabel('predicted')

    ax1.set_ylim([-5, 20])

    ax2.set_xlabel('train index')

    ax2.set_ylabel('time to failure')

    ax2.set_ylim([-2, 18])

    ax1.scatter(y, oof_predictions, color='brown')

    ax1.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)], color='blue')

    ax2.plot(y, color='blue', label='y_train')

    ax2.plot(oof_predictions, color='orange')
scaler = StandardScaler()

scaler.fit(train_X)

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

scaled_test_X.shape
grid = {

    'depth': [4, 6, 8],

    'random_seed' : [400, 100, 200],

    'learning_rate' : [0.01, 0.03, 0.1]

}

cat_model = CatBoostRegressor(loss_function='MAE', task_type = "GPU")

cat_model = search_cv(scaled_train_X, train_y, cat_model, grid, max_iter=3)
# Feature Importance

fea_imp = pd.DataFrame({'imp': cat_model.feature_importances_, 'col': scaled_train_X.columns})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]

_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

plt.savefig('catboost_feature_importance.png')  
oof = make_predictions(scaled_train_X, train_y, cat_model)
oof, sub = make_predictions(scaled_train_X, train_y, cat_model,

                                  test=scaled_test_X, plot=False)
submission.time_to_failure = sub

submission.to_csv('submission.csv',index=True)