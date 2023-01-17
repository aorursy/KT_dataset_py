import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product



import altair as alt

from altair.vega import v5

from IPython.display import HTML



# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

def prepare_altair():

    """

    Helper function to prepare altair for working.

    """



    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

    noext = "?noext"

    

    paths = {

        'vega': vega_url + noext,

        'vega-lib': vega_lib_url + noext,

        'vega-lite': vega_lite_url + noext,

        'vega-embed': vega_embed_url + noext

    }

    

    workaround = f"""    requirejs.config({{

        baseUrl: 'https://cdn.jsdelivr.net/npm/',

        paths: {paths}

    }});

    """

    

    return workaround

    



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

           



@add_autoincrement

def render(chart, id="vega-chart"):

    """

    Helper function to plot altair visualizations.

    """

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )

    



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    



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





def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()

    



def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3):

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

    columns = X.columns if columns is None else columns

    X_test = X_test[columns]

    splits = folds.split(X) if splits is None else splits

    n_splits = folds.n_splits if splits is None else n_folds

    

    # to set up scoring parameters

    metrics_dict = {'mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'sklearn_scoring_function': metrics.mean_absolute_error},

                    'group_mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'scoring_function': group_mean_log_mae},

                    'mse': {'lgb_metric_name': 'mse',

                        'catboost_metric_name': 'MSE',

                        'sklearn_scoring_function': metrics.mean_squared_error}

                    }



    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X))

    

    # averaged predictions on train data

    prediction = np.zeros(len(X_test))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(splits):

        if verbose:

            print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        if eval_metric != 'group_mae':

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        else:

            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict

    





def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1,groups=None):

    """

    A function to train a variety of classification models.

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

    columns = X.columns if columns is None else columns

    n_splits = folds.n_splits if splits is None else n_folds

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    if averaging == 'usual':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))

        

    elif averaging == 'rank':

        # out-of-fold predictions on train data

        oof = np.zeros((len(X), 1))



        # averaged predictions on train data

        prediction = np.zeros((len(X_test), 1))



    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,groups=groups)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

            

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

                                      loss_function=Logloss)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        if averaging == 'usual':

            

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

            

            prediction += y_pred.reshape(-1, 1)



        elif averaging == 'rank':

                                  

            oof[valid_index] = y_pred_valid.reshape(-1, 1)

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                                  

            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

            

        del X_train,X_valid

        gc.collect()



    prediction /= n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

            result_dict['top_columns'] = cols

        

    return result_dict



# setting up altair

workaround = prepare_altair()

HTML("".join((

    "<script>",

    workaround,

    "</script>",

)))
# From kernel https://www.kaggle.com/mpearmain/extended-timeseriessplitter

"""

This module provides a class to split time-series data for back-testing and evaluation.

The aim was to extend the current sklearn implementation and extend it's uses.



Might be useful for some ;)

"""



import logging

from typing import Optional



import numpy as np

from sklearn.model_selection._split import _BaseKFold

from sklearn.utils import indexable

from sklearn.utils.validation import _num_samples



LOGGER = logging.getLogger(__name__)





class TimeSeriesSplit_(_BaseKFold):  # pylint: disable=abstract-method

    """Time Series cross-validator



    Provides train/test indices to split time series data samples that are observed at fixed time intervals,

    in train/test sets. In each split, test indices must be higher than before, and thus shuffling in cross validator is

    inappropriate.



    This cross_validation object is a variation of :class:`TimeSeriesSplit` from the popular scikit-learn package.

    It extends its base functionality to allow for expanding windows, and rolling windows with configurable train and

    test sizes and delays between each. i.e. train on weeks 1-8, skip week 9, predict week 10-11.



    In this implementation we specifically force the test size to be equal across all splits.



    Expanding Window:



            Idx / Time  0..............................................n

            1           |  train  | delay |  test  |                   |

            2           |       train     | delay  |  test  |          |

            ...         |                                              |

            last        |            train            | delay |  test  |



    Rolling Windows:

            Idx / Time  0..............................................n

            1           | train   | delay |  test  |                   |

            2           | step |  train  | delay |  test  |            |

            ...         |                                              |

            last        | step | ... | step |  train  | delay |  test  |



    Parameters:

        n_splits : int, default=5

            Number of splits. Must be at least 4.



        train_size : int, optional

            Size for a single training set.



        test_size : int, optional, must be positive

            Size of a single testing set



        delay : int, default=0, must be positive

            Number of index shifts to make between train and test sets

            e.g,

            delay=0

                TRAIN: [0 1 2 3] TEST: [4]

            delay=1

                TRAIN: [0 1 2 3] TEST: [5]

            delay=2

                TRAIN: [0 1 2 3] TEST: [6]



        force_step_size : int, optional

            Ignore split logic and force the training data to shift by the step size forward for n_splits

            e.g

            TRAIN: [ 0  1  2  3] TEST: [4]

            TRAIN: [ 0  1  2  3  4] TEST: [5]

            TRAIN: [ 0  1  2  3  4  5] TEST: [6]

            TRAIN: [ 0  1  2  3  4  5  6] TEST: [7]



    Examples

    --------

    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

    >>> y = np.array([1, 2, 3, 4, 5, 6])

    >>> tscv = TimeSeriesSplit(n_splits=5)

    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE

    TimeSeriesSplit(train_size=None, n_splits=5)

    >>> for train_index, test_index in tscv.split(X):

    ...    print('TRAIN:', train_index, 'TEST:', test_index)

    ...    X_train, X_test = X[train_index], X[test_index]

    ...    y_train, y_test = y[train_index], y[test_index]

    TRAIN: [0] TEST: [1]

    TRAIN: [0 1] TEST: [2]

    TRAIN: [0 1 2] TEST: [3]

    TRAIN: [0 1 2 3] TEST: [4]

    TRAIN: [0 1 2 3 4] TEST: [5]

    """



    def __init__(self,

                 n_splits: Optional[int] = 5,

                 train_size: Optional[int] = None,

                 test_size: Optional[int] = None,

                 delay: int = 0,

                 force_step_size: Optional[int] = None):



        if n_splits and n_splits < 5:

            raise ValueError(f'Cannot have n_splits less than 5 (n_splits={n_splits})')

        super().__init__(n_splits, shuffle=False, random_state=None)



        self.train_size = train_size



        if test_size and test_size < 0:

            raise ValueError(f'Cannot have negative values of test_size (test_size={test_size})')

        self.test_size = test_size



        if delay < 0:

            raise ValueError(f'Cannot have negative values of delay (delay={delay})')

        self.delay = delay



        if force_step_size and force_step_size < 1:

            raise ValueError(f'Cannot have zero or negative values of force_step_size '

                             f'(force_step_size={force_step_size}).')



        self.force_step_size = force_step_size



    def split(self, X, y=None, groups=None):

        """Generate indices to split data into training and test set.



        Parameters:

            X : array-like, shape (n_samples, n_features)

                Training data, where n_samples is the number of samples  and n_features is the number of features.



            y : array-like, shape (n_samples,)

                Always ignored, exists for compatibility.



            groups : array-like, with shape (n_samples,), optional

                Always ignored, exists for compatibility.



        Yields:

            train : ndarray

                The training set indices for that split.



            test : ndarray

                The testing set indices for that split.

        """

        X, y, groups = indexable(X, y, groups)  # pylint: disable=unbalanced-tuple-unpacking

        n_samples = _num_samples(X)

        n_splits = self.n_splits

        n_folds = n_splits + 1

        delay = self.delay



        if n_folds > n_samples:

            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples: {n_samples}.')



        indices = np.arange(n_samples)

        split_size = n_samples // n_folds



        train_size = self.train_size or split_size * self.n_splits

        test_size = self.test_size or n_samples // n_folds

        full_test = test_size + delay



        if full_test + n_splits > n_samples:

            raise ValueError(f'test_size\\({test_size}\\) + delay\\({delay}\\) = {test_size + delay} + '

                             f'n_splits={n_splits} \n'

                             f' greater than the number of samples: {n_samples}. Cannot create fold logic.')



        # Generate logic for splits.

        # Overwrite fold test_starts ranges if force_step_size is specified.

        if self.force_step_size:

            step_size = self.force_step_size

            final_fold_start = n_samples - (train_size + full_test)

            range_start = (final_fold_start % step_size) + train_size



            test_starts = range(range_start, n_samples, step_size)



        else:

            if not self.train_size:

                step_size = split_size

                range_start = (split_size - full_test) + split_size + (n_samples % n_folds)

            else:

                step_size = (n_samples - (train_size + full_test)) // n_folds

                final_fold_start = n_samples - (train_size + full_test)

                range_start = (final_fold_start - (step_size * (n_splits - 1))) + train_size



            test_starts = range(range_start, n_samples, step_size)



        # Generate data splits.

        for test_start in test_starts:

            idx_start = test_start - train_size if self.train_size is not None else 0

            # Ensure we always return a test set of the same size

            if indices[test_start:test_start + full_test].size < full_test:

                continue

            yield (indices[idx_start:test_start],

                   indices[test_start + delay:test_start + full_test])
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage2(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc

# Any results you write to the current directory are saved as output.

import imblearn

from imblearn.under_sampling import RandomUnderSampler,TomekLinks

import datetime

import lightgbm as lgb

import sklearn as skl

from sklearn.model_selection import *

from sklearn.metrics import *

import matplotlib.pyplot as plt

from imblearn.over_sampling import *

from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.utils import resample
%%time

folder_path = '../input/ieee-fraud-detection/'

train_id = pd.read_csv(f'{folder_path}train_identity.csv')

train_tr = pd.read_csv(f'{folder_path}train_transaction.csv')

test_id = pd.read_csv(f'{folder_path}test_identity.csv')

test_tr = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

del train_id,test_id,train_tr,test_tr

gc.collect()
def make_time(df):

    START_DATE = '2017-11-30'

    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    #df['DT'] = df['TransactionDT'].apply(lambda x: (startdate  + datetime.timedelta(seconds = x)))

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    #df['year'] = df['TransactionDT'].dt.year

    df['month'] = df['TransactionDT'].dt.month

    #df['dow'] = df['TransactionDT'].dt.dayofweek

    #df['hour'] = df['TransactionDT'].dt.hour

    #df['day'] = df['TransactionDT'].dt.day

    #df['is_holiday'] = (df['TransactionDT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    #df = df.drop('DT',axis=1)

    return df

train = make_time(train)

test = make_time(test)

groups = train['month']
train = train.drop(['TransactionID','TransactionDT','month'],axis=1)

test = test.drop(['TransactionID','TransactionDT','month'],axis=1)
%%time

train = reduce_mem_usage2(train)

test = reduce_mem_usage2(test)
sub['isFraud']=0.

sub['isFraud'].head()
def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)
'''

not_fraud = train[train.isFraud==0]

fraud = train[train.isFraud==1]

not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled = downsampled.sort_index()

'''

downsampled = train

cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',

            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',

            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

for col in cat_cols:

    if col in downsampled.columns:

        le = LabelEncoder()

        le.fit(list(downsampled[col].astype(str).values) + list(test[col].astype(str).values))

        downsampled[col] = le.transform(list(downsampled[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values)) 

X = clean_inf_nan(downsampled)

X_test = clean_inf_nan(test)

n_fold = 5

folds = GroupKFold(n_fold)

SEED = 42

lgb_params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    #'n_jobs':-1,

                    'learning_rate':0.01,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.5,

                    'subsample_freq':1,

                    'subsample':0.7,

                    #'n_estimators':800,

                    'max_bin':255,

                    'verbose':-1,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                } 

y = X['isFraud']

X = X.drop(['isFraud'],axis=1)



result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=lgb_params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=100, n_estimators=800, averaging='usual', n_jobs=-1,groups=groups)
sub['isFraud'] = result_dict_lgb['prediction']

sub.to_csv('submission.csv', index=False)