import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15

from collections import defaultdict

import lightgbm as lgb

import xgboost as xgb

import catboost as cat

import time

from collections import Counter

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from bayes_opt import BayesianOptimization

import eli5

import shap

from IPython.display import HTML

import json

import altair as alt

from category_encoders.ordinal import OrdinalEncoder

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

from typing import List



import os

import time

import datetime

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

from typing import Any

from itertools import product

pd.set_option('max_rows', 500)

import re

from tqdm import tqdm

from joblib import Parallel, delayed



#add by NorwayPing

import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

@jit

def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e





def eval_qwk_lgb(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """



    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True





def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    1.07008134 1.68928498 2.25542093"""

   

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3 

    """y_pred[y_pred <= 1.09008134] = 0

    #score CV 0.6061

    y_pred[np.where(np.logical_and(y_pred > 1.09008134, y_pred <= 1.70928498))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.70928498, y_pred <= 2.22542093))] = 2

    y_pred[y_pred > 2.22542093] = 3"""

    

    #[1.07866106 1.72488106 2.19597466]

    #[1.08114242 1.76433952 2.21907389]

    

    #score CV 0.6067

    """y_pred[y_pred <= 1.08008134] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.08008134, y_pred <= 1.71528498))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.71528498, y_pred <= 2.21242093))] = 2

    y_pred[y_pred > 2.21242093] = 3"""



    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)



    return 'cappa', qwk(y_true, y_pred), True



#add by NorwayPing

class CatBoostWrapper_regr(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = cat.CatBoostRegressor()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        if params['objective'] == 'regression':

            eval_metric = eval_qwk_lgb_regr

        else:

            eval_metric = 'auc'



        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict(self, X_test):

        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)





class LGBWrapper_regr(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = lgb.LGBMRegressor()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        if params['objective'] == 'regression':

            eval_metric = eval_qwk_lgb_regr

        else:

            eval_metric = 'auc'



        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict(self, X_test):

        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)



    

def eval_qwk_xgb(y_pred, y_true):

    """

    Fast cappa eval function for xgb.

    """

    # print('y_true', y_true)

    # print('y_pred', y_pred)

    y_true = y_true.get_label()

    y_pred = y_pred.argmax(axis=1)

    return 'cappa', -qwk(y_true, y_pred)





class LGBWrapper(object):

    """

    A wrapper for lightgbm model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = lgb.LGBMClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        eval_names = ['train']

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))

            eval_names.append('valid')



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))

            eval_names.append('holdout')



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = 'auto'

        else:

            categorical_columns = 'auto'



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       categorical_feature=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if self.model.objective == 'binary':

            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]

        else:

            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)





class CatWrapper(object):

    """

    A wrapper for catboost model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = cat.CatBoostClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        self.model = self.model.set_params(**{k: v for k, v in params.items() if k != 'cat_cols'})



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))



        if 'cat_cols' in params.keys():

            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]

            if len(cat_cols) > 0:

                categorical_columns = params['cat_cols']

            else:

                categorical_columns = None

        else:

            categorical_columns = None

        

        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],

                       cat_features=categorical_columns)



        self.best_score_ = self.model.best_score_

        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if 'MultiClass' not in self.model.get_param('loss_function'):

            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)[:, 1]

        else:

            return self.model.predict_proba(X_test, ntree_end=self.model.best_iteration_)





class XGBWrapper(object):

    """

    A wrapper for xgboost model so that we will have a single api for various models.

    """



    def __init__(self):

        self.model = xgb.XGBClassifier()



    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):



        eval_set = [(X_train, y_train)]

        self.model = self.model.set_params(**params)



        if X_valid is not None:

            eval_set.append((X_valid, y_valid))



        if X_holdout is not None:

            eval_set.append((X_holdout, y_holdout))



        self.model.fit(X=X_train, y=y_train,

                       eval_set=eval_set, eval_metric=eval_qwk_xgb,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])



        scores = self.model.evals_result()

        self.best_score_ = {k: {m: m_v[-1] for m, m_v in v.items()} for k, v in scores.items()}

        self.best_score_ = {k: {m: n if m != 'cappa' else -n for m, n in v.items()} for k, v in self.best_score_.items()}



        self.feature_importances_ = self.model.feature_importances_



    def predict_proba(self, X_test):

        if self.model.objective == 'binary':

            return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)[:, 1]

        else:

            return self.model.predict_proba(X_test, ntree_limit=self.model.best_iteration)









class MainTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, convert_cyclical: bool = False, create_interactions: bool = False, n_interactions: int = 20):

        """

        Main transformer for the data. Can be used for processing on the whole data.



        :param convert_cyclical: convert cyclical features into continuous

        :param create_interactions: create interactions between features

        """



        self.convert_cyclical = convert_cyclical

        self.create_interactions = create_interactions

        self.feats_for_interaction = None

        self.n_interactions = n_interactions



    def fit(self, X, y=None):



        if self.create_interactions:

            self.feats_for_interaction = [col for col in X.columns if 'sum' in col

                                          or 'mean' in col or 'max' in col or 'std' in col

                                          or 'attempt' in col]

            self.feats_for_interaction1 = np.random.choice(self.feats_for_interaction, self.n_interactions)

            self.feats_for_interaction2 = np.random.choice(self.feats_for_interaction, self.n_interactions)



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

        if self.create_interactions:

            for col1 in self.feats_for_interaction1:

                for col2 in self.feats_for_interaction2:

                    data[f'{col1}_int_{col2}'] = data[col1] * data[col2]



        if self.convert_cyclical:

            data['timestampHour'] = np.sin(2 * np.pi * data['timestampHour'] / 23.0)

            data['timestampMonth'] = np.sin(2 * np.pi * data['timestampMonth'] / 23.0)

            data['timestampWeek'] = np.sin(2 * np.pi * data['timestampWeek'] / 23.0)

            data['timestampMinute'] = np.sin(2 * np.pi * data['timestampMinute'] / 23.0)



#         data['installation_session_count'] = data.groupby(['installation_id'])['Clip'].transform('count')

#         data['installation_duration_mean'] = data.groupby(['installation_id'])['duration_mean'].transform('mean')

#         data['installation_title_nunique'] = data.groupby(['installation_id'])['session_title'].transform('nunique')



#         data['sum_event_code_count'] = data[['2000', '3010', '3110', '4070', '4090', '4030', '4035', '4021', '4020', '4010', '2080', '2083', '2040', '2020', '2030', '3021', '3121', '2050', '3020', '3120', '2060', '2070', '4031', '4025', '5000', '5010', '2081', '2025', '4022', '2035', '4040', '4100', '2010', '4110', '4045', '4095', '4220', '2075', '4230', '4235', '4080', '4050']].sum(axis=1)



        # data['installation_event_code_count_mean'] = data.groupby(['installation_id'])['sum_event_code_count'].transform('mean')



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)





class FeatureTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, main_cat_features: list = None, num_cols: list = None):

        """



        :param main_cat_features:

        :param num_cols:

        """

        self.main_cat_features = main_cat_features

        self.num_cols = num_cols



    def fit(self, X, y=None):



#         self.num_cols = [col for col in X.columns if 'sum' in col or 'mean' in col or 'max' in col or 'std' in col

#                          or 'attempt' in col]

        



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

#         for col in self.num_cols:

#             data[f'{col}_to_mean'] = data[col] / data.groupby('installation_id')[col].transform('mean')

#             data[f'{col}_to_std'] = data[col] / data.groupby('installation_id')[col].transform('std')



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)
class RegressorModel(object):

    """

    A wrapper class for classification models.

    It can be used for training and prediction.

    Can plot feature importance and training progress (if relevant for model).



    """



    def __init__(self, columns: list = None, model_wrapper=None):

        """



        :param original_columns:

        :param model_wrapper:

        """

        self.columns = columns

        self.model_wrapper = model_wrapper

        self.result_dict = {}

        self.train_one_fold = False

        self.preprocesser = None



    def fit(self, X: pd.DataFrame, y,

            X_holdout: pd.DataFrame = None, y_holdout=None,

            folds=None,

            params: dict = None,

            eval_metric='rmse',

            cols_to_drop: list = None,

            preprocesser=None,

            transformers: dict = None,

            adversarial: bool = False,

            plot: bool = True):

        """

        Training the model.



        :param X: training data

        :param y: training target

        :param X_holdout: holdout data

        :param y_holdout: holdout target

        :param folds: folds to split the data. If not defined, then model will be trained on the whole X

        :param params: training parameters

        :param eval_metric: metric for validataion

        :param cols_to_drop: list of columns to drop (for example ID)

        :param preprocesser: preprocesser class

        :param transformers: transformer to use on folds

        :param adversarial

        :return:

        """



        if folds is None:

            folds = KFold(n_splits=3, random_state=42)

            self.train_one_fold = True



        self.columns = X.columns if self.columns is None else self.columns

        self.feature_importances = pd.DataFrame(columns=['feature', 'importance'])

        self.trained_transformers = {k: [] for k in transformers}

        self.transformers = transformers

        self.models = []

        self.folds_dict = {}

        self.eval_metric = eval_metric

        n_target = 1

        self.oof = np.zeros((len(X), n_target))

        self.n_target = n_target



        X = X[self.columns]

        if X_holdout is not None:

            X_holdout = X_holdout[self.columns]



        if preprocesser is not None:

            self.preprocesser = preprocesser

            self.preprocesser.fit(X, y)

            X = self.preprocesser.transform(X, y)

            self.columns = X.columns.tolist()

            if X_holdout is not None:

                X_holdout = self.preprocesser.transform(X_holdout)



        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, X['installation_id'])):



            if X_holdout is not None:

                X_hold = X_holdout.copy()

            else:

                X_hold = None

            self.folds_dict[fold_n] = {}

            if params['verbose']:

                print(f'Fold {fold_n + 1} started at {time.ctime()}')

            self.folds_dict[fold_n] = {}



            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            if self.train_one_fold:

                X_train = X[self.original_columns]

                y_train = y

                X_valid = None

                y_valid = None



            datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_hold, 'y_train': y_train}

            X_train, X_valid, X_hold = self.transform_(datasets, cols_to_drop)



            self.folds_dict[fold_n]['columns'] = X_train.columns.tolist()



            model = copy.deepcopy(self.model_wrapper)



            if adversarial:

                X_new1 = X_train.copy()

                if X_valid is not None:

                    X_new2 = X_valid.copy()

                elif X_holdout is not None:

                    X_new2 = X_holdout.copy()

                X_new = pd.concat([X_new1, X_new2], axis=0)

                y_new = np.hstack((np.zeros((X_new1.shape[0])), np.ones((X_new2.shape[0]))))

                X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_new)



            model.fit(X_train, y_train, X_valid, y_valid, X_hold, y_holdout, params=params)



            self.folds_dict[fold_n]['scores'] = model.best_score_

            if self.oof.shape[0] != len(X):

                self.oof = np.zeros((X.shape[0], self.oof.shape[1]))

            if not adversarial:

                self.oof[valid_index] = model.predict(X_valid).reshape(-1, n_target)



            fold_importance = pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)),

                                           columns=['feature', 'importance'])

            self.feature_importances = self.feature_importances.append(fold_importance)

            self.models.append(model)



        self.feature_importances['importance'] = self.feature_importances['importance'].astype(int)



        # if params['verbose']:

        self.calc_scores_()



        if plot:

            # print(classification_report(y, self.oof.argmax(1)))

            fig, ax = plt.subplots(figsize=(16, 18))

            plt.subplot(2, 2, 1)

            self.plot_feature_importance(top_n=40)

            plt.subplot(2, 2, 2)

            self.plot_metric()

            plt.subplot(2, 2, 3)

            plt.hist(y.values.reshape(-1, 1) - self.oof)

            plt.title('Distribution of errors')

            plt.subplot(2, 2, 4)

            plt.hist(self.oof)

            plt.title('Distribution of oof predictions');



    def transform_(self, datasets, cols_to_drop):

        for name, transformer in self.transformers.items():

            transformer.fit(datasets['X_train'], datasets['y_train'])

            datasets['X_train'] = transformer.transform(datasets['X_train'])

            if datasets['X_valid'] is not None:

                datasets['X_valid'] = transformer.transform(datasets['X_valid'])

            if datasets['X_holdout'] is not None:

                datasets['X_holdout'] = transformer.transform(datasets['X_holdout'])

            self.trained_transformers[name].append(transformer)

        if cols_to_drop is not None:

            cols_to_drop = [col for col in cols_to_drop if col in datasets['X_train'].columns]



            datasets['X_train'] = datasets['X_train'].drop(cols_to_drop, axis=1)

            if datasets['X_valid'] is not None:

                datasets['X_valid'] = datasets['X_valid'].drop(cols_to_drop, axis=1)

            if datasets['X_holdout'] is not None:

                datasets['X_holdout'] = datasets['X_holdout'].drop(cols_to_drop, axis=1)

        self.cols_to_drop = cols_to_drop



        return datasets['X_train'], datasets['X_valid'], datasets['X_holdout']



    def calc_scores_(self):

        print()

        datasets = [k for k, v in [v['scores'] for k, v in self.folds_dict.items()][0].items() if len(v) > 0]

        self.scores = {}

        for d in datasets:

            scores = [v['scores'][d][self.eval_metric] for k, v in self.folds_dict.items()]

            print(f"CV mean score on {d}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} std.")

            self.scores[d] = np.mean(scores)



    def predict(self, X_test, averaging: str = 'usual'):

        """

        Make prediction



        :param X_test:

        :param averaging: method of averaging

        :return:

        """

        full_prediction = np.zeros((X_test.shape[0], self.oof.shape[1]))

        if self.preprocesser is not None:

            X_test = self.preprocesser.transform(X_test)

        for i in range(len(self.models)):

            X_t = X_test.copy()

            for name, transformers in self.trained_transformers.items():

                X_t = transformers[i].transform(X_t)



            if self.cols_to_drop is not None:

                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]

                X_t = X_t.drop(cols_to_drop, axis=1)

            y_pred = self.models[i].predict(X_t[self.folds_dict[i]['columns']]).reshape(-1, full_prediction.shape[1])



            # if case transformation changes the number of the rows

            if full_prediction.shape[0] != len(y_pred):

                full_prediction = np.zeros((y_pred.shape[0], self.oof.shape[1]))



            if averaging == 'usual':

                full_prediction += y_pred

            elif averaging == 'rank':

                full_prediction += pd.Series(y_pred).rank().values



        return full_prediction / len(self.models)



    def plot_feature_importance(self, drop_null_importance: bool = True, top_n: int = 10):

        """

        Plot default feature importance.



        :param drop_null_importance: drop columns with null feature importance

        :param top_n: show top n columns

        :return:

        """



        top_feats = self.get_top_features(drop_null_importance, top_n)

        feature_importances = self.feature_importances.loc[self.feature_importances['feature'].isin(top_feats)]

        feature_importances['feature'] = feature_importances['feature'].astype(str)

        top_feats = [str(i) for i in top_feats]

        sns.barplot(data=feature_importances, x='importance', y='feature', orient='h', order=top_feats)

        plt.title('Feature importances')



    def get_top_features(self, drop_null_importance: bool = True, top_n: int = 10):

        """

        Get top features by importance.



        :param drop_null_importance:

        :param top_n:

        :return:

        """

        grouped_feats = self.feature_importances.groupby(['feature'])['importance'].mean()

        if drop_null_importance:

            grouped_feats = grouped_feats[grouped_feats != 0]

        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]



    def plot_metric(self):

        """

        Plot training progress.

        Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html



        :return:

        """

        full_evals_results = pd.DataFrame()

        for model in self.models:

            evals_result = pd.DataFrame()

            for k in model.model.evals_result_.keys():

                evals_result[k] = model.model.evals_result_[k][self.eval_metric]

            evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})

            full_evals_results = full_evals_results.append(evals_result)



        full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,

                                                                                            'variable': 'dataset'})

        sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')

        plt.title('Training progress')

class CategoricalTransformer(BaseEstimator, TransformerMixin):



    def __init__(self, cat_cols=None, drop_original: bool = False, encoder=OrdinalEncoder()):

        """

        Categorical transformer. This is a wrapper for categorical encoders.



        :param cat_cols:

        :param drop_original:

        :param encoder:

        """

        self.cat_cols = cat_cols

        self.drop_original = drop_original

        self.encoder = encoder

        self.default_encoder = OrdinalEncoder()



    def fit(self, X, y=None):



        if self.cat_cols is None:

            kinds = np.array([dt.kind for dt in X.dtypes])

            is_cat = kinds == 'O'

            self.cat_cols = list(X.columns[is_cat])

        self.encoder.set_params(cols=self.cat_cols)

        self.default_encoder.set_params(cols=self.cat_cols)



        self.encoder.fit(X[self.cat_cols], y)

        self.default_encoder.fit(X[self.cat_cols], y)



        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)

        new_cat_names = [f'{col}_encoded' for col in self.cat_cols]

        encoded_data = self.encoder.transform(data[self.cat_cols])

        if encoded_data.shape[1] == len(self.cat_cols):

            data[new_cat_names] = encoded_data

        else:

            pass



        if self.drop_original:

            data = data.drop(self.cat_cols, axis=1)

        else:

            data[self.cat_cols] = self.default_encoder.transform(data[self.cat_cols])



        return data



    def fit_transform(self, X, y=None, **fit_params):

        data = copy.deepcopy(X)

        self.fit(data)

        return self.transform(data)
def encode_title_2(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = sorted(list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique())))

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = sorted(list(set(train['title'].unique()).union(set(test['title'].unique()))))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = sorted(list(set(train['event_code'].unique()).union(set(test['event_code'].unique()))))

    list_of_event_id = sorted(list(set(train['event_id'].unique()).union(set(test['event_id'].unique()))))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = sorted(list(set(train['world'].unique()).union(set(test['world'].unique()))))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = sorted(list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index))))

    

    #add by NorwayPing

    game_titles = sorted(list(set(train[train['type'] == 'Game']['title'].value_counts().index).union(set(test[test['type'] == 'Game']['title'].value_counts().index))))



    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    #add by NorwayPing

    labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])) # get the mode

    labels_map_acc_mean = dict(train_labels.groupby('title')['accuracy'].mean()) # get the mode

    #add by NorwayPing 2019-12-29

    labels_map_acc_std = dict(train_labels.groupby('title')['accuracy'].std()) # get the mode

 



    

    win_code_game = dict(zip(activities_map.values(), (4020*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code_game[activities_map['Air Show']] = 4100

    win_code_game[activities_map['Pan Balance']] = 4100



    return train, test, train_labels, win_code, win_code_game, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, game_titles, list_of_event_id, all_title_event_code, labels_map, labels_map_acc_mean, labels_map_acc_std 

def get_train_and_test_2(train, test):

    compiled_train = []

    compiled_test = []

    #add by NorwayPing to reduce processing time for train data. Because of installation_id don't take any assesement

    train_labels_id=pd.DataFrame(train_labels['installation_id'].unique())

    train_labels_id.columns =  ['installation_id'] 

    train_pre = pd.merge(train, train_labels_id, on = ['installation_id'])



    

    for i, (ins_id, user_sample) in tqdm(enumerate(train_pre.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data_2(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data_2(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
def read_data_2():

    print('Reading train.csv file....')

    

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission
def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission
def get_data_2(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    

    

    #add by NorwayPing

    adjust_accumulated_accuracy = 0

    accuracy_weight_maps = {title: 1 for title in assess_titles}

    #accuracy_weight_maps['Cart Balancer (Assessment)'] = 1

    ##accuracy_weight_maps['Cauldron Filler (Assessment)'] = 1

    #accuracy_weight_maps['Mushroom Sorter (Assessment)'] = 1

    accuracy_weight_maps['Bird Measurer (Assessment)'] = 2

    accuracy_weight_maps['Chest Sorter (Assessment)'] = 3

    

    acu_accuracy_title = {'acum_' + title: -1 for title in assess_titles}

    

    counter_game = 0

    accumulated_accuracy_game = 0

    accumulated_correct_attempts_game = 0 

    accumulated_uncorrect_attempts_game = 0

    duration_all =0

    duration_game =[]

    

    #add by NorwayPing, not used yet variables

    dura_title = {'dura_' + title: -1 for title in assess_titles}

    dura_title_game = {'dura_' + title: -1 for title in game_titles}

     

      

    acum_count_title_game = {'acum_count_' + title: -1 for title in game_titles}

    

    acu_dura_title = {'acum_dura_' + title: -1 for title in assess_titles}

    

    #combined_last_accumulated_accuracy = {'acc_N1_0_0_0_0'}

       

    last_accuracy_title_game = {'acc_' + title: -1 for title in game_titles}

    acu_accuracy_title_game = {title: -1 for title in game_titles}

    acumulated_accuracy_title_game = {'acum_acc_' + title: -1 for title in game_titles}

    count_title_game = {title: 0 for title in game_titles}

    acum_accuracy_title_game = {title: 0 for title in game_titles}

    

   

   

    #last_accuracy_std_title = {'acc_std' + title: -1 for title in assess_titles}

    



    

    

    clip_session_count =0

    

    

    title_event_code_percent: Dict[str, int] = {'pec_' + t_eve: -1 for t_eve in all_title_event_code}

   

    

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    

    #add by NorwayPing

    event_code_log: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    title_log: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_log: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    

    #title duration

       

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session



        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

        

        

        

        # add by NorwayPing

        session_duration =(session.iloc[-1, 2] - session.iloc[0, 2] ).seconds

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            

           

            

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            

                   

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            

            '''

            features.update(event_code_count.copy())

            #delete by NorwayPing to reduce duplication

            #features.update(event_id_count.copy())

            features.update(title_count.copy())

            

            features.update(title_event_code_count.copy())

            '''

            #chang to log value by NorwayPing

            features.update(event_code_log.copy())

            #delete by NorwayPing to reduce duplication

            #features.update(event_id_count.copy())

            features.update(title_log.copy())

            

            features.update(title_event_code_log.copy())

            

            

            features.update(last_accuracy_title.copy())

            

            features.update(last_accuracy_title_game.copy())

           # features.update( last_accuracy_std_title.copy())

            

                        

            #with this feature 0.6054, max_depth =12

            features.update(title_event_code_percent.copy())

            

            

            features.update(acu_dura_title.copy())

            features.update(acumulated_accuracy_title_game.copy())

            features.update(acum_count_title_game.copy())

            

            

             #add by NorwayPing, 2019-12-30

               

            

            features['Month'] = session.iloc[-1]['timestamp'].month

            #df['Year_Month'] = df['Date'].map(lambda x: 100*x.year + x.month)

            features['Day'] = session.iloc[-1]['timestamp'].day

            features['weekday'] = session.iloc[-1]['timestamp'].weekday()

            features['hour'] = session.iloc[-1]['timestamp'].hour

            features['timeOfDay'] = (session.iloc[-1]['timestamp'].hour*60+session.iloc[-1]['timestamp'].minute)/1440

            

                     

            #features['event_to_mean'] =df['title_event_count']/df['title_count']

            

            features['last_accurancy_same_title'] = last_accuracy_title['acc_' + session_title_text]

           

            

            #add by NorwayPing, average accuracy of the title

            features['most_accurancy_same_title'] =labels_map[session_title]

            features['mean_accurancy_same_title'] = labels_map_acc_mean[session_title]

            features['std_accurancy_same_title'] = labels_map_acc_std[session_title]

            

            features['title_nunique'] = sum(value > 0 for value in title_count.values()) 

            #sum(value == 0 for value in D.values())

            

            #features['duration_all'] =duration_all

            

                      

            #add by NorwayPing, debug

            #get session_id for aggregated features

            features['session_id'] = session['game_session'].iloc[-1]

            #features['clip_session_count'] = clip_session_count

            

            

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            

            #add by NorwayPing

            # the time spent in the app so far

           # if duration_game == []:

           #     features['duration_game'] = 0

           # else:

           #     features['duration_game'] = np.mean(duration_game)

                

            # the accurace is the all time wins divided by the all time attempts

            

            '''features['accumulated_accuracy_to_mean_title1'] = accumulated_accuracy/counter if counter > 0 else 0

            features['accumulated_accuracy_to_mean_title2'] = accumulated_accuracy/counter if counter > 0 else 0

            features['accumulated_accuracy_to_mean_title3'] = accumulated_accuracy/counter if counter > 0 else 0

            features['accumulated_accuracy_to_mean_title4'] = accumulated_accuracy/counter if counter > 0 else 0

            features['accumulated_accuracy_to_mean_title5'] = accumulated_accuracy/counter if counter > 0 else 0'''

            

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else -1

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

             

            features['accumulated_accuracy_game'] = accumulated_accuracy_game/counter_game if counter_game > 0 else -1

            features['accumulated_correct_attempts_game'] = accumulated_correct_attempts_game

            features['accumulated_uncorrect_attempts_game'] = accumulated_uncorrect_attempts_game

            #add by NorwayPing 2019-12-30 

            

            features['adjust_acum_accuracy'] = adjust_accumulated_accuracy/counter if counter > 0 else -1

            

            for title_game_i in count_title_game.keys():

                if(count_title_game[title_game_i]!=0):

                    acumulated_accuracy_title_game ['acum_acc_' + title_game_i] =acum_accuracy_title_game[title_game_i]/count_title_game[title_game_i]

                

           

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            

            #add by NorwayPing

            features['accuracy'] = accuracy

            accumulated_accuracy += accuracy

            #add by NorwayPing 2019-12-30 

            '''title

            4     0.387378936777966

            8     0.742232447460104

            9     0.735536607179642

            10    0.248625352736353

            30    0.711041950242111'''

            adjust_accuracy = min(accuracy * accuracy_weight_maps[session_title_text],1)

            adjust_accumulated_accuracy += adjust_accuracy

            

            if accuracy <= 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else -1

            

                #add by NorwayPing

            if len(all_attempts)!=0:

                accuracy_groups[features['accuracy_group']] += 1     

                #last_accuracy_title['acc_' + session_title_text] = features['accuracy_group']

                last_accuracy_title['acc_' + session_title_text] = accuracy

                accumulated_accuracy_group += features['accuracy_group']            

               

                #last_accuracy_mean_title['acc_mean_' + session_title_text] =   labels_map_acc_mean[session_title]

                #last_accuracy_std_title['acc_std_' + session_title_text] =   labels_map_acc_std[session_title]

                #add by NorwayPing 20129-12-30

                acu_dura_title['acum_dura_'+ session_title_text]+= (session.iloc[-1, 2] - session.iloc[0, 2] ).seconds

                

               # last_accur   

            # last_accuracy_std_title['acc_std_' + session_title_text] =  accuracy/labels_map_acc_num[session_title]

                

            #accuracy_groups[features['accuracy_group']] += 1

            #last_accuracy_title['acc_' + session_title_text] = features['accuracy_group']

            #accumulated_accuracy_group += features['accuracy_group']

                  

         

            

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            #features['accumulated_actions'] = accumulated_actions

            features['accumulated_actions'] =  np.log1p(accumulated_actions)

           

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            #add by NorwayPing

            if len(all_attempts)!=0:

                counter += 1

          #  counter += 1

        

       

        

        # Add by NorwayPing for each game, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Game') & (len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts_game = session.query(f'event_code == {win_code_game[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts_game = all_attempts_game['event_data'].str.contains('true').sum()

            false_attempts_game = all_attempts_game['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            

          



            accumulated_correct_attempts_game += true_attempts_game

            accumulated_uncorrect_attempts_game += false_attempts_game

          

            

            accuracy_game = true_attempts_game /(true_attempts_game +false_attempts_game) if (true_attempts_game+false_attempts_game) != 0 else -1

            if (true_attempts_game+false_attempts_game) != 0:

                accumulated_accuracy_game += accuracy_game

                last_accuracy_title_game['acc_' + session_title_text] = accuracy_game

                acum_accuracy_title_game[session_title_text] += accuracy_game

                count_title_game[session_title_text]+=1

                acum_count_title_game ['acum_count_' + session_title_text] += len(session)

                #last_dura_title_game['dura_' + session_title_text] = session_duration/true_attempts_game if true_attempts_game > 0 else -1 

                #duration_game.append(session_duration)

           

                

            counter_game += 1

        '''

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter1: dict,counter2: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter1[x] += num_of_session_count[k]

                return counter1,counter2

            

        event_code_count = update_counters(event_code_count, "event_code")

         #delete by NorwayPing to reduce duplication

        #event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code') '''   

        

                   # this piece counts how many actions was made in each event_code so far

        def update_counters_log(counter: dict, counter_log: dict, col: str):

            num_of_session_count = Counter(session[col])

            for k in num_of_session_count.keys():

                x = k

                if col == 'title':

                    x = activities_labels[k]

                counter[x] += num_of_session_count[k]

                counter_log[x] =np.log1p(counter[x])

            return counter,counter_log

        

        event_code_count,event_code_log = update_counters_log(event_code_count, event_code_log,"event_code")

         #delete by NorwayPing to reduce duplication

        #event_id_count = update_counters(event_id_count, "event_id")

        title_count,title_log = update_counters_log(title_count,title_log,'title')

        title_event_code_count,title_event_code_log = update_counters_log(title_event_code_count, title_event_code_log, 'title_event_code')

        

        # add by NorwayPing this piece counts how many actions was made in each event_code so far

        def pec_counters(pec:dict, col: str):

                num_of_session_count = Counter(session[col])

                # title_event_code_percent.loc[title_event_code_percent['title_event_code'].str.contains(session['title']+'_'),0]=0

                title_all_code= [s for s in all_title_event_code if activities_labels[session_title] in s]

                #breakpoint()

                for k in title_all_code:

                    pec['pec_'+k] = 0

                 

               #breakpoint()

                   

                for k in num_of_session_count.keys():

                    x = k

                    pec['pec_'+x] = num_of_session_count[k]/len(session)

                return pec   

        #add by NorwayPing 2019/12/19

        title_event_code_percent= pec_counters(title_event_code_percent, 'title_event_code')

        

        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        

        #Change by NorwayPing

        user_activities_count[session_type] += 1

        #clip_session_count += 1 if(session_type) == 'Clip' else 0

   

          

        #add by NorwayPing

        #duration_all += session_duration

        '''if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type '''

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
# read data during submission

train1, test1, train_labels1, specs, sample_submission = read_data_2()

#read local file

#train1, test1, train_labels1, specs, sample_submission = read_data()
# get usefull dict with maping encode

train, test, train_labels, win_code, win_code_game, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, game_titles, list_of_event_id, all_title_event_code, labels_map,labels_map_acc_mean,labels_map_acc_std = encode_title_2(train1, test1, train_labels1)



reduce_train, reduce_test, categoricals = get_train_and_test_2(train, test)
def preprocess(reduce_train, reduce_test):

    #for df in [reduce_train, reduce_test]:

        #move to get_data by NorwayPing due to data leakage

        #df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')

        

        #delete by NorwayPing due to data leakage

        #df['title_duration_mean'] = df.groupby(['session_title'])['duration_mean'].transform('mean')

        

        #df['title_duration_std'] = df.groupby(['session_title'])['duration_mean'].transform('std')

        

        #move to get_data by NorwayPing due to data leakage

        #df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        

        #delete by NorwayPing, it is the same as accumulated actions

        #df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

        #                                4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

        #                                2040, 4090, 4220, 4095]].sum(axis = 1)

        

        #df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

      

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

    #add by NorwayPing to solve reprodutivity issue

    '''num_cols = reduce_train.select_dtypes('number').columns

    reduce_train_numberic = reduce_train[sorted( num_cols.tolist())]

    reduce_test_numberic = reduce_test[sorted(reduce_train.select_dtypes('number').columns.tolist())]

    

    reduce_train_obj= reduce_train[(reduce_train.select_dtypes('object').columns.tolist())]

    reduce_test_obj = reduce_test[(reduce_train.select_dtypes('object').columns.tolist())]

    

    reduce_train = [reduce_train_numberic , reduce_train_obj]

    reduce_test = [reduce_test_numberic , reduce_test_obj]'''

    

    return reduce_train, reduce_test, features

# call feature engineering function

reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
params = {'n_estimators':2000,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.04,

            'feature_fraction': 0.9,

         'max_depth': 18,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 100, 'eval_metric': 'cappa'

            }
y = reduce_train['accuracy_group']



#add by NorwayPing

#y = reduce_train['accuracy']
reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]
# Added by TJ Klein : single unique and correlated features removal

cols_to_drop = ['installation_id', 'accuracy_group', 'accuracy','session_id','time','date'] 
cols_to_drop = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in cols_to_drop]
n_fold = 5

folds = GroupKFold(n_splits=n_fold)
mt = MainTransformer()

ft = FeatureTransformer()

transformers = {'ft': ft}



regressor_model1 = RegressorModel(model_wrapper=LGBWrapper_regr())

regressor_model1.fit(X=reduce_train, y=y, folds=folds, params=params, preprocesser=mt, transformers=transformers,

                    eval_metric='cappa', cols_to_drop=cols_to_drop)



from functools import partial

import scipy as sp

class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa (QWK) score

    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

    """

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        

        :param coef: A list of coefficients that will be used for rounding

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])



        return -qwk(y, X_p)



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        

        :param X: The raw predictions

        :param y: The ground truth labels

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        

        :param X: The raw predictions

        :param coef: A list of coefficients that will be used for rounding

        """

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])





    def coefficients(self):

        """

        Return the optimized coefficients

        """

        return self.coef_['x']
pr1 = regressor_model1.predict(reduce_train)

optR = OptimizedRounder()

optR.fit(pr1.reshape(-1,), y)

coefficients = optR.coefficients()
opt_preds = optR.predict(pr1.reshape(-1, ), coefficients)

qwk(y, opt_preds)
# some coefficients calculated by me.

pr1 = regressor_model1.predict(reduce_test)

pr1[pr1 <= 1.12232214] = 0

pr1[np.where(np.logical_and(pr1 > 1.12232214, pr1 <= 1.73925866))] = 1

pr1[np.where(np.logical_and(pr1 > 1.73925866, pr1 <= 2.22506454))] = 2

pr1[pr1 > 2.22506454] = 3



sample_submission['accuracy_group'] = pr1.astype(int)

sample_submission.to_csv('submission.csv', index=False)
sample_submission['accuracy_group'].value_counts(normalize=True)
def adversarial_validation_func(x_train,x_test):

    # add identifier and combine

    x_train['istrain'] = 1

    x_test['istrain'] = 0

    #reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

    #reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]

    xdat = pd.concat([x_train, x_test], axis = 0)

    

   

    '''

    # convert non-numerical columns to integers

    df_numeric = xdat.select_dtypes(exclude=['object'])

    df_obj = xdat.select_dtypes(include=['object']).copy()



    for c in df_obj:

        df_obj[c] = pd.factorize(df_obj[c])[0]



    xdat = pd.concat([df_numeric, df_obj], axis=1)'''

    y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)

    return xdat,y

    
from sklearn.metrics import roc_auc_score

xdat,y_a = adversarial_validation_func(reduce_train,reduce_test)

#cols_to_drop = ['installation_id', 'accuracy_group', 'accuracy','session_id']

cols_to_drop2=cols_to_drop+['acum_dura_Bird_Measurer__Assessment_','acum_dura_Chest_Sorter__Assessment_','acum_dura_Mushroom_Sorter__Assessment_','acum_dura_Cauldron_Filler__Assessment_','acum_dura_Cart_Balancer__Assessment_']

xdat = xdat[[col for col in xdat.columns if col not in cols_to_drop]] 

#xdat = xdat.drop([cols_to_drop], axis=1)

#del xdat[cols_to_drop]

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44)

xgb_params = {

            'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,

            'colsample_bytree': 0.9,'objective': 'binary:logistic',

            'silent': 1, 'n_estimators':100, 'gamma':1,

            'min_child_weight':4

            }   

clf = xgb.XGBClassifier(**xgb_params, seed = 10)     

for train_index, test_index in skf.split(xdat, y_a):

    x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]

    y0, y1 = y_a.iloc[train_index], y_a.iloc[test_index]        

    print(x0.shape)

    clf.fit(x0, y0, eval_set=[(x1, y1)],

               eval_metric='logloss', verbose=False,early_stopping_rounds=10)

                

    prval = clf.predict_proba(x1)[:,1]

    print(roc_auc_score(y1,prval))
importance = clf.feature_importances_
feat_importances = pd.Series(clf.feature_importances_, index=xdat.columns)

feat_importances.nlargest(20).plot(kind='barh')