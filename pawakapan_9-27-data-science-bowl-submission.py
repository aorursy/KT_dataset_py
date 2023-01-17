'''

Heavily borrows from https://www.kaggle.com/artgor/quick-and-dirty-regression

'''



import numpy as np

import pandas as pd

import os

import copy

import matplotlib.pyplot as plt

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

from sklearn.base import BaseEstimator, TransformerMixin

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

from typing import List



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

from typing import Any

from itertools import product

pd.set_option('max_rows', 500)

import re

from tqdm import tqdm

from joblib import Parallel, delayed



print(lgb.__version__)
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





def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3



    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)



    return 'cappa', qwk(y_true, y_pred), True





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

            fig, ax = plt.subplots(figsize=(16, 12))

            plt.subplot(2, 2, 1)

            self.plot_feature_importance(top_n=20)

            plt.subplot(2, 2, 2)

            self.plot_metric()

            plt.subplot(2, 2, 3)

            plt.hist(y.values.reshape(-1, 1) - self.oof)

            plt.title('Distribution of errors')

            plt.subplot(2, 2, 4)

            plt.hist(self.oof)

            plt.title('Distribution of oof predictions');

            

        return np.mean([self.folds_dict[i]['scores']['valid']['cappa'] for i in self.folds_dict])



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
'''

Reads relevant competition files into dataframes

'''

def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('/kaggle/input/dsbowl2019original/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/dsbowl2019original/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/dsbowl2019original/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    return train, test, train_labels



'''

Categorical encodings for titles, events, and world names (text)

Returns lists enumerating each encoding type

'''

def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    

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

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
# Read data from competition files

train, test, train_labels = read_data()

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# dictionary of clip lengths (in seconds) for each clip name - this was additional data provided during the competition

cliplen = {'Welcome to Lost Lagoon!': 19.0,

 'Tree Top City - Level 1': 17.0,

 'Ordering Spheres': 61.0,

 'Costume Box': 61.0,

 '12 Monkeys': 109.0,

 'Tree Top City - Level 2': 25.0,

 "Pirate's Tale": 80.0,

 'Treasure Map': 156.0,

 'Tree Top City - Level 3': 26.0,

 'Rulers': 126.0,

 'Magma Peak - Level 1': 20.0,

 'Slop Problem': 60.0,

 'Magma Peak - Level 2': 22.0,

 'Crystal Caves - Level 1': 18.0,

 'Balancing Act': 72.0,

 'Lifting Heavy Things': 118.0,

 'Crystal Caves - Level 2': 24.0,

 'Honey Cake': 142.0,

 'Crystal Caves - Level 3': 19.0,

 'Heavy, Heavier, Heaviest': 61.0}



# issue with latest lightgbm version - non-alphanumeric characters in feature names need to be replaced w/ "_"

def toalphacol(x):

    return "".join (c if c.isalnum() else "_" for c in str(x))



'''

For each installation_id in the dataset (instance of the game),

features are extracted from a sequence of events into a single feature set for that installation_id.

This is the first stage of feature processing - further engineering based on stats/aggregation is done later

'''

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    

    last_activity = 0

    

    # accumulated values during the installation

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    user_activities_count = {'Clip_short':0, 'Clip_long':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    user_activities_duration = {'Clip_short':0, 'Clip_long':0, 'Activity':0, 'Assessment':0, 'Game':0}

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

    event_code_count = {ev: 0 for ev in list_of_event_code}

    event_id_count = {eve: 0 for eve in list_of_event_id}

    title_count = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count = {t_eve: 0 for t_eve in all_title_event_code}

    world_count = {world: 0 for world in range(4)}

    

    title_duration = {eve: 0 for eve in activities_labels.values()}

    world_duration = {world: 0 for world in range(4)}

    

    last_event_code_count = {ev: 0 for ev in list_of_event_code}



    # iterate through each session_id (game session) of an installation

    for i, session in user_sample.groupby('game_session', sort=False):

        # session is a DataFrame that contain only one game_session



        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]



        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some items: 

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            

            # stats for the current world, and most recent event code

            features.update({'World_'+str(k):v for k,v in world_count.items()})

            features.update({'Last_'+str(k):v for k,v in last_event_code_count.items()})

            

            # duration spent in each activity, title, and world type

            features.update({'DRT '+str(k):v for k,v in user_activities_duration.items()})

            features.update({'DRT '+str(k):v for k,v in title_duration.items()})

            features.update({'DRT World_'+str(k):v for k,v in world_duration.items()})



            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            features['world'] = session['world'].iloc[0]

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

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            # last accuracy

            features['self_last_accuracy'] = float(last_accuracy_title['acc_'+session['title_event_code'].iloc[0].split('_')[0]])

            

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # title count

            features['self_session_title_count'] = title_count[activities_labels[session['title'].iloc[0]]]

            # type count

            features['self_type_count'] = user_activities_count[session['type'].iloc[0]]

            # title_event_code count

            features['self_tec_count'] = title_event_code_count[session['title_event_code'].iloc[0]]

            # world count

            features['self_world_count'] = world_count[session['world'].iloc[0]]                  

            

            # title,type,world duration

            features['DRT self_session_title'] = title_duration[activities_labels[session['title'].iloc[0]]]

            features['DRT self_type'] = user_activities_duration[session['type'].iloc[0]]

            features['DRT self_world'] = world_duration[session['world'].iloc[0]]

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)



            counter += 1



        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

            num_of_session_count = Counter(session[col])

            for k in num_of_session_count.keys():

                x = k

                if col == 'title':

                    x = activities_labels[k]

                counter[x] += num_of_session_count[k]

            return counter

        

        if session_type == 'Clip':

            clip_name = session['title_event_code'].iloc[0].split('_')[0]

            drt = cliplen[clip_name]

            session_type = 'Clip_long' if '-' in clip_name else 'Clip_short'

        else:

            drt = (session['timestamp'].iloc[-1]-session['timestamp'].iloc[0]).seconds

            

        # update durations for user_activities, title, world

        title_duration[activities_labels[session_title]] += drt

        world_duration[session['world'].iloc[0]] += drt



        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        world_count = update_counters(world_count, 'world')

        

        last_event_code = session['event_code'].iloc[-1]

        last_event_code_count[last_event_code] += 1



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            user_activities_duration[session_type] += drt

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments



def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals



train_df, test_df, categoricals = get_train_and_test(train, test)
'''

Further feature engineering, primarily based on aggregations of features generated by <get_data()>

'''

def preprocess(reduce_train, reduce_test, drop_nonpct=False):

    for df in [reduce_train, reduce_test]:

        

        # action count, time spent, etc. for each title, event

        df['installation_session_count'] = df.groupby(['installation_id'])['Clip_short'].transform('count')

        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')

        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')

        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 

                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 

                                        2040, 4090, 4220, 4095]].sum(axis = 1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        

        # collect sub-dataframes for all event_codes/event_ids/titles/worlds, etc. for percentage calculations

        # i.e. count of <> type as percentage of all types

        event_codes = df[list_of_event_code]

        event_id = df[list_of_event_id]

        event_tec = df[all_title_event_code]

        

        gametypes = ['Activity','Assessment','Clip_short','Clip_long','Game']

        titles = [activities_labels[k] for k in activities_labels]

        event_types = df[gametypes]

        event_titles = df[titles]

        event_worlds = df[['World_'+str(i) for i in range(4)]]



        # compute sum over all types

        codes_denom = event_codes.sum(axis=1)

        id_denom = event_id.sum(axis=1)

        tec_denom = event_tec.sum(axis=1)

        type_denom = event_types.sum(axis=1)

        title_denom = event_titles.sum(axis=1)

        world_denom = event_worlds.sum(axis=1)

        

        last_code_denom = df[['Last_'+str(ec) for ec in list_of_event_code]].sum(axis=1)

        

        type_duration_denom = df[['DRT '+t for t in gametypes]].sum(axis=1)

        title_duration_denom = df[['DRT '+t for t in titles]].sum(axis=1)

        world_duration_denom = df[['DRT World_'+str(i) for i in range(4)]].sum(axis=1)

        

        # add percentage features

        for code in list_of_event_code:

            df['PCT '+str(code)] = df[code]/codes_denom

        for id_ in list_of_event_id:

            df['PCT '+id_] = df[id_]/id_denom

        for tec in all_title_event_code:

            df['PCT '+tec] = df[tec]/tec_denom

        for type_ in gametypes:

            df['PCT '+type_] = df[type_]/type_denom

        for title in titles:

            df['PCT '+title] = df[title]/title_denom

        for i in range(4):

            df['PCT World_'+str(i)] = df['World_'+str(i)]/world_denom

            

        for ec in list_of_event_code:

            df['PCT Last_'+str(ec)] = df['Last_'+str(ec)]/last_code_denom

            

        for t in gametypes:

            df['PCT DRT '+t] = df['DRT '+t]/type_duration_denom

        for t in titles:

            df['PCT DRT '+t] = df['DRT '+t]/title_duration_denom

        for i in range(4):

            df['PCT DRT World_'+str(i)] = df['DRT World_'+str(i)]/world_duration_denom

            

        df['self_session_title_count_pct'] = df['self_session_title_count']/title_denom

        df['self_type_count_pct'] = df['self_type_count']/type_denom

        df['self_tec_count_pct'] = df['self_tec_count']/tec_denom

        df['self_world_count_pct'] = df['self_world_count']/world_denom

        

        df['PCT DRT self_session_title'] = df['DRT self_session_title']/title_duration_denom

        df['PCT DRT self_type'] = df['DRT self_type']/type_duration_denom

        df['PCT DRT self_world'] = df['DRT self_world']/world_duration_denom

        

        # distinguish clip features based on duration (long/short)

        clipshort_cols = list(filter(lambda x:'Clip_short' in str(x), df.columns))

        cliplong_cols = list(map(lambda x:'Clip_long'.join(str(x).split('Clip_short')), clipshort_cols))

        for csc,csl in zip(clipshort_cols,cliplong_cols):

            df['Clip'.join(csc.split('Clip_short'))] = df[csc]+df[csl]

            

        # features based on clip duration (duration spent on video clips, per clip, etc.)

        clip_duration_raw = df[[x for x in cliplen]]*[cliplen[x] for x in cliplen]

        df['clip_duration_rawtotal'] = clip_duration_raw.sum(axis=1)

        df['clip_duration_rawperclip'] = df['clip_duration_rawtotal']/df['Clip']

        df['clip_duration_rawpertype'] = clip_duration_raw.mean(axis=1)

        

        # try splitting clip types by short/long (if '-' is in title)

        clipshortlen = {k:cliplen[k] for k in filter(lambda x:'-' in x, cliplen.keys())}

        cliplonglen = {k:cliplen[k] for k in filter(lambda x:not '-' in x, cliplen.keys())}

        df['Clipshort'] = df[[x for x in clipshortlen]].sum(axis=1)

        df['Cliplong'] = df[[x for x in cliplonglen]].sum(axis=1)



        # compute same clip stats, but separating long/short clips

        clipshort_duration_raw = df[[x for x in clipshortlen]]*[clipshortlen[x] for x in clipshortlen]

        df['clipshort_duration_rawtotal'] = clipshort_duration_raw.sum(axis=1)

        df['clipshort_duration_rawperclip'] = df['clipshort_duration_rawtotal']/df['Clip']

        df['clipshort_duration_rawpertype'] = clipshort_duration_raw.mean(axis=1)

        cliplong_duration_raw = df[[x for x in cliplonglen]]*[cliplonglen[x] for x in cliplonglen]

        df['cliplong_duration_rawtotal'] = cliplong_duration_raw.sum(axis=1)

        df['cliplong_duration_rawperclip'] = df['cliplong_duration_rawtotal']/df['Clip']

        df['cliplong_duration_rawpertype'] = cliplong_duration_raw.mean(axis=1)

        

    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]

   

    if drop_nonpct: # drops raw counts and keeps only percentages if true

        to_drop = list_of_event_code+list_of_event_id+all_title_event_code

        reduce_train = reduce_train.drop(columns=to_drop)

        reduce_test = reduce_test.drop(columns=to_drop)

    return reduce_train,reduce_test,features
# apply further feature engineering for each installation_id

train_df, test_df, feature_names = preprocess(train_df, test_df)
# process feature names to remove non-alphanumeric characters

# compatibility issue with latest lightgbm version

train_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_df.columns]

test_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test_df.columns]
def train_model(X, pms, cols_to_drop):

    mt = MainTransformer()

    regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())

    score = regressor_model.fit(X=X, y=y, folds=folds, params=pms, preprocesser=mt, transformers={},

                       eval_metric='cappa', cols_to_drop=cols_to_drop)

    return regressor_model



# hyperparameters from bayesian optimization with the same features

# (not done in this kernel)

params = {'n_estimators':2000,

          'num_leaves':31,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.04,

            'feature_fraction': 0.9,

         'max_depth': 15,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 100, 'eval_metric': 'cappa',

              'min_child_samples': 20

            }



y = train_df['accuracy_group']

n_fold = 5

folds = GroupKFold(n_splits=n_fold)



# label features

cols_to_drop = ['accuracy_group','installation_id']

# these features led to overfitting due to train/test mismatch

overfit_cols = ['installation_duration_mean','installation_title_nunique',

    'installation_session_count','installation_event_code_count_mean']
model = train_model(train_df.drop(columns=overfit_cols), params, cols_to_drop)
'''

Code for optimized rounding & making predictions

- LightGBM model is trained on regression problem, we convert to classification by finding optimal thresholds on out-of-fold 

'''

from functools import partial

import scipy as sp



def round_w_coef(preds, coefs):

    pr = np.array(preds).reshape([-1])

    c1,c2,c3 = coefs

    pr[pr <= c1] = 0

    pr[np.where(np.logical_and(pr > c1, pr <= c2))] = 1

    pr[np.where(np.logical_and(pr > c2, pr <= c3))] = 2

    pr[pr > c3] = 3

    return list(pr)



'''

Training is done as regression problem (0-3)

To convert to classification, optimized rounding finds ideal score thresholds for each category

'''

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk(y, X_p)



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])



    def coefficients(self):

        return self.coef_['x']

    

def get_optround_preds(trp, trY, X):

    optr = OptimizedRounder()

    optr.fit(trp, trY)

    preds = optr.predict(X, optr.coefficients())

    return np.array(preds)
# cross-validation, and adversarial validation



train_outputs = model.oof.reshape([-1]) # cross-validation out-of-fold predictions

train_preds = get_optround_preds(train_outputs, train_df['accuracy_group'], train_outputs)



y = train_df['accuracy_group']

train_qwk = qwk(y, train_preds)

train_accuracy = (train_preds == y).mean()



# adversarial validation



# the installation_session_count feature has a clear train/test mismatch - values in the test set are all 1

# during competition, saw that feature engineering on this improved CV score at expense of test scores

# this metric measures accuracy on points w/ value 1, to decrease train/test mismatch

train_accuracy_subset = (train_preds[train_df['installation_session_count']==1] == y[train_df['installation_session_count']==1]).mean()



print("VALIDATION out-of-fold")

print("Training QWK : {:.4f}".format(train_qwk))

print("Training accuracy : {:.4f}".format(train_accuracy))

print("Adversarial val | Training accuracy for installation_session_count=1 : {:.4f}".format(train_accuracy_subset))
# predict regression scores on train & test, and use train predictions & labels to do optimized rounding

# use optimized rounding to return classifications on test set

test_outputs = model.predict(test_df.drop(columns=cols_to_drop)).reshape([-1])

test_preds = get_optround_preds(train_outputs, y, test_outputs)



submission = pd.read_csv('/kaggle/input/dsbowl2019original/sample_submission.csv')

submission['accuracy_group'] = test_preds
submission