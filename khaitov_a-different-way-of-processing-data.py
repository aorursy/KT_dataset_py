import pandas as pd

import numpy as np

import ast

from tqdm import tqdm_notebook

import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import text

from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from numba import jit



import time

import copy

from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders.ordinal import OrdinalEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold

from functools import partial

import scipy as sp



# making the workspace wider

# from IPython.core.display import display, HTML

# display(HTML("<style>.container { width:80% !important; }</style>"))



# more information in the tables

pd.set_option('display.max_columns',150)

pd.set_option('display.max_rows',150)

pd.set_option('display.float_format', lambda x: '%.5f' % x)



# PARAMS_CALC = {

#     'delete from test/train assessment with correct&uncorrect == 0':True

# }



# if start on kaggle server set True

KAGGLE_START = True

BAYES_OPTIMIZATION = False



# if KAGGLE_START:

#     BAYES_OPTIMIZATION = True

# else:

#     BAYES_OPTIMIZATION = False

    

n_fold = 5

folds = False

# folds = GroupKFold(n_splits=n_fold)
# Source: https://www.kaggle.com/artgor/quick-and-dirty-regression

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

        

        if folds!=False:

            # simple folds split

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

        

        else:

            # split fold with KMeans clusters, and select in val test max assessment per user in group folds.

            for fold_n in sorted(X['KFold_Group'].astype('int8').unique()):

                valid_index = X[

                    X['KFold_Group']==fold_n

                ][['installation_id', 'timestamp']].groupby('installation_id').idxmax()['timestamp']

                train_index = X.drop(valid_index).index



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



# to maximize Quadratic Weighted Kappa (QWK) score

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
%%time



#### 1. GET DATA 

if KAGGLE_START:

    PATH = '/kaggle/input/data-science-bowl-2019/'

else:

    PATH = './'



train_df = pd.read_csv(PATH+'train.csv')

train_df['timestamp'] = pd.to_datetime(train_df['timestamp']).dt.tz_convert(None)



test_df = pd.read_csv(PATH+'test.csv')

test_df['timestamp'] = pd.to_datetime(test_df['timestamp']).dt.tz_convert(None)



# prepare float columns

train_df['event_code'] = train_df['event_code'].astype('uint16')

test_df['event_code'] = test_df['event_code'].astype('uint16')

train_df['event_count'] = train_df['event_count'].astype('uint16')

test_df['event_count'] = test_df['event_count'].astype('uint16')

train_df['game_time'] = train_df['game_time'].fillna(0).astype('uint32')

test_df['game_time'] = test_df['game_time'].fillna(0).astype('uint32')



train_labels_df = pd.read_csv(PATH+'train_labels.csv')

specs_df = pd.read_csv(PATH+'specs.csv')

sample_submission_df = pd.read_csv(PATH+'sample_submission.csv')

print('1. Read get data')





#### 2. Get correct/uncorrect in train/test

for df in [train_df, test_df]:

    

    df['correct'] = df['event_data'].str.contains('true').astype('uint8')

    df['correct'] = np.where(

        (

            (df['type'] == 'Assessment')&\

            (df['title'] == 'Bird Measurer (Assessment)')&\

            (df['event_code'] == 4110)

        ) | (

            (df['type'] == 'Assessment')&\

            (df['title'] != 'Bird Measurer (Assessment)')&\

            (df['event_code'] == 4100)

        )

        , df['correct']

        , 0

    ).astype('uint8')

    

    df['uncorrect'] = df['event_data'].str.contains('false').astype('uint8')

    df['uncorrect'] = np.where(

        (

            (df['type'] == 'Assessment')&\

            (df['title'] == 'Bird Measurer (Assessment)')&\

            (df['event_code'] == 4110)

        ) | (

            (df['type'] == 'Assessment')&\

            (df['title'] != 'Bird Measurer (Assessment)')&\

            (df['event_code'] == 4100)

        )

        , df['uncorrect']

        , 0

    ).astype('uint8')

    

    # df.drop('event_data', axis=1,inplace=True)

    

train_df['assessment_code'] = np.where(train_df['correct']+train_df['uncorrect']>0, 1,0).astype('uint8')

test_df['assessment_code'] = np.where(test_df['correct']+test_df['uncorrect']>0, 1,0).astype('uint8')

gc.collect();

print('2. Get correct/uncorrect in train/test')



# N_JOBS_PARALLEL = 8

# def mapAst(s,idrow, replace_true = True):

#     if replace_true:

#         s = ast.literal_eval(s.replace('true','True').replace('false', 'False'))

#     else:

#         s = ast.literal_eval(s)

#     s.update({'ID_ROW':idrow})

#     return s



# # train map dict

# %time train_event_data = Parallel(n_jobs=N_JOBS_PARALLEL)(delayed(mapAst)(s, idrow) for idrow, s in enumerate(tqdm_notebook(train_df[train_df['assessment_code']==1]['event_data'])))

# %time train_event_data = pd.DataFrame.from_dict(train_event_data)



# # test map dict

# %time test_event_data = Parallel(n_jobs=N_JOBS_PARALLEL)(delayed(mapAst)(s, idrow) for idrow, s in enumerate(tqdm_notebook(test_df[test_df['assessment_code']==1]['event_data'])))

# %time test_event_data = pd.DataFrame.from_dict(test_event_data)



# evendata_features = list(set(train_event_data.columns) & set(test_event_data.columns))

# train_event_data = train_event_data[evendata_features]

# test_event_data = test_event_data[evendata_features]



train_df.drop('event_data', axis=1,inplace=True)

test_df.drop('event_data', axis=1,inplace=True)

gc.collect();

print('2.1 parse event_data assessment only in dfs train_event_data/test_event_data')



#### 3. clear Assessment without accuracy_group

train_df = train_df[~((train_df['type']=='Assessment')&(train_df['correct']+train_df['uncorrect']==0))]



train_df.sort_values(['installation_id', 'world', 'timestamp'], inplace=True)

test_df.sort_values(['installation_id', 'world', 'timestamp'], inplace=True)



train_df.reset_index(drop=True, inplace=True);

test_df.reset_index(drop=True, inplace=True);

gc.collect();

print('3. clear Assessment without accuracy_group')





#### 4. type dummies

for feature in ['type']: # 'world', 

    train_df = pd.concat(

        [

            train_df,

            pd.get_dummies(train_df[feature], prefix='Is',prefix_sep='')

        ],

        axis = 1,

        sort=False

    )

    test_df = pd.concat(

        [

            test_df,

            pd.get_dummies(test_df[feature], prefix='Is',prefix_sep='')

        ],

        axis = 1,

        sort=False

    )



print('4. type dummies')



#### 5. Search last assessment for notes predict rows

dimension_key = ['installation_id', 'world', 'Assessment_id', 'name_assessment']

# search predict rows (last assessment's)

test_df = pd.merge(

    test_df,

    test_df.merge(

        test_df.groupby('installation_id')['timestamp'].max().reset_index()

    )[['game_session', 'installation_id', 'IsAssessment']].rename(columns={'IsAssessment':'Predict_rows'}),

    on = ['game_session', 'installation_id'],

    how = 'left'

)

test_df['Predict_rows'] = test_df['Predict_rows'].fillna(0).astype('uint8')

# so that the signs do not differ

train_df['Predict_rows'] = 0

print('5. Search last assessment for notes predict rows')



# #### 5.1 delete from test/train assessment with correct&uncorrect

# train_df = pd.concat(

#     [

#         train_df,

#         train_df.groupby(

#             ['installation_id', 'game_session', 'type']

#         )['correct', 'uncorrect', 'Predict_rows'].transform('sum').rename(

#             columns={

#                 'correct':'correct_ig',

#                 'uncorrect':'uncorrect_ig',

#                 'Predict_rows':'Predict_rows_ig'

#             }

#         )

#     ],

#     axis=1,

#     sort=False

# )

# test_df = pd.concat(

#     [

#         test_df,

#         test_df.groupby(

#             ['installation_id', 'game_session', 'type']

#         )['correct', 'uncorrect', 'Predict_rows'].transform('sum').rename(

#             columns={

#                 'correct':'correct_ig',

#                 'uncorrect':'uncorrect_ig',

#                 'Predict_rows':'Predict_rows_ig'

#             }

#         )

#     ],

#     axis=1,

#     sort=False

# )

# train_df['is_drop_assessment_id'] = np.where(

#     ((train_df['correct_ig'].fillna(0)+train_df['uncorrect_ig'].fillna(0))==0) & (train_df['type']=='Assessment') & (train_df['Predict_rows_ig']==0)

#     , 1

#     , 0

# )

# test_df['is_drop_assessment_id'] = np.where(

#     ((test_df['correct_ig'].fillna(0)+test_df['uncorrect_ig'].fillna(0))==0) & (test_df['type']=='Assessment') & (test_df['Predict_rows_ig']==0)

#     , 1

#     , 0

# )

# train_df.drop(train_df[train_df['is_drop_assessment_id']==1].index, inplace=True)

# train_df.drop(['is_drop_assessment_id', 'correct_ig', 'uncorrect_ig'], axis=1, inplace=True)

# test_df.drop(test_df[test_df['is_drop_assessment_id']==1].index, inplace=True)

# test_df.drop(['is_drop_assessment_id', 'correct_ig', 'uncorrect_ig'], axis=1, inplace=True)

# print('5.1 delete from test/train assessment with correct&uncorrect')



#### 6. get number assessment_id per installation_id, world, title assessment

def get_assessmentid(df):

    df['IsAssessment'] = np.where(df['type']=='Assessment', 1, 0)

    df = df.merge(

        df[df['IsAssessment']==1].groupby(['installation_id', 'game_session'])['event_count'].max().reset_index().rename(columns={'event_count':'max_step_assessment'}),

        how = 'left',

        copy = True

    )

    

    # find the last step in each Assessment

    df['max_step_assessment'] = np.where(df['max_step_assessment']==df['event_count'], 1, 0)

    # move the unit 1 line down (from the last assessment action) - this will be the beginning of the next one

    # shift in the context of the person/world, all that is empty is the first line, replace the void with 1.

    df['firstRow'] = df.groupby(['installation_id', 'world'])['max_step_assessment'].shift(1).fillna(1)

    # accumulate Assessment id

    df['Assessment_id'] = df.groupby(['installation_id', 'world'])['firstRow'].cumsum()

    # in the context of the accumulated Assessment_id (attempts), output the name of the test

    df['title_assessment'] = np.where(df['IsAssessment']==1, df['title'], '')

    df['name_assessment'] = df.groupby(['installation_id', 'world', 'Assessment_id'])['title_assessment'].transform('max')

    df.drop('title_assessment', axis=1, inplace=True)

    # accumulate the Assessment id in the test section

    df['Assessment_id'] = df.groupby(['installation_id', 'world', 'name_assessment'])['firstRow'].cumsum().astype('int8')

    df.drop(['firstRow', 'max_step_assessment'], axis=1, inplace=True)

    return df

train_df = get_assessmentid(train_df)

test_df = get_assessmentid(test_df)

gc.collect();

print('6. get number assessment_id per installation_id, world, title assessment')





#### 7. timestamp - duration

for df in [train_df, test_df]:

    df['timestamp_shift'] = df.groupby(dimension_key)['timestamp'].shift()

    df['duration'] = ((df['timestamp'] - df['timestamp_shift']).fillna(0).astype('int64') / int(1e9)).astype('float32')

    df['duration'].fillna(0,inplace=True)

    df.drop('timestamp_shift', axis=1, inplace=True)

gc.collect();

print('7. timestamp - duration')





#### 8. name_types first row 1, for unique types with cumsum 1

name_types = ['IsActivity', 'IsAssessment', 'IsClip', 'IsGame']

for df in [train_df, test_df]:

    for name_type in name_types:

        df['shift'+name_type] = df.groupby(dimension_key)[name_type].shift(-1).fillna(0).astype('uint8')

        df[name_type+'UniqueAction'] = (df[name_type]-df['shift'+name_type]).astype('int8')

        df[name_type+'UniqueAction'] = np.where(df[name_type+'UniqueAction']==1, 1, 0)

        df.drop(['shift'+name_type], axis=1, inplace=True)

print('8. name_types first row 1, for unique types with cumsum 1')





#### 9. Del world NONE

train_df = train_df[(train_df.world != 'NONE')]

test_df = test_df[(test_df.world != 'NONE')]

print('9. Del world NONE')





#### 10. get game_time_diff

for df in [train_df, test_df]:

    df['game_time_diff'] = ((df.groupby(['installation_id', 'world', 'Assessment_id', 'name_assessment', 'type'])['game_time'].diff().fillna(0))/1000).astype('float32')

    df['game_time_diff'] = np.where(

        df['event_count'] == 1

        , 0

        , df['game_time_diff']

    )

print('10. get game_time_diff')



#### 11. top7_event_code

top7_event_code = list(train_df['event_code'].value_counts(normalize=True)[:5].index)

features_event_code = []

for e in top7_event_code:

    train_df['event_code_'+str(e)] = np.where(train_df['event_code']==e, 1, 0)

    test_df['event_code_'+str(e)] = np.where(test_df['event_code']==e, 1, 0)

    features_event_code.append('event_code_'+str(e))

print('11. top7_event_code')



#### 12. assessment stats on event code

train_df['assessment_event_count'] = np.where(train_df['assessment_code']==1, train_df['event_count'], 0)

test_df['assessment_event_count'] = np.where(test_df['assessment_code']==1, test_df['event_count'], 0)

train_df['assessment_game_time'] = np.where(train_df['assessment_code']==1, train_df['game_time'], 0)

test_df['assessment_game_time'] = np.where(test_df['assessment_code']==1, test_df['game_time'], 0)



# if PARAMS_CALC['delete from test/train assessment with correct&uncorrect == 0']:    

# in the test, there are many cases when there was a test (maybe just a person came in and left), for such cases, the score is 0, so we delete it from the sample, re-sort it

# and mark up the Assessment_id again



# # get stats per dimensions

# potencial_train = train_df.groupby(['installation_id', 'world', 'name_assessment', 'Assessment_id'])['correct', 'uncorrect', 'Predict_rows'].sum().reset_index()

# potencial_test = test_df.groupby(['installation_id', 'world', 'name_assessment', 'Assessment_id'])['correct', 'uncorrect', 'Predict_rows'].sum().reset_index()



# # search bad example with 0 correct and 0 uncorrect actions in assessment

# potencial_train['is_drop_assessment_id'] = np.where(

#     ((potencial_train['correct'].fillna(0)+potencial_train['uncorrect'].fillna(0))==0) & (potencial_train['name_assessment']!='') & (potencial_train['Predict_rows']==0)

#     , 1

#     , 0

# )

# potencial_test['is_drop_assessment_id'] = np.where(

#     ((potencial_test['correct'].fillna(0)+potencial_test['uncorrect'].fillna(0))==0) & (potencial_test['name_assessment']!='') & (potencial_test['Predict_rows']==0)

#     , 1

#     , 0

# )



# # join bad example

# train_df = train_df.merge(

#     potencial_train.drop(['correct', 'uncorrect', 'Predict_rows'], axis=1),

#     how='left'

# )

# test_df = test_df.merge(

#     potencial_test.drop(['correct', 'uncorrect', 'Predict_rows'], axis=1),

#     how='left'

# )



# # drop bad example

# train_df = train_df[train_df['is_drop_assessment_id']==0].copy()

# train_df.drop('is_drop_assessment_id', axis=1, inplace=True)

# test_df = test_df[test_df['is_drop_assessment_id']==0].copy()

# test_df.drop('is_drop_assessment_id', axis=1, inplace=True)



# # del potencial_train, potencial_test;

# # gc.collect();



# # update Assessment_id before drop

# train_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# test_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# train_df = get_assessmentid(train_df)

# test_df = get_assessmentid(test_df)

# gc.collect();





# train_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# test_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# train_df = pd.concat(

#     [

#         train_df,

#         train_df.groupby(

#             ['installation_id', 'world', 'name_assessment', 'Assessment_id']

#         )['correct', 'uncorrect', 'Predict_rows'].transform('sum').rename(

#             columns={

#                 'correct':'correct_iwna',

#                 'uncorrect':'uncorrect_iwna',

#                 'Predict_rows':'Predict_rows_iwna'

#             }

#         )

#     ],

#     axis=1,

#     sort=False

# )



# test_df = pd.concat(

#     [

#         test_df,

#         test_df.groupby(

#             ['installation_id', 'world', 'name_assessment', 'Assessment_id']

#         )['correct', 'uncorrect', 'Predict_rows'].transform('sum').rename(

#             columns={

#                 'correct':'correct_iwna',

#                 'uncorrect':'uncorrect_iwna',

#                 'Predict_rows':'Predict_rows_iwna'

#             }

#         )

#     ],

#     axis=1,

#     sort=False

# )



# train_df['is_drop_assessment_id'] = np.where(

#     ((train_df['correct_iwna'].fillna(0)+train_df['uncorrect_iwna'].fillna(0))==0) & (train_df['name_assessment']!='') & (train_df['Predict_rows_iwna']==0)

#     , 1

#     , 0

# )

# test_df['is_drop_assessment_id'] = np.where(

#     ((test_df['correct_iwna'].fillna(0)+test_df['uncorrect_iwna'].fillna(0))==0) & (test_df['name_assessment']!='') & (test_df['Predict_rows_iwna']==0)

#     , 1

#     , 0

# )



# train_df.drop(train_df[train_df['is_drop_assessment_id']==1].index, inplace=True)

# train_df.drop('is_drop_assessment_id', axis=1, inplace=True)

# test_df.drop(test_df[test_df['is_drop_assessment_id']==1].index, inplace=True)

# test_df.drop('is_drop_assessment_id', axis=1, inplace=True)



# # update Assessment_id before drop

# train_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# test_df.sort_values(['installation_id', 'timestamp'], inplace=True)

# train_df = get_assessmentid(train_df)

# test_df = get_assessmentid(test_df)

# gc.collect();



# print('11. delete from test/train assessment with correct&uncorrect == 0')



print('Ready')
%%time

#### Create train/test df partition by dimension key and aggregate



# just copy for agg min time

train_df['timestamp_min'] = train_df['timestamp']

test_df['timestamp_min'] = test_df['timestamp']



types = ['IsActivity', 'IsAssessment', 'IsClip', 'IsGame']

agg_features = {

    f:'sum' for f in ['IsActivityUniqueAction', 'IsAssessmentUniqueAction', 'IsClipUniqueAction', 'IsGameUniqueAction']

}



agg_features.update(

    {

        t:'sum' for t in types

    }

)



agg_features.update(

    {

        'title':'nunique',

        'correct':'sum', 'uncorrect':'sum', 

        'game_time_diff' : 'mean', 'Predict_rows':'max',

        'timestamp_min':'min', 'timestamp' : 'max',

        'assessment_event_count':'max',

        'assessment_game_time':'max',

    }

)

agg_features.update({f:'sum' for f in features_event_code})



train = train_df.groupby(dimension_key).agg(agg_features).reset_index().rename(columns = {f:f+'CountAction' for f in types})

train.rename(columns = {'game_time_diff':'mean_gametime_all'}, inplace=True)

test = test_df.groupby(dimension_key).agg(agg_features).reset_index().rename(columns = {f:f+'CountAction' for f in types}).rename(columns = {'game_time_diff':'mean_gametime_all'})

test.rename(columns = {'game_time_diff':'mean_gametime_all'}, inplace=True)



train['FullTimeSession'] = ((train['timestamp'] - train['timestamp_min']).fillna(0).astype('int64') / int(1e9)).astype('float32')

test['FullTimeSession'] = ((test['timestamp'] - test['timestamp_min']).fillna(0).astype('int64') / int(1e9)).astype('float32')



train.sort_values(['installation_id', 'timestamp'], inplace=True)

train.reset_index(drop=True, inplace=True)

test.sort_values(['installation_id', 'timestamp'], inplace=True)

test.reset_index(drop=True, inplace=True)



gc.collect();



# add mean correct & uncorrect all users stats

df = pd.concat(

    [

        train.query('(name_assessment != "") & (Predict_rows == 0)'),

        test.query('(name_assessment != "") & (Predict_rows == 0)')

    ], 

    axis=0,

    sort=False

).groupby(

    [

        'world', 'name_assessment', 'Assessment_id'

    ]

)['correct', 'uncorrect'].mean().reset_index().rename(

    columns={

        'correct':'correct_mean_all_users', 'uncorrect':'uncorrect_mean_all_users'

    }

)



train = train.merge(

    df

    , how='left'

)

test = test.merge(

    df

    , how='left'

)



del df;

gc.collect();





#### create time-features with cyclical encode

from tqdm import tqdm_notebook

def get_cyclical_encode(

    df,

    cols_maxval = {},

    is_drop = False

):

    df = df.copy()

    for col in tqdm_notebook(cols_maxval.keys()):

        print('Start ', col)

        df[col + '_sin'] = (np.sin(2 * np.pi * df[col]/cols_maxval[col])).astype('float16')

        df[col + '_cos'] = (np.cos(2 * np.pi * df[col]/cols_maxval[col])).astype('float16')

        print('Add', col + '_sin',col + '_cos')



        if is_drop:

            # drop non-cycle features

            df.drop(col, axis=1, inplace=True)

            print('Drop in features')

    return df





for df in [train, test]:

    # df['Month'] = df['timestamp'].dt.month.astype("uint8")

    # df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

    # df['is_year_start'] = df['timestamp'].dt.is_year_start.astype("uint8")

    # df['is_year_end'] = df['timestamp'].dt.is_year_end.astype("uint8")

    # df['weekofyear'] = df['timestamp'].dt.weekofyear.astype("uint8")

    # df['is_month_end'] = df['timestamp'].dt.is_month_end.astype("uint8")

    # df['is_month_start'] = df['timestamp'].dt.is_month_start.astype("uint8")

    # df['dayofyear'] = df['timestamp'].dt.dayofyear.astype("uint16")



# Not enough memory for commit, so commented out a small number of time features.

cols_maxval = train[['DayOfWeek', 'Hour']].nunique().to_dict()

train = get_cyclical_encode(train, cols_maxval, is_drop = True)

test = get_cyclical_encode(test, cols_maxval, is_drop = True)



gc.collect();





#### CUMSUM per user, actions & correct/uncorrect

agg_features = {

    f:'cumsum' for f in [

        'IsActivityUniqueAction', 'IsAssessmentUniqueAction',

        'IsClipUniqueAction', 'IsGameUniqueAction', 'IsActivityCountAction',

        'IsAssessmentCountAction', 'IsClipCountAction', 'IsGameCountAction',

        'assessment_game_time', 'title', 'correct', 'uncorrect'

    ] + features_event_code

}



train = pd.concat([

    train,

    train.groupby(['installation_id', 'world', 'name_assessment']).agg(agg_features).rename(

        columns={f:f+'_cumsum' for f in train.columns}

    )

], axis=1, sort=False)

train = pd.concat([

    train,

    train.groupby(['installation_id']).agg({'correct':'cumsum', 'uncorrect':'cumsum'}).rename(

        columns={f:f+'_user_cumsum' for f in train.columns}

    )

], axis=1, sort=False)

test = pd.concat([

    test,

    test.groupby(['installation_id', 'world', 'name_assessment']).agg(agg_features).rename(

        columns={f:f+'_cumsum' for f in test.columns}

    )

], axis=1, sort=False)

test = pd.concat([

    test,

    test.groupby(['installation_id']).agg({'correct':'cumsum', 'uncorrect':'cumsum'}).rename(

        columns={f:f+'_user_cumsum' for f in test.columns}

    )

], axis=1,sort=False)

gc.collect();





#### feature generation per TYPE in Action

def cols_name_mindex(multiindex):

    map_cols=[]

    for i in multiindex:

        x = i[0]

        for a in range(1, len(i)):

            x += '_' + str(i[a])

        map_cols.append(x)

    return map_cols

def feature_generation(df_agg, df):



    df_agg = df_agg.merge(

        df.groupby(

            ['installation_id', 'world', 'Assessment_id']

        )['game_time_diff'].sum().reset_index().rename(columns={'game_time_diff':'sum_gametime_session'})

    )



    prepare_df_agg = pd.pivot_table(

        df.rename(columns={'game_time_diff':'sum_gametime'}),

        aggfunc=np.sum,

        index = ['installation_id', 'world', 'Assessment_id'],

        columns = 'type',

        values = ['sum_gametime']

    )

    prepare_df_agg.columns = cols_name_mindex(prepare_df_agg.columns)

    prepare_df_agg.reset_index(inplace=True)

    df_agg = df_agg.merge(prepare_df_agg)

    return df_agg



%time train = feature_generation(train, train_df)

%time test = feature_generation(test, test_df)

gc.collect();





#### Drop not need columns

drop_cols = ['accuracy_calc', 'sum_gametime_Assessment', 'sum_gametime_Clip']

drop_cols = list(set(drop_cols) - (set(drop_cols) - set(train.columns)))

train.drop(drop_cols, axis=1,inplace=True)

test.drop(drop_cols, axis=1,inplace=True)

gc.collect();





#### Delete assessment with many assessment per worlds

train = train.merge(

    test[

        test['name_assessment']!=''

    ].groupby(

        ['world', 'name_assessment']

    )['Assessment_id'].max().reset_index().rename(columns={'Assessment_id':'TestCountAction'})

    , how = 'left'

)



train['TestCountAction'].fillna(12, inplace=True)

train = train[train['Assessment_id']<=train['TestCountAction']+1].reset_index(drop=True)

train.drop(['TestCountAction'], axis=1, inplace=True)





#### set target

for df in [train, test]:

    

    df['accuracy_calc'] = (df['correct']/(df['correct']+df['uncorrect']))# .fillna(0)

    

    # accuracy group

    df['accuracy_group'] = np.where(

        

        df['accuracy_calc'].isnull()==True,

        np.nan, 

        np.where(

            df['accuracy_calc']==0,

            0,

            np.where(

                df['accuracy_calc']==1,

                3,

                np.where(

                    df['accuracy_calc']>=0.5,

                    2,

                    1

                )

            )

        )

    )

target = 'accuracy_group'



#### add cumulative accuracy calc

train['cumulative_accuracy_calc'] = train['correct_cumsum'] / (train['correct_cumsum']+train['uncorrect_cumsum'])

test['cumulative_accuracy_calc'] = test['correct_cumsum'] / (test['correct_cumsum']+test['uncorrect_cumsum'])





#### with cumsum dummies accuracy

########### TRAIN



# columnsaccuracy

train[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

] = pd.get_dummies(train['accuracy_group'])

train[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_i' for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

] = train.groupby(['installation_id'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()

train[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_iw' for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

] = train.groupby(['installation_id', 'world'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()

train[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_iwn' for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

] = train.groupby(['installation_id', 'world', 'name_assessment'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(train[train.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()

########### TEST

test[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

] = pd.get_dummies(test['accuracy_group'])

test[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_i' for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

] = test.groupby(['installation_id'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()

test[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_iw' for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

] = test.groupby(['installation_id', 'world'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()

test[

    ['accuracy_group_'+format(i, '.0f')+'_cumsum_iwn' for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

] = test.groupby(['installation_id', 'world', 'name_assessment'])[

    ['accuracy_group_'+format(i, '.0f') for i in sorted(test[test.accuracy_group.isnull()==False].accuracy_group.unique())]

].cumsum()
dum_features = []

for column in ['world', 'name_assessment']:

    dummies = pd.get_dummies(test[column])

    test[[column+'_'+i for i in dummies.columns]] = dummies



    dummies = pd.get_dummies(train[column])

    train[[column+'_'+i for i in dummies.columns]] = dummies  



    [dum_features.append(column+'_'+f) for f in dummies.columns if f != '']



try:

    train.drop(['name_assessment_'], axis=1, inplace=True)

    test.drop(['name_assessment_'], axis=1, inplace=True)

except:

    pass



del dummies;

gc.collect();
train.columns
shift_features = [

    'correct_mean_all_users',

    'uncorrect_mean_all_users',

    'IsActivityUniqueAction','IsAssessmentUniqueAction', 'IsClipUniqueAction', 'IsGameUniqueAction',

    'IsActivityCountAction','IsAssessmentCountAction', 'IsClipCountAction', 'IsGameCountAction',

    'mean_gametime_all',

    'IsActivityUniqueAction_cumsum', 'IsAssessmentUniqueAction_cumsum',

    'IsClipUniqueAction_cumsum', 'IsGameUniqueAction_cumsum',

    'IsActivityCountAction_cumsum', 'IsAssessmentCountAction_cumsum',

    'IsClipCountAction_cumsum', 'IsGameCountAction_cumsum',

    'correct_cumsum', 'uncorrect_cumsum', 

    'correct_user_cumsum',

    'uncorrect_user_cumsum',

    'sum_gametime_session',

    'sum_gametime_Activity',

    'sum_gametime_Game',

    'FullTimeSession', 

    'assessment_event_count', 'assessment_game_time', 'assessment_game_time_cumsum', 'title_cumsum',



    'accuracy_calc', 'cumulative_accuracy_calc',

    'accuracy_group_0',

    'accuracy_group_1', 'accuracy_group_2', 'accuracy_group_3',

    'accuracy_group_0_cumsum_i', 'accuracy_group_1_cumsum_i',

    'accuracy_group_2_cumsum_i', 'accuracy_group_3_cumsum_i',

    'accuracy_group_0_cumsum_iw', 'accuracy_group_1_cumsum_iw',

    'accuracy_group_2_cumsum_iw', 'accuracy_group_3_cumsum_iw',

    'accuracy_group_0_cumsum_iwn', 'accuracy_group_1_cumsum_iwn',

    'accuracy_group_2_cumsum_iwn', 'accuracy_group_3_cumsum_iwn'

] + features_event_code + [f+'_cumsum' for f in features_event_code]



all_shift_features = []



train['is_train_df'] = 1

test['is_train_df'] = 0



df = pd.concat(

    [

        train,

        test

    ], 

    axis=0,

    sort=False

)



df.sort_values(['installation_id', 'timestamp'], inplace=True)



for sf in tqdm_notebook(shift_features):

    # print(sf)

    # last assessment per installation, world, name_assessment

    df[sf+'_shift'] = df.groupby(

        ['installation_id', 'world', 'name_assessment']

    )[sf].shift(1).astype('float32')

    all_shift_features.append(sf+'_shift')

    

    # shift per world, installation (without 'name_assessment')

    df[sf+'_shift_iw'] = df.groupby(

        ['installation_id', 'world']

    )[sf].shift(1).astype('float32')

    all_shift_features.append(sf+'_shift_iw')

    

    # mean shift per world, name_assessment, assessment_id

    df = df.merge(

        df.groupby(

            ['world', 'name_assessment', 'Assessment_id']

        )[sf+'_shift'].mean().astype('float32').reset_index().rename(columns={sf+'_shift':sf+'_shift_mean_wnA'}),

        how='left'

    )

    all_shift_features.append(sf+'_shift_mean_wnA')

    

    # mean shift_IW (installation, world) last action per world, name_assessment, assessment_id

    df = df.merge(

        df.groupby(

            ['world', 'name_assessment', 'Assessment_id']

        )[sf+'_shift_iw'].mean().astype('float32').reset_index().rename(columns={sf+'_shift_iw':sf+'_shift_iw_mean_wnA'}),

        how='left'

    )

    all_shift_features.append(sf+'_shift_iw_mean_wnA')

    

    

    # mean shift per world, name_assessment (without assessment_id)

    df = df.merge(

        df.groupby(

            ['world', 'Assessment_id']

        )[sf+'_shift'].mean().astype('float32').reset_index().rename(columns={sf+'_shift':sf+'_shift_mean_wA'}),

        how='left'

    )

    all_shift_features.append(sf+'_shift_mean_wA')



    df = df.merge(

        df.groupby(

            ['world', 'Assessment_id']

        )[sf+'_shift_iw'].mean().astype('float32').reset_index().rename(columns={sf+'_shift_iw':sf+'_shift_iw_mean_wA'}),

        how='left'

    )

    all_shift_features.append(sf+'_shift_iw_mean_wA')



    

#     print(df.shape)

#     # expanding mean shift per world, name_assessment (without assessment_id)

#     df = df.merge(

#         df.groupby(

#             ['installation_id', 'world']

#         )[sf+'_shift'].expanding(

#         ).mean(

#         ).astype('float32').reset_index().rename(

#             columns={sf+'_shift':sf+'_shift_ExpandingMean_wi'}

#         )[['installation_id', 'world',sf+'_shift_ExpandingMean_wi']],

#         how='left'

#     )

    



# # func for shift variables

agg_shift_features = []

# func_ag

# for f_agg in tqdm_notebook(func_agg):

funcs = [



    

    # standard deviation for all people

    [['world', 'Assessment_id', 'name_assessment'], 'std'],

    # standard deviation for all people and competitions

    [['world', 'Assessment_id'], 'std'],

    

    # standard deviation and mean per user and competitions

    # [['installation_id', 'world', 'Assessment_id'], 'mean'],

    [['installation_id', 'world', 'Assessment_id'], 'std'],

    [['installation_id', 'name_assessment', 'Assessment_id'], 'mean'],

    # [['installation_id', 'name_assessment', 'Assessment_id'], 'std'],

    

    # person / attempt

    [['installation_id', 'Assessment_id'], 'mean'],

    # [['installation_id', 'Assessment_id'], 'cumsum'],

    # [['installation_id', 'Assessment_id'], 'std'],

    

    # average for such tests

    [['name_assessment', 'Assessment_id'], 'mean'],

    [['Assessment_id'], 'mean'],

]



for dim, f_agg in tqdm_notebook(funcs):

    for sh in all_shift_features:

        name_concat = ''.join([d[:1] for d in dim])

        # print(f_agg+'_'+sh+'_'+name_concat)

        df[f_agg+'_'+sh+'_'+name_concat] = df.groupby(

            dim

        )[sh].transform(f_agg).astype('float32')



        # add new features in list

        agg_shift_features.append(f_agg+'_'+sh+'_'+name_concat)

        

train = df[df['is_train_df'] == 1].drop('is_train_df', axis=1).copy()

test = df[df['is_train_df'] == 0].drop('is_train_df', axis=1).copy()



del df;

gc.collect();
from sklearn.cluster import KMeans

N_CLUSTER = 5



user_unique_stats = pd.concat(

    [

        train,

        test

    ],

    axis=0    

).groupby(['installation_id']).agg(

    {

        'IsActivityUniqueAction' : 'count',

        'world':'nunique',

        'name_assessment':'nunique',

        'Assessment_id':'max'

    }

).reset_index().rename(columns = {'IsActivityUniqueAction':'CountAssessment'})





# .to_excel('smart_split_to_5_group.xlsx', index=None)

# .sort_values('CountAssessment', ascending=False)



kmeans = KMeans(n_clusters=N_CLUSTER, random_state=1).fit(user_unique_stats.drop('installation_id', axis=1))

user_unique_stats['Cluster'] = kmeans.labels_



user_unique_stats.sort_values('Cluster', inplace=True)

user_unique_stats['NumInst_per_cluster'] = user_unique_stats.groupby('Cluster').cumcount()



user_unique_stats = user_unique_stats.merge(

    user_unique_stats.groupby('Cluster').agg(

        {

            'installation_id':'count'

        }

    ).reset_index().rename(

        columns = {

            'installation_id':'inst_count'

        }

    )

)



user_unique_stats['RoundGroup'] = user_unique_stats['inst_count'] * (1./n_fold)

user_unique_stats['KFold_Group'] = (user_unique_stats['NumInst_per_cluster'] // user_unique_stats['RoundGroup'])

# user_unique_stats.to_excel('smart_split_to_5_group.xlsx', index=None)



train = train.merge(user_unique_stats[['installation_id', 'KFold_Group']], how='left')

test = test.merge(user_unique_stats[['installation_id', 'KFold_Group']], how='left')



user_unique_stats.groupby('Cluster').agg(

    {

        'installation_id':'count',

        'CountAssessment':'mean',

        'world':'mean',

        'name_assessment':'mean',

        'Assessment_id':'mean',

        # 'Assessment_id':'std',

    }

)
user_unique_stats.groupby(['KFold_Group', 'Cluster'])['installation_id'].count()
train = train[(train.accuracy_group.isnull()==False)]

train = pd.concat(

    [

        train,

        test[(test.accuracy_group.isnull()==False)]

    ],

    axis=0,

    sort=False

)

train[target] = train[target].astype('uint8')



test = test[(test['Predict_rows']==1)]

test.reset_index(drop=True, inplace=True)

train.reset_index(drop=True, inplace=True)
display(train[train['installation_id'] == '00abaee7'])

display(test[test['installation_id'] == '00abaee7'])
if KAGGLE_START:

    del train_df, test_df;

    gc.collect();
features = dum_features + [

    'Assessment_id',

    'DayOfWeek_sin', 'DayOfWeek_cos', 'Hour_sin',

    'Hour_cos',

    'correct_mean_all_users',

    'uncorrect_mean_all_users',

    'cumulative_accuracy_calc_shift',

] + all_shift_features + agg_shift_features
len(features)
%%time

def drop_corr_features(df, features, perc = 0.95, is_print = False):

    def set_to_list(cols, excepted):

        return list(set(cols) - set(excepted))

    # Identify Highly Correlated Features

    # Create correlation matrix

    corr_matrix = df[features].corr().abs()

    # Select upper triangle of correlation matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95/perc var

    to_drop_corr_feat = [column for column in upper.columns if any(upper[column] > perc)]

    if is_print:

        print(', '.join(to_drop_corr_feat))

    # Drop Marked Features

    # df.drop(to_drop_corr_feat, axis=1, inplace = True)

    features = set_to_list(features, to_drop_corr_feat)

    return to_drop_corr_feat



to_drop_corr_feat = drop_corr_features(pd.concat([train, test], axis=0, sort=False), features)

features = list(set(features) - set(to_drop_corr_feat))



# to_drop_corr_feat = drop_corr_features(test, features)

# features = list(set(features) - set(to_drop_corr_feat))

print('del', len(to_drop_corr_feat), 'correlation features')
@jit

def data_skew(df):

    

    sk_df = pd.DataFrame(

        [

            {

                'column': c,

                'uniq': df[c].nunique(),

                'skewness': df[c].value_counts(normalize=True).values[0] * 100

            } for c in df.columns

        ]

    )

    sk_df = sk_df.sort_values('skewness', ascending=False)

    return sk_df





pd.options.display.float_format = '{:,.3f}'.format

df_skew = data_skew(train)

# display(df_skew.head(20))



skewness_columns = list(df_skew[df_skew['skewness']>95.0]['column'].to_numpy())

features = list(set(features) - set(skewness_columns))



print('del', len(skewness_columns), 'skewness columns')
null_features_in_test = test[features].sum().reset_index().rename(columns={0:'sum_f'}).query('sum_f==0')['index'].values

features = list(set(features) - set(null_features_in_test))



print('del', len(null_features_in_test), 'null features in test')
# Outlier detection

def detect_outliers(

    df

    , n

    , features

):



    df = df.copy()



    outlier_indices = []

    from collections import Counter

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1



        # outlier step

        outlier_step = 1.5 * IQR



        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index



        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)



    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )



    return multiple_outliers





outliers = detect_outliers(df = train, n = 10, features = features)

len(outliers)

# train.drop(outliers, inplace=True)

# train.reset_index(drop=True, inplace=True)
train.shape
len(features)
%%time



y = train['accuracy_group']

cols_to_drop = list(set(train.columns)-set(features))



if BAYES_OPTIMIZATION == False:

    import warnings

    # warnings.simplefilter('ignore')

    warnings.filterwarnings("ignore")



    params = {

        'n_estimators':2000,

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'metric': 'rmse',

        'subsample': 0.75,

        'subsample_freq': 1,

        'learning_rate': 0.04,

        'feature_fraction': 0.75,

        'max_depth': 32,

        'lambda_l1': 0.75,  

        'lambda_l2': 0.75,

        'verbose': 100,

        'early_stopping_rounds': 100,

        'eval_metric': 'cappa'

    }



    mt = MainTransformer()

    ft = FeatureTransformer()

    transformers = {'ft': ft}

    regressor_model1 = RegressorModel(model_wrapper=LGBWrapper_regr())

    regressor_model1.fit(

        X=train,

        y=y,

        folds=folds,

        params=params,

        preprocesser=mt,

        transformers=transformers,

        eval_metric='cappa',

        cols_to_drop=cols_to_drop

    )





    # %%time

    pr1 = regressor_model1.predict(train)

    optR = OptimizedRounder()

    optR.fit(pr1.reshape(-1,), y)

    coefficients = optR.coefficients()



    opt_preds = optR.predict(pr1.reshape(-1, ), coefficients)

    print('val qwk:', qwk(y, opt_preds))



    # some coefficients calculated by me.

    pr1 = regressor_model1.predict(test)

    # pr1[pr1 <= 1.12232214] = 0

    # pr1[np.where(np.logical_and(pr1 > 1.12232214, pr1 <= 1.73925866))] = 1

    # pr1[np.where(np.logical_and(pr1 > 1.73925866, pr1 <= 2.22506454))] = 2

    # pr1[pr1 > 2.22506454] = 3



    print(', '.join(['Coeff '+str(i)+': ' + format(c, '.2f') for i, c in enumerate(coefficients)]))



    pr1[pr1 <= coefficients[0]] = 0

    pr1[np.where(np.logical_and(pr1 > coefficients[0], pr1 <= coefficients[1]))] = 1

    pr1[np.where(np.logical_and(pr1 > coefficients[1], pr1 <= coefficients[2]))] = 2

    pr1[pr1 > coefficients[2]] = 3





    test['accuracy_group'] = pr1.astype(int)

    sample_submission_df = sample_submission_df.drop('accuracy_group', axis=1).merge(

        test[['installation_id', 'accuracy_group']]

    )

    sample_submission_df.to_csv('submission.csv', index=False)

    print('sample_submission_df')

    display(sample_submission_df['accuracy_group'].value_counts(normalize=True))

    print('train distrb')

    train['accuracy_group'].value_counts(normalize=True)

else:

    from bayes_opt import BayesianOptimization



    def LGB_bayesian(max_depth,

                     lambda_l1,

                     lambda_l2,

                     bagging_fraction,

                     bagging_freq,

                     colsample_bytree,

                     learning_rate):



        params = {

            'boosting_type': 'gbdt',

            'metric': 'rmse',

            'objective': 'regression',

            'eval_metric': 'cappa',

            'n_jobs': -1,

            'seed': 42,

            'early_stopping_rounds': 100,

            'n_estimators': 2000,

            'learning_rate': learning_rate,

            'max_depth': int(max_depth),

            'lambda_l1': lambda_l1,

            'lambda_l2': lambda_l2,

            'bagging_fraction': bagging_fraction,

            'bagging_freq': int(bagging_freq),

            'colsample_bytree': colsample_bytree,

            'verbose': 0

        }



        mt = MainTransformer()

        ft = FeatureTransformer()

        transformers = {'ft': ft}

        model = RegressorModel(model_wrapper=LGBWrapper_regr())

        model.fit(

            X=train,

            y=y,

            folds=folds,

            params=params,

            preprocesser=mt,

            transformers=transformers,

            eval_metric='cappa',

            cols_to_drop=cols_to_drop,

            plot=False

        )



        return model.scores['valid']

    

    # set params

    init_points = 16

    n_iter = 16

    bounds_LGB = {

        'max_depth': (8, 16),

        'lambda_l1': (0, 5),

        'lambda_l2': (0, 5),

        'bagging_fraction': (0.4, 0.6),

        'bagging_freq': (1, 10),

        'colsample_bytree': (0.4, 0.6),

        'learning_rate': (0.05, 0.1)

    }



    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1029)



    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)





    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'eval_metric': 'cappa',

        'n_jobs': -1,

        'seed': 42,

        'early_stopping_rounds': 100,

        'n_estimators': 2000,

        'learning_rate': LGB_BO.max['params']['learning_rate'],

        'max_depth': int(LGB_BO.max['params']['max_depth']),

        'lambda_l1': LGB_BO.max['params']['lambda_l1'],

        'lambda_l2': LGB_BO.max['params']['lambda_l2'],

        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],

        'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),

        'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],

        'verbose': 100

    }



    mt = MainTransformer()

    ft = FeatureTransformer()

    transformers = {'ft': ft}

    regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())

    regressor_model.fit(X=train, 

                        y=y, 

                        folds=folds, 

                        params=params, 

                        preprocesser=mt, 

                        transformers=transformers,

                        eval_metric='cappa', 

                        cols_to_drop=cols_to_drop)



    pr1 = regressor_model.predict(train)

    optR = OptimizedRounder()

    optR.fit(pr1.reshape(-1,), y)

    coefficients1 = optR.coefficients()



    preds_1 = regressor_model.predict(test)

    w_1 = LGB_BO.max['target']



    del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model

    gc.collect();





    bounds_LGB = {

        'max_depth': (11, 14),

        'lambda_l1': (0, 10),

        'lambda_l2': (0, 10),

        'bagging_fraction': (0.7, 1),

        'bagging_freq': (1, 10),

        'colsample_bytree': (0.7, 1),

        'learning_rate': (0.08, 0.2)

    }



    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1030)



    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)



    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'eval_metric': 'cappa',

        'n_jobs': -1,

        'seed': 42,

        'early_stopping_rounds': 100,

        'n_estimators': 2000,

        'learning_rate': LGB_BO.max['params']['learning_rate'],

        'max_depth': int(LGB_BO.max['params']['max_depth']),

        'lambda_l1': LGB_BO.max['params']['lambda_l1'],

        'lambda_l2': LGB_BO.max['params']['lambda_l2'],

        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],

        'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),

        'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],

        'verbose': 100

    }



    mt = MainTransformer()

    ft = FeatureTransformer()

    transformers = {'ft': ft}

    regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())

    regressor_model.fit(X=train, 

                        y=y, 

                        folds=folds, 

                        params=params, 

                        preprocesser=mt, 

                        transformers=transformers,

                        eval_metric='cappa', 

                        cols_to_drop=cols_to_drop)



    pr2 = regressor_model.predict(train)

    optR = OptimizedRounder()

    optR.fit(pr2.reshape(-1,), y)

    coefficients2 = optR.coefficients()



    preds_2 = regressor_model.predict(test)

    w_2 = LGB_BO.max['target']



    del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model

    gc.collect();



    preds = (w_1/(w_1+w_2)) * preds_1 + (w_2/(w_1+w_2)) * preds_2



    del preds_1, preds_2

    gc.collect();



    coefficients = np.mean([coefficients1, coefficients2], axis=0) # [1.12232214, 1.73925866, 2.22506454]

    print('Coefs: ', ', '.join(coefficients))

    preds[preds <= coefficients[0]] = 0

    preds[np.where(np.logical_and(preds > coefficients[0], preds <= coefficients[1]))] = 1

    preds[np.where(np.logical_and(preds > coefficients[1], preds <= coefficients[2]))] = 2

    preds[preds > coefficients[2]] = 3



    test['accuracy_group'] = preds.astype(int)

    sample_submission_df = sample_submission_df.drop('accuracy_group', axis=1).merge(

        test[['installation_id', 'accuracy_group']]

    )

    sample_submission_df.to_csv('submission.csv', index=False)

    print('sample_submission_df')

    display(sample_submission_df['accuracy_group'].value_counts(normalize=True))

    print('train distrb')

    train['accuracy_group'].value_counts(normalize=True)