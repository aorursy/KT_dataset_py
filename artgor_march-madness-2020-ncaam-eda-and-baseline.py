# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

%matplotlib inline

import copy

import datetime

import lightgbm as lgb

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report, confusion_matrix

import json

import ast

import time

from sklearn import linear_model



import warnings

warnings.filterwarnings('ignore')



import os

import glob



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder
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

                       eval_set=eval_set, eval_names=eval_names,

                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'])



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

            pass

        return self



    def transform(self, X, y=None):

        data = copy.deepcopy(X)



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



        self.num_cols = [col for col in X.columns if 'sum' in col or 'mean' in col or 'max' in col or 'std' in col

                         or 'attempt' in col]



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
class ClassifierModel(object):

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

            eval_metric='auc',

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

        self.cols_to_drop = cols_to_drop



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

        n_target = 1 if len(set(y.values)) == 2 else len(set(y.values))

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

            # y = X['accuracy_group']



        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

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

                self.oof[valid_index] = model.predict_proba(X_valid).reshape(-1, n_target)



            fold_importance = pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)),

                                           columns=['feature', 'importance'])

            self.feature_importances = self.feature_importances.append(fold_importance)

            self.models.append(model)



        self.feature_importances['importance'] = self.feature_importances['importance'].astype(float)



        # if params['verbose']:

        self.calc_scores_()



        if plot:

#             print(classification_report(y, self.oof.argmax(1)))

            print(classification_report(y, (self.oof > 0.5) * 1))

            fig, ax = plt.subplots(figsize=(16, 12))

            plt.subplot(2, 2, 1)

            self.plot_feature_importance(top_n=25)

            plt.subplot(2, 2, 2)

            self.plot_metric()

            plt.subplot(2, 2, 3)

            g = sns.heatmap(confusion_matrix(y, (self.oof > 0.5) * 1), annot=True, cmap=plt.cm.Blues,fmt="d")

            g.set(ylim=(-0.5, 4), xlim=(-0.5, 4), title='Confusion matrix')



            plt.subplot(2, 2, 4)

            plt.hist(self.oof)

            plt.xticks(range(self.n_target), range(self.n_target))

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

            self.cols_to_drop = cols_to_drop

            datasets['X_train'] = datasets['X_train'].drop(cols_to_drop, axis=1)

            if datasets['X_valid'] is not None:

                datasets['X_valid'] = datasets['X_valid'].drop(cols_to_drop, axis=1)

            if datasets['X_holdout'] is not None:

                datasets['X_holdout'] = datasets['X_holdout'].drop(cols_to_drop, axis=1)



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

            if self.cols_to_drop:

                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]

                X_t = X_t.drop(cols_to_drop, axis=1)

            y_pred = self.models[i].predict_proba(X_t[self.folds_dict[i]['columns']]).reshape(-1, full_prediction.shape[1])



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

        full_evals_results[self.eval_metric] = np.abs(full_evals_results[self.eval_metric])

        sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')

        plt.title('Training progress')
data_dict = {}

for i in glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/*'):

    name = i.split('/')[-1].split('.')[0]

    if name != 'MTeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='cp1252')
data_dict.keys()
data_dict['MNCAATourneySeeds'].head()
data_dict['MNCAATourneyCompactResults'].head()
data_dict['MNCAATourneyCompactResults'].groupby(['Season'])['WScore'].mean().plot(kind='line');

plt.title('Mean scores of winning teams by season in tourneys');
data_dict['MRegularSeasonCompactResults']
data_dict['MRegularSeasonCompactResults'].groupby(['Season'])['WScore'].mean().plot();

plt.title('Mean scores of winning teams by season in regular plays');
# process seed

data_dict['MNCAATourneySeeds'] = data_dict['MNCAATourneySeeds'].loc[data_dict['MNCAATourneySeeds']['Season'] <= 2014]

data_dict['MNCAATourneySeeds']['Seed'] = data_dict['MNCAATourneySeeds']['Seed'].apply(lambda x: int(x[1:3]))

# take only useful columns

data_dict['MNCAATourneySeeds'] = data_dict['MNCAATourneySeeds'][['Season', 'TeamID', 'Seed']]

data_dict['MNCAATourneyCompactResults'] = data_dict['MNCAATourneyCompactResults'][['Season','WTeamID', 'LTeamID']]

data_dict['MNCAATourneyCompactResults'] = data_dict['MNCAATourneyCompactResults'].loc[data_dict['MNCAATourneyCompactResults']['Season'] <= 2014]

# merge the data and rename the columns

df = pd.merge(data_dict['MNCAATourneyCompactResults'], data_dict['MNCAATourneySeeds'],

              how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

df = pd.merge(df, data_dict['MNCAATourneySeeds'], how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])

df = df.drop(['TeamID_x', 'TeamID_y'], axis=1)

df.columns = ['Season', 'WTeamID', 'LTeamID', 'WSeed', 'LSeed']

df.head()
df.head()
team_win_score = data_dict['MRegularSeasonCompactResults'].groupby(['Season', 'WTeamID']).agg({'WScore':['sum', 'count']}).reset_index()

team_win_score.columns = ['Season', 'WTeamID', 'WScore_sum', 'WScore_count']

team_loss_score = data_dict['MRegularSeasonCompactResults'].groupby(['Season', 'LTeamID']).agg({'LScore':['sum', 'count']}).reset_index()

team_loss_score.columns = ['Season', 'LTeamID', 'LScore_sum', 'LScore_count']

df = pd.merge(df, team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])

df = pd.merge(df, team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])

df = pd.merge(df, team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])

df = pd.merge(df, team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])

df.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

df.head()
df['x_score'] = df['WScore_sum_x'] + df['LScore_sum_y']

df['y_score'] = df['WScore_sum_y'] + df['LScore_sum_x']

df['x_count'] = df['WScore_count_x'] + df['LScore_count_y']

df['y_count'] = df['WScore_count_y'] + df['LScore_count_x']
df.head()
df_win = df.copy()

df_los = df.copy()

df_win = df_win[['WSeed', 'LSeed', 'x_score', 'y_score', 'x_count', 'y_count']]

df_los = df_los[['LSeed', 'WSeed', 'y_score', 'x_score', 'y_count', 'x_count']]

df_win.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']

df_los.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']
df_win['Seed_diff'] = df_win['Seed_1'] - df_win['Seed_2']

df_win['Score_diff'] = df_win['Score_1'] - df_win['Score_2']

df_los['Seed_diff'] = df_los['Seed_1'] - df_los['Seed_2']

df_los['Score_diff'] = df_los['Score_1'] - df_los['Score_2']



df_win['Count_diff'] = df_win['Count_1'] - df_win['Count_2']

df_win['Mean_score1'] = df_win['Score_1'] / df_win['Count_1']

df_win['Mean_score2'] = df_win['Score_2'] / df_win['Count_2']

df_win['Mean_score_diff'] = df_win['Mean_score1'] - df_win['Mean_score2']

df_los['Count_diff'] = df_los['Count_1'] - df_los['Count_2']

df_los['Mean_score1'] = df_los['Score_1'] / df_los['Count_1']

df_los['Mean_score2'] = df_los['Score_2'] / df_los['Count_2']

df_los['Mean_score_diff'] = df_los['Mean_score1'] - df_los['Mean_score2']
df_win['result'] = 1

df_los['result'] = 0

data = pd.concat((df_win, df_los)).reset_index(drop=True)
for col in ['Score_1', 'Score_2', 'Count_1', 'Count_2', 'Score_diff', 'Count_diff']:

    print(col)

    data[col] = data[col].fillna(0).astype(int)
data.head()
n_fold = 5

folds = RepeatedStratifiedKFold(n_splits=n_fold)

# folds = StratifiedKFold(n_splits=n_fold)
X = data.drop(['result'], axis=1)

y = data['result']
# some of params are from this kernel: https://www.kaggle.com/ratan123/march-madness-2020-ncaam-simple-lightgbm-on-kfold

param = {'n_estimators':10000,

          'num_leaves': 400,

          'min_child_weight': 0.034,

          'feature_fraction': 0.379,

          'bagging_fraction': 0.418,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.007,

          "boosting_type": "gbdt",

          #"bagging_seed": 11,

          "metric": 'binary_logloss',

          "verbosity": 10,

          'reg_alpha': 0.3899,

          'reg_lambda': 0.648,

          'random_state': 47,

          'task':'train', 'nthread':-1, 

         'verbose': 100,

         'early_stopping_rounds': 30,

         'eval_metric': 'binary_logloss'

         }

cat_cols = []

mt = MainTransformer(create_interactions=False)

# ct = CategoricalTransformer(drop_original=True, cat_cols=cat_cols)

ft = FeatureTransformer()

transformers = {'ft': ft}

lgb_model = ClassifierModel(model_wrapper=LGBWrapper())

lgb_model.fit(X=X, y=y, folds=folds, params=param, preprocesser=mt, transformers=transformers,

                    eval_metric='binary_logloss', cols_to_drop=None, plot=True)
test = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

test = test.drop(['Pred'], axis=1)

test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))

test['Team1'] = test['ID'].apply(lambda x: int(x.split('_')[1]))

test['Team2'] = test['ID'].apply(lambda x: int(x.split('_')[2]))

test = pd.merge(test, data_dict['MNCAATourneySeeds'], how='left', left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'])

test = pd.merge(test, data_dict['MNCAATourneySeeds'], how='left', left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'])

test = pd.merge(test, team_win_score, how='left', left_on=['Season', 'Team1'], right_on=['Season', 'WTeamID'])

test = pd.merge(test, team_loss_score, how='left', left_on=['Season', 'Team2'], right_on=['Season', 'LTeamID'])

test = pd.merge(test, team_loss_score, how='left', left_on=['Season', 'Team1'], right_on=['Season', 'LTeamID'])

test = pd.merge(test, team_win_score, how='left', left_on=['Season', 'Team2'], right_on=['Season', 'WTeamID'])

test['seed_diff'] = test['Seed_x'] - test['Seed_y']
test['x_score'] = test['WScore_sum_x'] + test['LScore_sum_y']

test['y_score'] = test['WScore_sum_y'] + test['LScore_sum_x']

test['x_count'] = test['WScore_count_x'] + test['LScore_count_y']

test['y_count'] = test['WScore_count_y'] + test['WScore_count_x']
test.head()
test = test[['Seed_x', 'Seed_y', 'x_score', 'y_score', 'x_count', 'y_count']]

test.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']
test['Seed_diff'] = test['Seed_1'] - test['Seed_2']

test['Score_diff'] = test['Score_1'] - test['Score_2']

test['Seed_diff'] = test['Seed_1'] - test['Seed_2']

test['Score_diff'] = test['Score_1'] - test['Score_2']



test['Count_diff'] = test['Count_1'] - test['Count_2']

test['Mean_score1'] = test['Score_1'] / test['Count_1']

test['Mean_score2'] = test['Score_2'] / test['Count_2']

test['Mean_score_diff'] = test['Mean_score1'] - test['Mean_score2']

test['Count_diff'] = test['Count_1'] - test['Count_2']

test['Mean_score1'] = test['Score_1'] / test['Count_1']

test['Mean_score2'] = test['Score_2'] / test['Count_2']

test['Mean_score_diff'] = test['Mean_score1'] - test['Mean_score2']
test.head()
test_preds = lgb_model.predict(test)
plt.hist(test_preds);
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = test_preds

submission_df
submission_df.to_csv('submission.csv', index=False)