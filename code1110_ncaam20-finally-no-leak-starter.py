# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

%matplotlib inline

import copy

import datetime

from sklearn.utils import shuffle

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit

from sklearn.preprocessing import StandardScaler, LabelEncoder

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostRegressor, CatBoostClassifier

import optuna

from optuna.visualization import plot_optimization_history

from sklearn import model_selection

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, log_loss, classification_report, confusion_matrix

import json

import ast

import time

from sklearn import linear_model



# keras

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LambdaCallback

from keras.optimizers import Adam, SGD

from keras.models import Model

from keras.layers import Input, Layer, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization, GaussianNoise

from keras.layers.embeddings import Embedding

from keras import backend as K

from keras.layers import Layer

from keras.callbacks import *

import tensorflow as tf

import math



import warnings

warnings.filterwarnings('ignore')



import os

import glob

import gc



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder



print("Libraries imported!")
import numpy as np

import pandas as pd

import os, sys

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error

from sklearn import linear_model



class RunModel(object):

    """

    Model Fitting and Prediction Class:



    train_df : train pandas dataframe

    test_df : test pandas dataframe

    target : target column name (str)

    features : list of feature names

    categoricals : list of categorical feature names

    model : lgb, xgb, catb, linear, or nn

    task : options are ... regression, multiclass, or binary

    n_splits : K in KFold (default is 3)

    cv_method : options are ... KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, StratifiedGroupKFold

    group : group feature name when GroupKFold or StratifiedGroupKFold are used

    parameter_tuning : bool, only for LGB

    seed : seed (int)

    scaler : options are ... None, MinMax, Standard

    verbose : bool

    """



    def __init__(self, train_df, test_df, target, features, categoricals=[],

                model="linear", task="regression", n_splits=3, cv_method="KFold", 

                group=None, parameter_tuning=False, seed=1220, scaler=None, verbose=True):

        self.train_df = train_df

        self.test_df = test_df

        self.target = target

        self.features = features

        self.categoricals = categoricals

        self.model = model

        self.task = task

        self.n_splits = n_splits

        self.cv_method = cv_method

        self.group = group

        self.parameter_tuning = parameter_tuning

        self.seed = seed

        self.scaler = scaler

        self.verbose = verbose

        self.cv = self.get_cv()

        self.params = self.get_params()

        self.y_pred, self.score, self.model, self.oof, self.y_val, self.fi_df = self.fit()



    def train_model(self, train_set, val_set):

        # verbose

        verbosity = 1000 if self.verbose else 0



        if self.task == "regression":

            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

            model = linear_model.Ridge(**{'alpha': 220, 'solver': 'lsqr', 'fit_intercept': self.params['fit_intercept'],

                                    'max_iter': self.params['max_iter'], 'random_state': self.params['random_state']})

        elif (self.task == "binary") | (self.task == "multiclass"):

            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

            model = linear_model.LogisticRegression(**{"C": 1.0, "fit_intercept": self.params['fit_intercept'], 

                                    "random_state": self.params['random_state'], "solver": "lbfgs", "max_iter": self.params['max_iter'], 

                                    "multi_class": 'auto', "verbose":0, "warm_start":False})

        model.fit(train_set['X'], train_set['y'])



        # permutation importance to get a feature importance (off in default)

        # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)

        fi = np.zeros(len(self.features)) # no feature importance computed

        

        return model, fi # fitted model and feature importance



    def get_params(self):

        params = {

            'max_iter': 5000,

            'fit_intercept': True,

            'random_state': self.seed

        }

        return params



    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = {'X': x_train, 'y': y_train}

        val_set = {'X': x_val, 'y': y_val}

        return train_set, val_set



    def convert_x(self, x):

        return x



    def calc_metric(self, y_true, y_pred): # this may need to be changed based on the metric of interest

        if self.task == "multiclass":

            return log_loss(y_true, y_pred)

        elif self.task == "binary":

            return log_loss(y_true, y_pred)

        elif self.task == "regression":

            return np.sqrt(mean_squared_error(y_true, y_pred))



    def get_cv(self):

        if self.cv_method == "KFold":

            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

            return cv.split(self.train_df)

        elif self.cv_method == "StratifiedKFold":

            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

            return cv.split(self.train_df, self.train_df[self.target])

        elif self.cv_method == "TimeSeriesSplit":

            cv = TimeSeriesSplit(max_train_size=None, n_splits=self.n_splits)

            return cv.split(self.train_df)



    def fit(self):

        # initialize

        oof_pred = np.zeros((self.train_df.shape[0], ))

        y_vals = np.zeros((self.train_df.shape[0], ))

        y_pred = np.zeros((self.test_df.shape[0], ))

        if self.group is not None:

            if self.group in self.features:

                self.features.remove(self.group)

            if self.group in self.categoricals:

                self.categoricals.remove(self.group)

        fi = np.zeros((self.n_splits, len(self.features)))



        # scaling, if necessary

        if self.scaler is not None:

            # fill NaN

            numerical_features = [f for f in self.features if f not in self.categoricals]

            self.train_df[numerical_features] = self.train_df[numerical_features].fillna(self.train_df[numerical_features].median())

            self.test_df[numerical_features] = self.test_df[numerical_features].fillna(self.test_df[numerical_features].median())

            self.train_df[self.categoricals] = self.train_df[self.categoricals].fillna(self.train_df[self.categoricals].mode().iloc[0])

            self.test_df[self.categoricals] = self.test_df[self.categoricals].fillna(self.test_df[self.categoricals].mode().iloc[0])



            # scaling

            if self.scaler == "MinMax":

                scaler = MinMaxScaler()

            elif self.scaler == "Standard":

                scaler = StandardScaler()

            df = pd.concat([self.train_df[numerical_features], self.test_df[numerical_features]], ignore_index=True)

            scaler.fit(df[numerical_features])

            x_test = self.test_df.copy()

            x_test[numerical_features] = scaler.transform(x_test[numerical_features])

            if self.model == "nn":

                x_test = [np.absolute(x_test[i]) for i in self.categoricals] + [x_test[numerical_features]]

            else:

                x_test = x_test[self.features]

        else:

            x_test = self.test_df[self.features]



        # fitting with out of fold

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            # train test split

            x_train, x_val = self.train_df.loc[train_idx, self.features], self.train_df.loc[val_idx, self.features]

            y_train, y_val = self.train_df.loc[train_idx, self.target], self.train_df.loc[val_idx, self.target]



            # fitting & get feature importance

            if self.scaler is not None:

                x_train[numerical_features] = scaler.transform(x_train[numerical_features])

                x_val[numerical_features] = scaler.transform(x_val[numerical_features])

                if self.model == "nn":

                    x_train = [np.absolute(x_train[i]) for i in self.categoricals] + [x_train[numerical_features]]

                    x_val = [np.absolute(x_val[i]) for i in self.categoricals] + [x_val[numerical_features]]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model, importance = self.train_model(train_set, val_set)

            fi[fold, :] = importance

            conv_x_val = self.convert_x(x_val)

            y_vals[val_idx] = y_val

            x_test = self.convert_x(x_test)

            if (self.model == "linear") & (self.task != "regression"):

                oofs = model.predict_proba(conv_x_val)

                ypred = model.predict_proba(x_test) / self.n_splits

            else:

                oofs = model.predict(conv_x_val)

                ypred = model.predict(x_test) / self.n_splits

                if (self.model == "nn") & (self.task != "multiclass"):

                    oofs = oofs.ravel()

                    ypred = ypred.ravel()

            if len(oofs.shape) == 2:

                if oofs.shape[1] == 2:

                    oof_pred[val_idx] = oofs[:, -1]

                    y_pred += ypred[:, -1]

                elif oofs.shape[1] > 2:

                    oof_pred[val_idx] = np.argmax(oofs, axis=1)

                    y_pred += np.argmax(ypred, axis=1)

            else:

                oof_pred[val_idx] = oofs.reshape(oof_pred[val_idx].shape)

                y_pred += ypred.reshape(y_pred.shape)

            print('Partial score of fold {} is: {}'.format(fold, self.calc_metric(y_val, oof_pred[val_idx])))



        # feature importance data frame

        fi_df = pd.DataFrame()

        for n in np.arange(self.n_splits):

            tmp = pd.DataFrame()

            tmp["features"] = self.features

            tmp["importance"] = fi[n, :]

            tmp["fold"] = n

            fi_df = pd.concat([fi_df, tmp], ignore_index=True)

        gfi = fi_df[["features", "importance"]].groupby(["features"]).mean().reset_index()

        fi_df = fi_df.merge(gfi, on="features", how="left", suffixes=('', '_mean'))



        # outputs

        loss_score = self.calc_metric(y_vals, oof_pred)

        if self.verbose:

            print('Our oof loss score is: ', loss_score)

        return y_pred, loss_score, model, oof_pred, y_vals, fi_df
data_dict = {}

for i in glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/*'):

    name = i.split('/')[-1].split('.')[0]

    if name != 'MTeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='cp1252')
data_dict.keys()
fname = 'Cities'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MTeamCoaches'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MTeamSpellings'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MMasseyOrdinals'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MSeasons'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MTeams'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MSecondaryTourneyTeams'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MNCAATourneyCompactResults'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MGameCities'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'Conferences'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MNCAATourneySeeds'

print(data_dict[fname].shape)

data_dict[fname].head()
# get int from seed

data_dict['MNCAATourneySeeds']['Seed'] = data_dict['MNCAATourneySeeds']['Seed'].apply(lambda x: int(x[1:3]))

data_dict[fname].head()
fname = 'MNCAATourneySlots'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MTeamConferences'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MNCAATourneySeedRoundSlots'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MNCAATourneyDetailedResults'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MConferenceTourneyGames'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MRegularSeasonDetailedResults'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MRegularSeasonCompactResults'

print(data_dict[fname].shape)

data_dict[fname].head()
fname = 'MSecondaryTourneyCompactResults'

print(data_dict[fname].shape)

data_dict[fname].head()
# let's also have a look at test

test = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

print(test.shape)

test.head()
# format ID

test = test.drop(['Pred'], axis=1)

test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))

test['WTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[1]))

test['LTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[2]))

test.head()
# merge tables ============

train = data_dict['MNCAATourneyCompactResults'] # use compact data only for now



# # compact <- detailed (Tourney files)

# train = pd.merge(data_dict['MNCAATourneyCompactResults'], data_dict['MNCAATourneyDetailedResults'], how='left',

#              on=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'])

print(train.shape)

train.head()
# Train =================================

# merge with Game Cities

gameCities = pd.merge(data_dict['MGameCities'], data_dict['Cities'], how='left', on=['CityID'])

cols_to_use = gameCities.columns.difference(train.columns).tolist() + ["Season", "WTeamID", "LTeamID"]

train = train.merge(gameCities[cols_to_use].drop_duplicates(subset=["Season", "WTeamID", "LTeamID"]),

                    how="left", on=["Season", "WTeamID", "LTeamID"])

train.head()



# merge with MSeasons

cols_to_use = data_dict["MSeasons"].columns.difference(train.columns).tolist() + ["Season"]

train = train.merge(data_dict["MSeasons"][cols_to_use].drop_duplicates(subset=["Season"]),

                    how="left", on=["Season"])

train.head()



# merge with MTeams

cols_to_use = data_dict["MTeams"].columns.difference(train.columns).tolist()

train = train.merge(data_dict["MTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]),

                    how="left", left_on=["WTeamID"], right_on=["TeamID"])

train.drop(['TeamID'], axis=1, inplace=True)

train = train.merge(data_dict["MTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]),

                    how="left", left_on=["LTeamID"], right_on=["TeamID"], suffixes=('_W', '_L'))

train.drop(['TeamID'], axis=1, inplace=True)

print(train.shape)

train.head()
# merge with MTeamCoaches

cols_to_use = data_dict["MTeamCoaches"].columns.difference(train.columns).tolist() + ["Season"]

train = train.merge(data_dict["MTeamCoaches"][cols_to_use].drop_duplicates(subset=["Season","TeamID"]), 

                    how="left", left_on=["Season","WTeamID"], right_on=["Season","TeamID"])

train.drop(['TeamID'], axis=1, inplace=True)



train = train.merge(data_dict["MTeamCoaches"][cols_to_use].drop_duplicates(subset=["Season","TeamID"]), 

                    how="left", left_on=["Season","LTeamID"], right_on=["Season","TeamID"], suffixes=('_W', '_L'))

train.drop(['TeamID'], axis=1, inplace=True)

print(train.shape)

train.head()



# # merge with MMasseyOrdinals (too heavy for kaggle kernel?)

# cols_to_use = data_dict["MMasseyOrdinals"].columns.difference(train.columns).tolist() + ["Season"]

# train = train.merge(data_dict["MMasseyOrdinals"], how="left", left_on=["Season","WTeamID"], right_on=["Season","TeamID"])

# train.drop(['TeamID'], axis=1, inplace=True)

# train = train.merge(data_dict["MMasseyOrdinals"], how="left", left_on=["Season","LTeamID"], right_on=["Season","TeamID"], suffixes=('_W', '_L'))

# train.drop(['TeamID'], axis=1, inplace=True)
# merge with MNCAATourneySeeds

cols_to_use = data_dict['MNCAATourneySeeds'].columns.difference(train.columns).tolist() + ['Season']

train = train.merge(data_dict['MNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),

                    how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

train.drop(['TeamID'], axis=1, inplace=True)

train = train.merge(data_dict['MNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),

                    how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))

train.drop(['TeamID'], axis=1, inplace=True)



print(train.shape)

train.head()
# test =================================

# merge with Game Cities

cols_to_use = gameCities.columns.difference(test.columns).tolist() + ["Season", "WTeamID", "LTeamID"]

test = test.merge(gameCities[cols_to_use].drop_duplicates(subset=["Season", "WTeamID", "LTeamID"]),

                  how="left", on=["Season", "WTeamID", "LTeamID"])

del gameCities

gc.collect()

test.head()



# merge with MSeasons

cols_to_use = data_dict["MSeasons"].columns.difference(test.columns).tolist() + ["Season"]

test = test.merge(data_dict["MSeasons"][cols_to_use].drop_duplicates(subset=["Season"]),

                  how="left", on=["Season"])

test.head()



# merge with MTeams

cols_to_use = data_dict["MTeams"].columns.difference(test.columns).tolist()

test = test.merge(data_dict["MTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]),

                  how="left", left_on=["WTeamID"], right_on=["TeamID"])

test.drop(['TeamID'], axis=1, inplace=True)

test = test.merge(data_dict["MTeams"][cols_to_use].drop_duplicates(subset=["TeamID"]), 

                  how="left", left_on=["LTeamID"], right_on=["TeamID"], suffixes=('_W', '_L'))

test.drop(['TeamID'], axis=1, inplace=True)

test.head()



# merge with MTeamCoaches

cols_to_use = data_dict["MTeamCoaches"].columns.difference(test.columns).tolist() + ["Season"]

test = test.merge(data_dict["MTeamCoaches"][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),

                  how="left", left_on=["Season","WTeamID"], right_on=["Season","TeamID"])

test.drop(['TeamID'], axis=1, inplace=True)

test = test.merge(data_dict["MTeamCoaches"][cols_to_use].drop_duplicates(subset=["Season","TeamID"]), 

                  how="left", left_on=["Season","LTeamID"], right_on=["Season","TeamID"], suffixes=('_W', '_L'))

test.drop(['TeamID'], axis=1, inplace=True)



# # merge with MMasseyOrdinals

# cols_to_use = data_dict["MMasseyOrdinals"].columns.difference(test.columns).tolist() + ["Season"]

# test = test.merge(data_dict["MMasseyOrdinals"], how="left", left_on=["Season","WTeamID"], right_on=["Season","TeamID"])

# test.drop(['TeamID'], axis=1, inplace=True)

# test = test.merge(data_dict["MMasseyOrdinals"], how="left", left_on=["Season","LTeamID"], right_on=["Season","TeamID"], suffixes=('_W', '_L'))

# test.drop(['TeamID'], axis=1, inplace=True)



# merge with MNCAATourneySeeds

cols_to_use = data_dict['MNCAATourneySeeds'].columns.difference(test.columns).tolist() + ['Season']

test = test.merge(data_dict['MNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),

                  how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

test.drop(['TeamID'], axis=1, inplace=True)

test = test.merge(data_dict['MNCAATourneySeeds'][cols_to_use].drop_duplicates(subset=["Season","TeamID"]),

                  how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))

test.drop(['TeamID'], axis=1, inplace=True)



print(test.shape)

test.head()
not_exist_in_test = [c for c in train.columns.values.tolist() if c not in test.columns.values.tolist()]

print(not_exist_in_test)

train = train.drop(not_exist_in_test, axis=1)

train.head()
# to datetime format

train["DayZero"] = pd.to_datetime(train["DayZero"], infer_datetime_format=True)



# sort by date

train = train.sort_values(by=["DayZero", "Season", "DayNum"]).reset_index(drop=True)
# compact <- detailed (regular season files)

regularSeason = data_dict['MRegularSeasonCompactResults']

# regularSeason = pd.merge(data_dict['MRegularSeasonCompactResults'], data_dict['MRegularSeasonDetailedResults'], how='left',

#              on=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT'])

print(regularSeason.shape)

regularSeason.head()
# merge with regularSeason using only the previous data

def merge_regularSeason(df, regularSeason):

    df_new = pd.DataFrame()

    for i, season in enumerate(df["Season"].unique()):

        print(season)

        if season <= 1998:

            continue

            

        # split winners and losers (make sure not to use the future data!)

        team_win_score = regularSeason.loc[regularSeason["Season"] <= season, :].groupby(['WTeamID']).agg({'WScore':['sum', 'count', 'var']}).reset_index()

        team_win_score.columns = [' '.join(col).strip() for col in team_win_score.columns.values]

        team_loss_score = regularSeason.loc[regularSeason["Season"] <= season, :].groupby(['LTeamID']).agg({'LScore':['sum', 'count', 'var']}).reset_index()

        team_loss_score.columns = [' '.join(col).strip() for col in team_loss_score.columns.values]

        

        # merge with train 

        team_win_score["Season"] = season

        team_loss_score["Season"] = season

        df_fold = df.loc[df["Season"] == season, :].reset_index(drop=True)

        df_fold = df_fold.merge(team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])

        df_fold = df_fold.merge(team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])

        df_fold = df_fold.merge(team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])

        df_fold = df_fold.merge(team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])

        df_fold.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

        

        # restore

        df_new = pd.concat([df_new, df_fold], ignore_index=True)

    return df_new



train = merge_regularSeason(train, regularSeason)

test = merge_regularSeason(test, regularSeason)
print(train.shape)

train.head()
print(test.shape)

test.tail()
# preprocess

def preprocess(df):

    df['x_score'] = df['WScore sum_x'] + df['LScore sum_y']

    df['y_score'] = df['WScore sum_y'] + df['LScore sum_x']

    df['x_count'] = df['WScore count_x'] + df['LScore count_y']

    df['y_count'] = df['WScore count_y'] + df['WScore count_x']

    df['x_var'] = df['WScore var_x'] + df['LScore var_x']

    df['y_var'] = df['WScore var_y'] + df['LScore var_y']

    return df

train = preprocess(train)

test = preprocess(test)
# make winner and loser train

train_win = train.copy()

train_los = train.copy()

train_win = train_win[['DayNum', 'DayZero', 'Season', 'Seed_W', 'Seed_L', 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]

train_los = train_los[['DayNum', 'DayZero', 'Season', 'Seed_L', 'Seed_W', 'y_score', 'x_score', 'x_count', 'y_count', 'x_var', 'y_var']]

train_win.columns = ['DayNum', 'DayZero', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']

train_los.columns = ['DayNum', 'DayZero', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']



# same processing for test

test = test[['ID', 'Season', 'Seed_W', 'Seed_L', 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]

test.columns = ['ID', 'Season', 'Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']
# feature enginnering

def feature_engineering(df):

    df['Seed_diff'] = df['Seed_1'] - df['Seed_2']

    df['Seed_ratio'] = df['Seed_1'] / df['Seed_2']

    df['Score_ratio'] = df['Score_1'] / df['Score_2']

    df['Count_ratio'] = df['Count_1'] / df['Count_2']

    df['Var_ratio'] = df['Var_1'] / df['Var_2']

    df['Mean_score1'] = df['Score_1'] / df['Count_1']

    df['Mean_score2'] = df['Score_2'] / df['Count_2']

    df['Mean_score_diff'] = df['Mean_score1'] - df['Mean_score2']

    df['Mean_score_ratio'] = df['Mean_score1'] / df['Mean_score2']

    df = df.drop(['Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2'], axis=1)

    return df

train_win = feature_engineering(train_win)

train_los = feature_engineering(train_los)

test = feature_engineering(test)
train_win["result"] = 1

print(train_win.shape)

train_win.head()
train_los["result"] = 0

print(train_los.shape)

train_los.head()
data = pd.concat([train_win, train_los], ignore_index=True)

data = data.sort_values(by=['DayZero', 'Season', 'DayNum']).reset_index(drop=True)

print(data.shape)

data.head()
target = 'result'

features = data.columns.values.tolist()

dropcols = [target,'DayNum', 'Season', 'DayZero']

features = [f for f in features if f not in dropcols]

print(features)
# predict 2015 by using ~2014, for example

models = {}

test["Pred"] = 0.5

pred = np.zeros(test.shape[0])

for season in test["Season"].unique():

    print(f"Predicting {season}...")

    lin = RunModel(data.loc[data["Season"] < season, :], test.loc[test["Season"] == season, :], target, features, categoricals=[], n_splits=5, 

                   model='linear', cv_method="TimeSeriesSplit", group=None, task="binary", scaler="Standard", seed=1220, verbose=True)

    models[season] = lin

    test.loc[test["Season"] == season, "Pred"] = lin.y_pred
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = test['Pred']

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)