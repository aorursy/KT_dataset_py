import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from matplotlib import pyplot

import xgboost as xgb

import lightgbm as lgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from catboost import CatBoostClassifier, CatBoostRegressor

import shap



from time import time

from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import OneHotEncoder

import gc

import json



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def read_data():

    tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

    tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')

    season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

    test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

    submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

    return tourney_result, tourney_seed, season_result, test_df
def get_train_test(tourney_result,tourney_seed,season_result,test_df):

    # deleting unnecessary columns

    tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

    # Merge Seed

    tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

    tourney_result = tourney_result.drop('TeamID', axis=1)

    tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

    tourney_result = tourney_result.drop('TeamID', axis=1)



    def get_seed(x):

        return int(x[1:3])



    tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))

    tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))

    # Merge Score

    season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

    season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]

    season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

    season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

    season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

    season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

    tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)

    tourney_result = tourney_result.drop('TeamID', axis=1)

    tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)

    tourney_result = tourney_result.drop('TeamID', axis=1)

    tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

    tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)

    tourney_lose_result = tourney_win_result.copy()

    tourney_lose_result['Seed1'] = tourney_win_result['Seed2']

    tourney_lose_result['Seed2'] = tourney_win_result['Seed1']

    tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']

    tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']

    tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']

    tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']

    tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

    tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']

    tourney_win_result['result'] = 1

    tourney_lose_result['result'] = 0

    tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

    train_df = tourney_result

    # Get Test

    test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

    test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

    test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

    test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

    test_df = test_df.drop('TeamID', axis=1)

    test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    test_df.rename(columns={'Seed':'Seed2'}, inplace=True)

    test_df = test_df.drop('TeamID', axis=1)

    test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

    test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)

    test_df = test_df.drop('TeamID', axis=1)

    test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

    test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)

    test_df = test_df.drop('TeamID', axis=1)

    test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))

    test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

    test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

    test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

    test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

    return train_df, test_df
tourney_result, tourney_seed, season_result, test_df = read_data()

train_df, test_df = get_train_test(tourney_result,tourney_seed,season_result,test_df)

test_df['result']=np.NaN

del tourney_result, tourney_seed, season_result
print(f"Train dataset has {train_df.shape[0]} rows and {train_df.shape[1]} cols")

print(f"Test dataset has {test_df.shape[0]} rows and {test_df.shape[1]} cols")
test_df
class Base_Model(object):

    

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'result'

        self.cv = self.get_cv()

        self.verbose = verbose

        self.params = self.get_params()

        self.y_pred, self.model = self.fit()

        

    def train_model(self, train_set, val_set):

        raise NotImplementedError

        

    def get_cv(self):

        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        return cv.split(self.train_df, self.train_df[self.target])

    

    def get_params(self):

        raise NotImplementedError

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError

        

    def convert_x(self, x):

        return x

        

    def fit(self):

        oof_pred = np.zeros((len(train_df), ))

        y_pred = np.zeros((len(test_df), ))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            print('Fold:',fold+1)

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

        return y_pred, model
class Lgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return lgb.train(self.params, train_set, 10000, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)

        return train_set, val_set

        

    def get_params(self):

        params = {'num_leaves': 400,

                  'min_child_weight': 0.034,

                  'feature_fraction': 0.379,

                  'bagging_fraction': 0.418,

                  'min_data_in_leaf': 106,

                  'objective': 'binary',

                  'max_depth': -1,

                  'learning_rate': 0.0068,

                  "boosting_type": "gbdt",

                  "bagging_seed": 11,

                  "metric": 'logloss',

                  "verbosity": -1,

                  'reg_alpha': 0.3899,

                  'reg_lambda': 0.648,

                  'random_state': 47,

                    }

        return params
class Xgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return xgb.train(self.params, train_set, 

                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 

                         verbose_eval=verbosity, early_stopping_rounds=100)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = xgb.DMatrix(x_train, y_train)

        val_set = xgb.DMatrix(x_val, y_val)

        return train_set, val_set

    

    def convert_x(self, x):

        return xgb.DMatrix(x)

        

    def get_params(self):

        params = { 'colsample_bytree': 0.8,                 

                   'learning_rate': 0.01,

                   'max_depth': 3,

                   'subsample': 1,

                   'objective':'binary:logistic',

                   'eval_metric':'logloss',

                   'min_child_weight':3,

                   'gamma':0.25,

                   'n_estimators':5000}

        return params
class Catb_Model(Base_Model):

    

    def train_model(self, train_df, test_df):

        verbosity = 100 if self.verbose else 0

        clf = CatBoostClassifier(**self.params)

        clf.fit(train_df['X'], 

                train_df['y'], 

                eval_set=(test_df['X'], test_df['y']),

                verbose=verbosity, 

                cat_features=self.categoricals)

        return clf

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = {'X': x_train, 'y': y_train}

        val_set = {'X': x_val, 'y': y_val}

        return train_set, val_set

        

    def get_params(self):

        params = {'loss_function': 'Logloss',

                   'task_type': "CPU",

                   'iterations': 5000,

                   'od_type': "Iter",

                    'depth': 3,

                  'colsample_bylevel': 0.5, 

                   'early_stopping_rounds': 300,

                    'l2_leaf_reg': 18,

                   'random_seed': 42,

                    'use_best_model': True

                    }

        return params
features = train_df.columns

features = [x for x in features if x not in ['result']]

print(features)

categoricals = []



#cat_model = Catb_Model(train_df, test_df, features, categoricals=categoricals)

lgb_model = Lgb_Model(train_df, test_df, features, categoricals=categoricals)

xgb_model = Xgb_Model(train_df, test_df, features, categoricals=categoricals)
weights = {'lgb': 0.60, 'cat':0, 'xgb':0.40}

submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = (lgb_model.y_pred*weights['lgb']) + (xgb_model.y_pred*weights['xgb'])

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)