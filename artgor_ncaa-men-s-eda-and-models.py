# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



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

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

import json

import ast

import time

from sklearn import linear_model

import eli5

from eli5.sklearn import PermutationImportance

import shap



from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')



import os

import glob
print(os.listdir('../input/'))

print(os.listdir('../input/datafiles'))
masseyordinals = pd.read_csv('../input/masseyordinals/MasseyOrdinals.csv')

sub = pd.read_csv('../input/SampleSubmissionStage1.csv')

sub.head()
data_dict = {}

for i in glob.glob('../input/datafiles/*'):

    name = i.split('/')[-1].split('.')[0]

    print(i)

    if name != 'TeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='latin-1')        
data_dict['Teams'].head()
data_dict['TeamSpellings'].head()
team_counts = data_dict['TeamSpellings'].groupby('TeamID')['TeamNameSpelling'].count().reset_index()

team_counts.columns = ['TeamID', 'TeamSpellingCount']
plt.title('Count of team counts');

team_counts['TeamSpellingCount'].value_counts().sort_index().plot(kind='barh', color='teal');
data_dict['NCAATourneySeeds'].head()
data_dict['NCAATourneySeeds']['Seed'] = data_dict['NCAATourneySeeds']['Seed'].apply(lambda x: int(x[1:3]))

data_dict['NCAATourneySeeds'] = data_dict['NCAATourneySeeds'][['Season', 'TeamID', 'Seed']]

data_dict['NCAATourneySeeds'].head()
data_dict['NCAATourneyCompactResults'] = data_dict['NCAATourneyCompactResults'][['Season','WTeamID', 'LTeamID']]

data_dict['NCAATourneyCompactResults'].head()
df = pd.merge(data_dict['NCAATourneyCompactResults'], data_dict['NCAATourneySeeds'], how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

df = pd.merge(df, data_dict['NCAATourneySeeds'], how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])

df = df.drop(['TeamID_x', 'TeamID_y'], axis=1)

df['seed_diff'] = df['Seed_x'] - df['Seed_y']

df.head()
data_dict['RegularSeasonCompactResults'].head()
plt.title('Mean scores of winning teams by season');

data_dict['RegularSeasonCompactResults'].groupby(['Season'])['WScore'].mean().plot();
data_dict['RegularSeasonCompactResults']['Season'] += 1
team_win_score = data_dict['RegularSeasonCompactResults'].groupby(['Season', 'WTeamID'])['WScore'].mean().reset_index()

team_loss_score = data_dict['RegularSeasonCompactResults'].groupby(['Season', 'LTeamID'])['LScore'].mean().reset_index()

df = pd.merge(df, team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])

df = pd.merge(df, team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])

df = pd.merge(df, team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])

df = pd.merge(df, team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])

df.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

df = df.loc[(df['Season'] > 1985) & (df['Season'] < 2014)]

df.head()
data_dict['RegularSeasonDetailedResults'].head()
data_dict['RegularSeasonDetailedResults']['Season_join'] = data_dict['RegularSeasonDetailedResults']['Season'] + 1
plt.title('Mean number of field goals made by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFGM'].mean().plot();
plt.title('Mean number of field goals attempted by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFGA'].mean().plot();
plt.title('Mean number of three pointers made by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFGM3'].mean().plot();
plt.title('Mean number of three pointers attempted by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFGA3'].mean().plot();
plt.title('Mean number of free throws made by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFTM'].mean().plot();
plt.title('Mean number of free throws made by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WFTA'].mean().plot();
plt.title('Mean number of offensive rebounds pulled by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WOR'].mean().plot();
plt.title('Mean number of defensive rebounds pulled by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WDR'].mean().plot();
plt.title('Mean number of assists by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WAst'].mean().plot();
plt.title('Mean number of turnovers committed by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WTO'].mean().plot();
plt.title('Mean number of steals accomplished by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WStl'].mean().plot();
plt.title('Mean number of blocks accomplished by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WBlk'].mean().plot();
plt.title('Mean number of personal fouls committed by winning teams by season');

data_dict['RegularSeasonDetailedResults'].groupby(['Season'])['WPF'].mean().plot();
df.head()
loss_df = df[df['WTeamID_x'] > df['LTeamID_x']]

win_df = df[df['WTeamID_x'] < df['LTeamID_x']]

win_df['target'] = 1

win_df.columns = ['Season', 'Team1', 'Team2', 'Seed_1', 'Seed_2', 'seed_diff',

       'WScore_1', 'LScore_1', 'LScore_2', 'WScore_2', 'target']

loss_df['target'] = 0

loss_df = loss_df[['Season', 'LTeamID_x', 'WTeamID_x', 'Seed_y', 'Seed_x', 'seed_diff',

       'LScore_y', 'WScore_y', 'WScore_x', 'LScore_x', 'target']]

loss_df.columns = ['Season', 'Team1', 'Team2', 'Seed_1', 'Seed_2', 'seed_diff',

       'WScore_1', 'LScore_1', 'LScore_2', 'WScore_2', 'target']

loss_df['seed_diff'] = -1 * loss_df['seed_diff']

new_df = win_df.append(loss_df)
new_df.head()
test = sub.copy()

sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))

sub['Team1'] = sub['ID'].apply(lambda x: int(x.split('_')[1]))

sub['Team2'] = sub['ID'].apply(lambda x: int(x.split('_')[2]))

sub = pd.merge(sub, data_dict['NCAATourneySeeds'], how='left', left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'])

sub = pd.merge(sub, data_dict['NCAATourneySeeds'], how='left', left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'])

sub = pd.merge(sub, team_win_score, how='left', left_on=['Season', 'Team1'], right_on=['Season', 'WTeamID'])

sub = pd.merge(sub, team_loss_score, how='left', left_on=['Season', 'Team2'], right_on=['Season', 'LTeamID'])

sub = pd.merge(sub, team_loss_score, how='left', left_on=['Season', 'Team1'], right_on=['Season', 'LTeamID'])

sub = pd.merge(sub, team_win_score, how='left', left_on=['Season', 'Team2'], right_on=['Season', 'WTeamID'])

sub['seed_diff'] = sub['Seed_x'] - sub['Seed_y']

sub.head()
new_df = pd.merge(new_df, team_counts, how='left', left_on='Team1', right_on='TeamID')

new_df = new_df.drop(['TeamID'], axis=1)

new_df = pd.merge(new_df, team_counts, how='left', left_on='Team2', right_on='TeamID')

new_df = new_df.drop(['TeamID'], axis=1)



sub = pd.merge(sub, team_counts, how='left', left_on='Team1', right_on='TeamID')

sub = sub.drop(['TeamID'], axis=1)

sub = pd.merge(sub, team_counts, how='left', left_on='Team2', right_on='TeamID')

sub = sub.drop(['TeamID'], axis=1)



new_df = new_df.drop(['Season', 'Team1', 'Team2'], axis=1)

sub = sub.drop(['Pred', 'Season', 'Team1', 'Team2', 'TeamID_x', 'TeamID_y', 'WTeamID_x', 'WTeamID_y', 'LTeamID_x', 'LTeamID_y'], axis=1)

sub.columns = ['ID', 'Seed_1', 'Seed_2', 'WScore_1', 'LScore_1', 'LScore_2', 'WScore_2', 'seed_diff', 'TeamSpellingCount_x', 'TeamSpellingCount_y']

sub = sub[['ID', 'Seed_1', 'Seed_2', 'seed_diff', 'WScore_1', 'LScore_1', 'LScore_2', 'WScore_2', 'TeamSpellingCount_x', 'TeamSpellingCount_y']]

new_df = new_df.fillna(0)

sub = sub.fillna(0)
X = new_df.drop(['target'], axis=1)

y = new_df['target']

X_test = sub.drop(['ID'], axis=1)
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y.values[train_index], y.values[valid_index]

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train)

            valid_data = lgb.Dataset(X_valid, label=y_valid)

            

            model = lgb.train(params,

                    train_data,

                    num_boost_round=20000,

                    valid_sets = [train_data, valid_data],

                    verbose_eval=1000,

                    early_stopping_rounds = 200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)[:, 1].reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            # print(f'Fold {fold_n}. AUC: {score:.4f}.')

            # print('')

            

            y_pred = model.predict_proba(X_test)[:, 1]

            

        if model_type == 'glm':

            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())

            model_results = model.fit()

            model_results.predict(X_test)

            y_pred_valid = model_results.predict(X_valid).reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            

            y_pred = model_results.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000, learning_rate=0.1, loss_function='Logloss',  eval_metric='AUC', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test)

            

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(log_loss(y_valid, y_pred_valid))



        if averaging == 'usual':

            prediction += y_pred

        elif averaging == 'rank':

            prediction += pd.Series(y_pred).rank().values  

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importance()

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction, scores

    

    else:

        return oof, prediction, scores
X1 = X.copy()

X_test1 = X_test.copy()
scaler = StandardScaler()

X1[['WScore_1', 'LScore_1', 'LScore_2', 'WScore_2']] = scaler.fit_transform(X1[['WScore_1', 'LScore_1', 'LScore_2', 'WScore_2']])

X_test1[['WScore_1', 'LScore_1', 'LScore_2', 'WScore_2']] = scaler.transform(X_test1[['WScore_1', 'LScore_1', 'LScore_2', 'WScore_2']])
%%time

model = linear_model.LogisticRegression(C=0.0001)

oof_lr, prediction_lr, scores = train_model(X1, X_test1, y, params=None, folds=folds, model_type='sklearn', model=model)
params = {'num_leaves': 8,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 5,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1.728910519108444,

         'reg_lambda': 4.9847051755586085,

         'random_state': 42,

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 4}

oof_lgb, prediction_lgb, scores = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
test['Pred'] = prediction_lr

test.to_csv('submission.csv', index=False)
test['Pred'] = prediction_lgb

test.to_csv('lgb.csv', index=False)