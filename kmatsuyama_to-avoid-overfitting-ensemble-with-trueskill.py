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
data_dict = {}

for i in glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/*'):

    name = i.split('/')[-1].split('.')[0]

    if name != 'MTeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='cp1252')
# process seed

data_dict['MNCAATourneySeeds']['Seed'] = data_dict['MNCAATourneySeeds']['Seed'].apply(lambda x: int(x[1:3]))

# take only useful columns

data_dict['MNCAATourneySeeds'] = data_dict['MNCAATourneySeeds'][['Season', 'TeamID', 'Seed']]

data_dict['MNCAATourneyCompactResults'] = data_dict['MNCAATourneyCompactResults'][['Season','WTeamID', 'LTeamID']]



# merge the data and rename the columns

df = pd.merge(data_dict['MNCAATourneyCompactResults'], data_dict['MNCAATourneySeeds'],

              how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'])

df = pd.merge(df, data_dict['MNCAATourneySeeds'], how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'])

df = df.drop(['TeamID_x', 'TeamID_y'], axis=1)

df.columns = ['Season', 'WTeamID', 'LTeamID', 'WSeed', 'LSeed']

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



df['x_score'] = df['WScore_sum_x'] + df['LScore_sum_y']

df['y_score'] = df['WScore_sum_y'] + df['LScore_sum_x']

df['x_count'] = df['WScore_count_x'] + df['LScore_count_y']

df['y_count'] = df['WScore_count_y'] + df['WScore_count_x']



df = df.drop(['WScore_sum_x','WScore_count_x','LScore_sum_x','LScore_count_x',

              'LScore_sum_y','LScore_count_y','WScore_sum_y','WScore_count_y'], axis =1)

df.head()
df_win = df.copy()

df_los = df.copy()

df_win = df_win[['WSeed', 'LSeed', 'x_score', 'y_score', 'x_count', 'y_count']]

df_los = df_los[['LSeed', 'WSeed', 'y_score', 'x_score', 'x_count', 'y_count']]

df_win.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']

df_los.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']

df_win['Seed_diff'] = df_win['Seed_1'] - df_win['Seed_2']

df_win['Score_diff'] = df_win['Score_1'] - df_win['Score_2']

df_los['Seed_diff'] = df_los['Seed_1'] - df_los['Seed_2']

df_los['Score_diff'] = df_los['Score_1'] - df_los['Score_2']
df_win['result'] = 1

df_los['result'] = 0

data = pd.concat((df_win, df_los)).reset_index(drop=True)
data.head()
# visualize the data



plt.figure(figsize=(24, 12))

tmp1 = data[(data['result']==1)]

tmp0 = data[(data['result']==0)]

vis_cols = [c for c in data.columns if c not in ['result']]

for idx, col in enumerate(vis_cols):

    plt.subplot(3,  3, idx+1)

    plt.hist(tmp1[col], bins=25, alpha=0.5, label='win')

    plt.hist(tmp0[col], bins=25, alpha=0.5, label='lose')

    plt.legend(loc='best')

    plt.title(col)

plt.tight_layout()

plt.show()
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

test.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

test.head()
test['x_score'] = test['WScore_sum_x'] + test['LScore_sum_y']

test['y_score'] = test['WScore_sum_y'] + test['LScore_sum_x']

test['x_count'] = test['WScore_count_x'] + test['LScore_count_y']

test['y_count'] = test['WScore_count_y'] + test['WScore_count_x']



test = test[['Seed_x', 'Seed_y', 'x_score', 'y_score', 'x_count', 'y_count']]

test.columns = ['Seed_1', 'Seed_2', 'Score_1', 'Score_2', 'Count_1', 'Count_2']



test['Seed_diff'] = test['Seed_1'] - test['Seed_2']

test['Score_diff'] = test['Score_1'] - test['Score_2']



test_df = test
test_df.head()
params_lgb = {'num_leaves': 127,

          'min_data_in_leaf': 10,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'logloss',

          "verbosity": -1,

          'random_state': 42,

         }

X = data.drop('result', axis=1)

y = data['result']
import gc



NFOLDS = 10

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds_lgb = np.zeros(test_df.shape[0])

y_train_lgb = np.zeros(X.shape[0])

y_oof = np.zeros(X.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    

    y_train_lgb += clf.predict(X) / NFOLDS

    y_preds_lgb += clf.predict(test_df) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
plt.hist(y_preds_lgb);
"""

about TrueSkill, thanks to:

https://www.kaggle.com/zeemeen/ncaa-trueskill-script

"""

import trueskill

from trueskill import rate_1vs1
def expose_and_clip(rating, env=None, minimum=0., maximum=50.):

    env = env if env else trueskill.global_env()

    return min(max(minimum, env.expose(rating)), maximum)

env = trueskill.TrueSkill()
df_compact = data_dict['MNCAATourneyCompactResults']

df_detailed = data_dict['MNCAATourneyDetailedResults']

results_merged  = pd.merge(df_compact, df_detailed)



results_merged.head()
#we added 4 factors

results_merged['WeFG%'] = (results_merged['WFGM']+0.5*results_merged['WFGM3']) / results_merged['WFGA']

results_merged['LeFG%'] = (results_merged['LFGM']+0.5*results_merged['LFGM3']) / results_merged['LFGA']

results_merged['WTO%'] = results_merged['WTO'] / (results_merged['WFGA']+0.44*results_merged['WFTA'] +results_merged['WTO'])

results_merged['LTO%'] = results_merged['LTO'] / (results_merged['LFGA']+0.44*results_merged['LFTA'] +results_merged['LTO'])

results_merged['WFTR%'] = results_merged['WFTA'] / results_merged['WFGA']

results_merged['LFTR%'] = results_merged['LFTA'] / results_merged['LFGA']

results_merged['WORB%'] = results_merged['WOR'] / results_merged['LDR']

results_merged['LORB%'] = results_merged['LOR'] / results_merged['WDR']
import pandas as pd, numpy as np

from trueskill import TrueSkill, Rating, rate_1vs1



ts = TrueSkill(draw_probability=0.01) # 0.01 is arbitary small number

#beta = 25 / 6  # default value

beta = 25 / 6  



def win_probability(p1, p2):

    delta_mu = p1.mu - p2.mu

    sum_sigma = p1.sigma * p1.sigma + p2.sigma * p2.sigma

    denom = np.sqrt(2 * (beta * beta) + sum_sigma)

    return ts.cdf(delta_mu / denom)

    

submit = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submit[['Season', 'Team1', 'Team2']] = submit.apply(lambda r:pd.Series([int(t) for t in r.ID.split('_')]), axis=1)



df_tour = results_merged

teamIds = np.unique(np.concatenate([df_tour.WTeamID.values, df_tour.LTeamID.values]))

ratings = { tid:ts.Rating() for tid in teamIds }



def feed_season_results(season):

    print("season = {}".format(season))

    df1 = df_tour[df_tour.Season == season]

    for r in df1.itertuples():

        ratings[r.WTeamID], ratings[r.LTeamID] = rate_1vs1(ratings[r.WTeamID], ratings[r.LTeamID])



def update_pred(season):

    beta = np.std([r.mu for r in ratings.values()]) 

    print("beta = {}".format(beta))

    submit.loc[submit.Season==season, 'Pred'] = submit[submit.Season==season].apply(lambda r:win_probability(ratings[r.Team1], ratings[r.Team2]), axis=1)



for season in sorted(df_tour.Season.unique())[:-5]: # exclude last 4 years

    feed_season_results(season)



update_pred(2015)

feed_season_results(2015)

update_pred(2016)

feed_season_results(2016)

update_pred(2017)

feed_season_results(2017)

update_pred(2018)

feed_season_results(2018)

update_pred(2019)



submit.drop(['Season', 'Team1', 'Team2'], axis=1, inplace=True)

y_preds_ts = submit['Pred']
plt.hist(submit['Pred']);
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = 0.4*y_preds_lgb + 0.6*y_preds_ts
plt.hist(submission_df['Pred']);
submission_df.to_csv('submission.csv', index=False)