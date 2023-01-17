# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn import preprocessing

import lightgbm as lgb

import optuna

import glob
path = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'

Files = 'WDataFiles_Stage1/'
TourneyCompactResults = pd.read_csv(path + Files + 'WNCAATourneyCompactResults.csv')

GameCities = pd.read_csv(path+Files+'WGameCities.csv')

Seasons = pd.read_csv(path+Files+'WSeasons.csv')

TourneySeeds = pd.read_csv(path+Files+'WNCAATourneySeeds.csv')

RegularSeasonCompactResults = pd.read_csv(path+Files+'WRegularSeasonCompactResults.csv')
test= pd.read_csv(path +'WSampleSubmissionStage1_2020.csv')
print(TourneyCompactResults.shape)

TourneyCompactResults.head()
print(GameCities.shape)

GameCities.head()
print(Seasons.shape)

Seasons.head()
TourneySeeds['Seed'] = TourneySeeds['Seed'].apply(lambda x: int(x[1:3]))

print(TourneySeeds.shape)

TourneySeeds.head()
print(RegularSeasonCompactResults.shape)

RegularSeasonCompactResults.head()
print(test.shape)

test.head()
train = TourneyCompactResults

train.head()
train = train.replace({'H':0,'A':1,'N':2})

train.head()
le = preprocessing.LabelEncoder()

for column in ['CRType']:

    le.fit(GameCities[column])

    GameCities[column] = le.transform(GameCities[column])

GameCities.head()
#train = train.merge(GameCities,how='left',on=['Season','WTeamID','LTeamID'],indicator = False)

#train.drop('DayNum_y',inplace=True,axis=1)

#train = train.rename(columns={'DayNum_x': 'DayNum'})

#train.head()
le = preprocessing.LabelEncoder()

for column in ['RegionW','RegionX','RegionY','RegionZ']:

    le.fit(Seasons[column])

    Seasons[column] = le.transform(Seasons[column])



for i in range(0,23):

    print(Seasons['DayZero'][i].split('/'))

    Seasons['ZeroMonth'] = Seasons['DayZero'][i].split('/')[0]

    Seasons['ZeroDay'] = Seasons['DayZero'][i].split('/')[1]

    Seasons['ZeroYear'] = Seasons['DayZero'][i].split('/')[2]



Seasons = Seasons.drop('DayZero',axis=1)

Seasons['ZeroMonth'] = Seasons['ZeroMonth'].astype(int)

Seasons['ZeroDay'] = Seasons['ZeroMonth'].astype(int)

Seasons['ZeroYear'] = Seasons['ZeroMonth'].astype(int)

Seasons.head()
train = train.merge(Seasons, how='left',on=['Season'])

train.head()
train = train.merge(TourneySeeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season','TeamID'])

train = train.drop('TeamID',axis=1)

train = train.rename(columns={'Seed': 'WSeed'})



train = train.merge(TourneySeeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season','TeamID'])

train = train.drop('TeamID',axis=1)

train = train.rename(columns={'Seed': 'LSeed'})



train.head()
# format ID

test = test.drop(['Pred'], axis=1)

test['Season'] = test['ID'].apply(lambda x: int(x.split('_')[0]))

test['WTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[1]))

test['LTeamID'] = test['ID'].apply(lambda x: int(x.split('_')[2]))



test.head()
test = test.merge(TourneyCompactResults,how='left',on=['Season','WTeamID','LTeamID'])


test = test.replace({'H':0,'A':1,'N':2})



#test = test.merge(GameCities,how='left',on=['Season','WTeamID','LTeamID'])

#test = test.rename(columns={'DayNum_x': 'DayNum'})

#print(test.shape)



test = test.merge(Seasons, how='left',on=['Season'])



test = test.merge(TourneySeeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season','TeamID'])

test = test.drop('TeamID',axis=1)

test = test.rename(columns={'Seed': 'WSeed'})





test = test.merge(TourneySeeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season','TeamID'])

test = test.drop('TeamID',axis=1)

test = test.rename(columns={'Seed': 'LSeed'})

test.merge(test,how='left',on=['ID','Season','WTeamID','LTeamID'])
not_exist_in_test = [c for c in train.columns.values.tolist() if c not in test.columns.values.tolist()]

print(not_exist_in_test)

train = train.drop(not_exist_in_test, axis=1)

train.head()
#RegularSeasonCompactResults

# split winners and losers

team_win_score = RegularSeasonCompactResults.groupby(['Season', 'WTeamID']).agg({'WScore':['sum', 'count', 'var']}).reset_index()

team_win_score.columns = [' '.join(col).strip() for col in team_win_score.columns.values]

team_loss_score = RegularSeasonCompactResults.groupby(['Season', 'LTeamID']).agg({'LScore':['sum', 'count', 'var']}).reset_index()

team_loss_score.columns = [' '.join(col).strip() for col in team_loss_score.columns.values]



print(team_win_score.shape)

team_win_score.head()
# merge with train 

train = pd.merge(train, team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])

train = pd.merge(train, team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])

train = pd.merge(train, team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])

train = pd.merge(train, team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])

train.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

train.head()
# merge with test 

test = pd.merge(test, team_win_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'])

test = pd.merge(test, team_loss_score, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'LTeamID'])

test = pd.merge(test, team_loss_score, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'])

test = pd.merge(test, team_win_score, how='left', left_on=['Season', 'LTeamID_x'], right_on=['Season', 'WTeamID'])

test.drop(['LTeamID_y', 'WTeamID_y'], axis=1, inplace=True)

test.head()

def preprocess(df):

    df['x_score'] = df['WScore sum_x'] + df['LScore sum_y']

    df['y_score'] = df['WScore sum_y'] + df['LScore sum_x']

    df['x_count'] = df['WScore count_x'] + df['LScore count_y']

    df['y_count'] = df['WScore count_y'] + df['WScore count_x']

    df['x_var'] = df['WScore var_x'] + df['LScore count_y']

    df['y_var'] = df['WScore var_y'] + df['WScore var_x']

    return df

train = preprocess(train)

test = preprocess(test)

test.shape
# make winner and loser train

train_win = train.copy()

train_los = train.copy()

train_win = train_win[['WSeed', 'LSeed',

                 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]

train_los = train_los[['LSeed', 'WSeed', 

                 'y_score', 'x_score', 'x_count', 'y_count', 'x_var', 'y_var']]

train_win.columns = ['Seed_1', 'Seed_2',

                  'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']

train_los.columns = ['Seed_1', 'Seed_2', 

                  'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']



# same processing for test

test = test[['ID', 'WSeed', 'LSeed', 

                 'x_score', 'y_score', 'x_count', 'y_count', 'x_var', 'y_var']]

test.columns = ['ID', 'Seed_1', 'Seed_2', 

                  'Score_1', 'Score_2', 'Count_1', 'Count_2', 'Var_1', 'Var_2']
# feature enginnering

def feature_engineering(df):

    df['Seed_diff'] = df['Seed_1'] - df['Seed_2']

    df['Score_diff'] = df['Score_1'] - df['Score_2']

    df['Count_diff'] = df['Count_1'] - df['Count_2']

    df['Var_diff'] = df['Var_1'] - df['Var_2']

    df['Mean_score1'] = df['Score_1'] / df['Count_1']

    df['Mean_score2'] = df['Score_2'] / df['Count_2']

    df['Mean_score_diff'] = df['Mean_score1'] - df['Mean_score2']

    df['FanoFactor_1'] = df['Var_1'] / df['Mean_score1']

    df['FanoFactor_2'] = df['Var_2'] / df['Mean_score2']

    return df

train_win = feature_engineering(train_win)

train_los = feature_engineering(train_los)

test = feature_engineering(test)

test.shape
train_win["result"] = 1

print(train_win.shape)

train_win.head()

train_los["result"] = 0

print(train_los.shape)

train_los.head()

data = pd.concat((train_win, train_los)).reset_index(drop=True)

print(data.shape)

data.head()
test = test.drop(['ID'],axis=1)

test.head()
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

from xgboost import XGBClassifier
y_train=data['result']

X_train=data.drop(columns='result')
params_lgb = {'num_leaves': 400,

              'min_child_weight': 0.034,

              'feature_fraction': 0.379,

              'bagging_fraction': 0.418,

              'min_data_in_leaf': 106,

              'objective': 'binary',

              'max_depth': 50,

              'learning_rate': 0.0068,

              "boosting_type": "gbdt",

              "bagging_seed": 11,

              "metric": 'logloss',

              "verbosity": -1,

              'reg_alpha': 0.3899,

              'reg_lambda': 0.648,

              'random_state': 47,

              }



params_xgb = {'colsample_bytree': 0.8,                 

              'learning_rate': 0.0004,

              'max_depth': 31,

              'subsample': 1,

              'objective':'binary:logistic',

              'eval_metric':'logloss',

              'min_child_weight':3,

              'gamma':0.25,

              'n_estimators':5000

              }
NFOLDS = 10

folds = KFold(n_splits=NFOLDS)



columns = X_train.columns

splits = folds.split(X_train, y_train)
y_preds_lgb = np.zeros(test.shape[0])

y_oof_lgb = np.zeros(X_train.shape[0])

print(test.shape)

print(X_train.shape)
for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    X_train1, X_valid1 = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]

    y_train1, y_valid1 = y_train.iloc[train_index], y_train.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train1, label=y_train1)

    dvalid = lgb.Dataset(X_valid1, label=y_valid1)



    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)

    

    y_pred_valid = clf.predict(X_valid1)

    y_oof_lgb[valid_index] = y_pred_valid

    

    y_preds_lgb += clf.predict(test) / NFOLDS
y_preds_lgb.shape
submission_df = pd.read_csv(path + 'WSampleSubmissionStage1_2020.csv')

#submission_df['Pred'] = 0.94*y_preds_lgb + 0.06*y_preds_xgb

submission_df['Pred'] = y_preds_lgb

submission_df
test= pd.read_csv(path +'WSampleSubmissionStage1_2020.csv')

test.shape
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)