import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import eli5



from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

import lightgbm as lgb

import xgboost as xgb



import gc



import warnings

warnings.filterwarnings("ignore")
men_folder_path = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'

women_folder_path = '../kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'

Mstage1_folder_path = 'MDataFiles_Stage1/'

Wstage1_folder_path = 'WDataFiles_Stage1/'
Mfolder_path = men_folder_path

Mnacc_section1 = pd.read_csv(Mfolder_path+Mstage1_folder_path+'M'+'NCAATourneyCompactResults.csv')

Mseed_section1 = pd.read_csv(Mfolder_path+Mstage1_folder_path+'M'+'NCAATourneySeeds.csv')
Mnacc_section1.head()
Mseed_section1.head()
Mnacc_section1 = Mnacc_section1.drop(['WScore', 'LScore', 'WLoc', 'NumOT'], axis = 1)

Mnacc_section1 = Mnacc_section1.merge(Mseed_section1,left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

Mnacc_section1 = Mnacc_section1.merge(Mseed_section1,left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'),how='left').drop('TeamID',axis = 1)
Mnacc_section1.head()
Mnacc_section1['Region_W'] = Mnacc_section1['Seed_W'].map(lambda x: 'Region' + str(x[0]))

Mnacc_section1['Region_L'] = Mnacc_section1['Seed_L'].map(lambda x: 'Region' + str(x[0]))

Mnacc_section1['Seed_W'] = Mnacc_section1['Seed_W'].map(lambda x: int(x[1:3]))

Mnacc_section1['Seed_L'] = Mnacc_section1['Seed_L'].map(lambda x: int(x[1:3]))
Mseason_section1 = pd.read_csv(Mfolder_path+Mstage1_folder_path+'M'+'Seasons.csv')

Mseason_section1.head()
Mnacc_section1 = Mnacc_section1.merge(Mseason_section1, left_on=['Season'], right_on=['Season']).drop(['DayZero'], axis = 1)
Mnacc_section1.head()
Mnacc_section1['Region_W'] = Mnacc_section1.apply(lambda row : row[row['Region_W']], axis = 1)

Mnacc_section1['Region_L'] = Mnacc_section1.apply(lambda row : row[row['Region_L']], axis = 1)

Mnacc_section1 = Mnacc_section1.drop(['RegionW','RegionX', 'RegionY', 'RegionZ'], axis = 1)
Mnacc_section1.head()
Mnacc_section1.loc[:,'Region_W'].value_counts().sum()
#Mnacc_section1.query('DayNum == 154').groupby('Region_W')['Season'].count()

team2_W = Mnacc_section1.loc[(Mnacc_section1['DayNum'] == 154), 'Region_W'].value_counts()

team2_L = Mnacc_section1.loc[(Mnacc_section1['DayNum'] == 154), 'Region_L'].value_counts()

Mnacc_section1.loc[:,'Region_W'] = Mnacc_section1.loc[:,'Region_W'].map(team2_W)

Mnacc_section1.loc[:,'Region_L'] = Mnacc_section1.loc[:,'Region_L'].map(team2_L)

#np.where(Mnacc_section1.loc[:,'Region_W'].isnull())

Mnacc_section1 = Mnacc_section1.replace(np.nan,0)
Mnacc_section1.Region_W.value_counts().sum()
Mnacc_section1.head()
#每个球队每个赛季常规赛的总得分

Mregular_section1 = pd.read_csv(Mfolder_path+Mstage1_folder_path+'M'+'RegularSeasonCompactResults.csv')

Mregular_section1_win = Mregular_section1[['Season', 'WTeamID', 'WScore']]

Mregular_section1_lose = Mregular_section1[['Season', 'LTeamID', 'LScore']]
Mregular_section1_win.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

Mregular_section1_lose.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

Mregular_section1 = pd.concat((Mregular_section1_win, Mregular_section1_lose)).reset_index(drop=True)
Mseason_score = Mregular_section1.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
Mregular_section1.shape
Mregular_section1_win.shape
Mregular_section1_lose.shape
Mseason_score = Mregular_section1.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

Mseason_score_win = Mregular_section1_win.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

Mseason_score_lose = Mregular_section1_lose.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

Mseason_score_win.rename(columns={'Score': 'Score_W'}, inplace=True)

Mseason_score_lose.rename(columns={'Score': 'Score_L'}, inplace=True)

Mseason_score = Mseason_score.merge(Mseason_score_win, left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID'], how = 'outer')

Mseason_score = Mseason_score.merge(Mseason_score_lose, left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID'], how = 'outer')
Mseason_score = Mseason_score.replace(np.nan, 0)
Mseason_count  = Mregular_section1.groupby(['Season', 'TeamID']).count().reset_index()

Mseason_count_win  = Mregular_section1_win.groupby(['Season', 'TeamID']).count().reset_index()

Mseason_count_lose = Mregular_section1_lose.groupby(['Season', 'TeamID']).count().reset_index()
Mseason_count.rename(columns={'Score': 'Count'}, inplace=True)

Mseason_count_win.rename(columns={'Score': 'Count_W'}, inplace=True)

Mseason_count_lose.rename(columns={'Score': 'Count_L'}, inplace=True)

Mseason_count = Mseason_count.merge(Mseason_count_win, left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID'], how = 'outer')

Mseason_count = Mseason_count.merge(Mseason_count_lose, left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID'], how = 'outer')
Mseason_count = Mseason_count.replace(np.nan, 0)
Mnacc_section1 = Mnacc_section1.merge(Mseason_score, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

Mnacc_section1 = Mnacc_section1.merge(Mseason_score, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'), how='left').drop('TeamID',axis = 1)
Mnacc_section1 = Mnacc_section1.merge(Mseason_count, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

Mnacc_section1 = Mnacc_section1.merge(Mseason_count, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'), how='left').drop('TeamID',axis = 1)
#构造result为1的训练集

train_win = Mnacc_section1.copy()

train_win = train_win.drop(['Season', 'WTeamID', 'LTeamID', 'DayNum'], axis=1)

train_win.rename(columns={'Seed_W':'Seed1', 'Seed_L':'Seed2', 'Score_W':'Score1', 'Score_L':'Score2', 'Region_W': 'Region1', 'Region_L': 'Region2', 'Count_W': 'Count1', 'Count_L': 'Count2'}, inplace=True)

train_win['result'] = 1
#Score_W_W  意味着第一支队伍赢球场次的总分

#Score_L_W  意味着第一支队伍输球场次的总分

train_lose = train_win.copy()

train_lose['Seed1'] = train_win['Seed2']

train_lose['Seed2'] = train_win['Seed1']

train_lose['Score1'] = train_win['Score2']

train_lose['Score2'] = train_win['Score1']

train_lose['Region1'] = train_win['Region2']

train_lose['Region2'] = train_win['Region1']

train_lose['Score_W_L'] = train_win['Score_W_W']

train_lose['Score_L_L'] = train_win['Score_L_W']

train_lose['Score_W_W'] = train_win['Score_W_L']

train_lose['Score_L_W'] = train_win['Score_L_L']

train_lose['Count_W_L'] = train_win['Count_W_W']

train_lose['Count_W_W'] = train_win['Count_W_L']

train_lose['Count_L_L'] = train_win['Count_L_W']

train_lose['Count_L_W'] = train_win['Count_L_L']

train_lose['result'] = 0

train_lose.head()
train_df = pd.concat((train_win,train_lose)).reset_index(drop = True)

train_df['Seed_diff'] = train_df['Seed1'] - train_df['Seed2'] 

train_df['Score_diff'] = train_df['Score1'] - train_df['Score2'] 

train_df['Count_diff'] = train_df['Count1'] - train_df['Count2'] 

train_df.head()
test_df = pd.read_csv(Mfolder_path+'M'+'SampleSubmissionStage1_2020.csv')
test_df['ID'][0]
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
test_df = test_df.merge(Mseed_section1,left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

test_df = test_df.merge(Mseed_section1,left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'),how='left').drop('TeamID',axis = 1)
test_df['Region_W'] = test_df['Seed_W'].map(lambda x: 'Region' + str(x[0]))

test_df['Region_L'] = test_df['Seed_L'].map(lambda x: 'Region' + str(x[0]))

test_df['Seed_W'] = test_df['Seed_W'].map(lambda x: int(x[1:3]))

test_df['Seed_L'] = test_df['Seed_L'].map(lambda x: int(x[1:3]))
test_df = test_df.merge(Mseason_section1, left_on=['Season'], right_on=['Season']).drop(['DayZero'], axis = 1)

test_df['Region_W'] = test_df.apply(lambda row : row[row['Region_W']], axis = 1)

test_df['Region_L'] = test_df.apply(lambda row : row[row['Region_L']], axis = 1)

test_df = test_df.drop(['RegionW','RegionX', 'RegionY', 'RegionZ'], axis = 1)
test_df.loc[:,'Region_W'] = test_df.loc[:,'Region_W'].map(team2_W)

test_df.loc[:,'Region_L'] = test_df.loc[:,'Region_L'].map(team2_L)

test_df = test_df.replace(np.nan,0)
test_df = test_df.merge(Mseason_score, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

test_df = test_df.merge(Mseason_score, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'), how='left').drop('TeamID',axis = 1)
test_df = test_df.merge(Mseason_count, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how='left').drop('TeamID',axis = 1)

test_df = test_df.merge(Mseason_count, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], suffixes=('_W','_L'), how='left').drop('TeamID',axis = 1)
test_df.rename(columns={'Seed_W':'Seed1', 'Seed_L':'Seed2', 'Score_W':'Score1', 'Score_L':'Score2', 'Count_W': 'Count1', 'Count_L': 'Count2','Region_W':'Region1', 'Region_L':'Region2'}, inplace=True)

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2'] 

test_df['Score_diff'] = test_df['Score1'] - test_df['Score2'] 

test_df['Count_diff'] = test_df['Count1'] - test_df['Count2'] 

#test_df = test_df.drop(['Score1', 'Score2', 'Count1', 'Count2'], axis = 1)
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
train_df.shape
test_df.shape
train_df.columns
test_df.columns
X = train_df.drop('result', axis=1)

y = train_df.result
params_lgb = {'num_leaves': 70,

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
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = y_preds_lgb

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)