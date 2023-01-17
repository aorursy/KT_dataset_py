import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV
tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
# deleting unnecessary columns

tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

tourney_result
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result
def get_seed(x):

    return int(x[1:3])



tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))

tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))

tourney_result
season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]

season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

season_result
season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

season_score
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result
tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)

tourney_win_result
tourney_lose_result = tourney_win_result.copy()

tourney_lose_result['Seed1'] = tourney_win_result['Seed2']

tourney_lose_result['Seed2'] = tourney_win_result['Seed1']

tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']

tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']

tourney_lose_result
tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']

tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
tourney_win_result['result'] = 1

tourney_lose_result['result'] = 0

tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

tourney_result
test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df
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

test_df
test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))

test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test_df
X = tourney_result.drop('result', axis=1)

y = tourney_result.result
from sklearn.model_selection import KFold

import lightgbm as lgb



params = {'num_leaves': 400,

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
import gc

NFOLDS = 50

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(test_df.shape[0])

y_oof = np.zeros(X.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    print('Fold:',fold_n+1)

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    

    y_preds += clf.predict(test_df) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = y_preds

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)
import seaborn as sns

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(8, 8))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));