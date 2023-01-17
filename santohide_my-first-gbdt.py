import pandas as pd

import numpy as np

#lightgbm

import lightgbm as lgb
tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

tourney_result

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result
# Get String

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

#WScoreT is Score in this year
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
# data split(Kfold)

# パラメタチューニングはベイズ最適化hyperoptなどを用いて行える.

#　まずはCVで平均を提出する

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV
grid_param ={'n_estimators':[100,500,1000,2000],

             'max_depth':[4,8,16,32,-1],

             'num_leaves': [8,16,31],

             'eta':[0.2,0.1,0.05,0.01],

            }



X = tourney_result.drop("result", axis = 1)

y = tourney_result.result



fit_params={'early_stopping_rounds':50, 

            'eval_set' : [(X, y)]

           }





#各々の値の設定

scores = []

bst =  lgb.LGBMClassifier(boosting_type = "gbdt",

    objective="binary")
# To view the default model params:

bst.get_params().keys()
bst_gs_cv = GridSearchCV(estimator=bst, 

                   param_grid=grid_param, 

                   scoring='neg_log_loss',

                   cv=KFold(n_splits = 5, shuffle = True, random_state = 50), 

                   n_jobs=-1,

                   verbose=2,

                   )

bst_gs_cv.fit(

            X, 

            y,

            **fit_params,

            verbose = 0

            )



best_param = bst_gs_cv.best_params_
best_param
#反映させる

kf = KFold(n_splits = 5, shuffle = True, random_state = 50)

params = {"objective": "binary", 

          "seed": 50, 

          "verbose": 0, 

          "metrics":"binary_logloss",

          "eta": 0.2,

          "max_depth": 4,

          "num_leaves": 8}

num_round = 100
sub_preds = np.zeros(test_df.shape[0])
#KFold(n_splits = 5)を用いてデータ分ける

#モデルの作成

for tr_id, va_id in kf.split(X):

    tr_x, va_x  = X.iloc[tr_id], X.iloc[va_id]

    tr_y, va_y  = y.iloc[tr_id], y.iloc[va_id]

    lgb_train = lgb.Dataset(tr_x,tr_y)

    lgb_eval = lgb.Dataset(va_x,va_y)

    model = lgb.train(params, lgb_train, num_boost_round = num_round,

                                 valid_names = ["train","valid"],valid_sets = [lgb_train, lgb_eval],

                     early_stopping_rounds=50)

    

    #スコア確認

    va_pred = model.predict(va_x)

    score = log_loss(va_y, va_pred)

    scores.append(score)

    sub_preds += model.predict(test_df)

    

sub_preds /= 5

print(f"logloss:{np.mean(scores):.4f}")
# to submit

submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = sub_preds

submission_df
submission_df['Pred'].hist()
submission_df.to_csv('submission.csv', index=False)