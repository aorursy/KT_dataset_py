# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from functools import reduce



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

input_dir = '/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'

stage1_dir = input_dir + 'MDataFiles_Stage1/'



teams = pd.read_csv(stage1_dir + "MTeams.csv")

seasons = pd.read_csv(stage1_dir + "MSeasons.csv")

seeds = pd.read_csv(stage1_dir + "MNCAATourneySeeds.csv")

regular_res = pd.read_csv(stage1_dir + "MRegularSeasonCompactResults.csv")

bracket_res = pd.read_csv(stage1_dir + "MNCAATourneyCompactResults.csv")

sample_sub = pd.read_csv(input_dir + "MSampleSubmissionStage1_2020.csv")

regular_detail = pd.read_csv(stage1_dir + "MRegularSeasonDetailedResults.csv")

bracket_detail = pd.read_csv(stage1_dir + "MNCAATourneyDetailedResults.csv")

ordinals = pd.read_csv(stage1_dir + "MMasseyOrdinals.csv")
regular_detail.columns
regular_detail['WFGP'] = regular_detail['WFGM'] / regular_detail['WFGA']

regular_detail['LFGP'] = regular_detail['LFGM'] / regular_detail['LFGA']

regular_detail['WFTP'] = regular_detail['WFTM'] / regular_detail['WFTA']

regular_detail['LFTP'] = regular_detail['LFTM'] / regular_detail['LFTA']

regular_detail['WFGP3'] = regular_detail['WFGM3'] / regular_detail['WFGA3']

regular_detail['LFGP3'] = regular_detail['LFGM3'] / regular_detail['LFGA3']
all_games = regular_res.merge(regular_detail, how='inner', on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], suffixes=('', '_y'))

all_games.drop(columns=['LScore_y', 'WScore_y', 'NumOT_y', 'WLoc_y'], inplace=True)

all_games.head()



stats = ['Score', 'FTM', 'FGM3', 'FGM', 'TO', 'OR', 'DR', 'Stl', 'Ast', 'Blk', 'FGP', 'FTP', 'FGP3']

_stat_abbrevs_overrides = {'Score': 'PPG'}

stat_abbrevs1 = {stat: stat for stat in stats}

stat_abbrevs2 = {'Opp'+stat: 'O'+stat for stat in stats}

stat_abbrevs = {**stat_abbrevs1, **stat_abbrevs2}

stat_abbrevs_overrides1 = {key: _stat_abbrevs_overrides[key] for key in _stat_abbrevs_overrides}

stat_abbrevs_overrides2 = {'Opp'+key: 'O'+_stat_abbrevs_overrides[key] for key in _stat_abbrevs_overrides}

stat_abbrevs_overrides = {**stat_abbrevs_overrides1, **stat_abbrevs_overrides2}

stat_abbrevs = {**stat_abbrevs, **stat_abbrevs_overrides}



final_stats_cols = reduce(lambda a,b: a+b, [[stat, 'Opp'+stat] for stat in stats])

stats_cols = reduce(lambda a,b: a+b, [['W'+stat, 'L'+stat] for stat in stats])

stats_win_map = {'W'+stat: stat for stat in stats}

stats_win_map_opp = {'L'+stat: 'Opp'+stat for stat in stats}

stats_loss_map = {'L'+stat: stat for stat in stats}

stats_loss_map_opp = {'W'+stat: 'Opp'+stat for stat in stats}



stat_rename_win = {s: stat_abbrevs[s]+'1' for s in stat_abbrevs}

stat_rename_loss = {s: stat_abbrevs[s]+'2' for s in stat_abbrevs}



wins = all_games[['Season', 'WTeamID'] + stats_cols]

wins.rename(columns={'WTeamID': 'TeamID', **stats_win_map, **stats_win_map_opp}, inplace=True)



losses = all_games[['Season', 'LTeamID'] + stats_cols]

losses.rename(columns={'LTeamID': 'TeamID', **stats_loss_map, **stats_loss_map_opp}, inplace=True)

reg_games = pd.concat((wins, losses), sort=True).reset_index(drop=True)

teams_group = reg_games.groupby(['Season', 'TeamID'])



# TODO support other aggregation methods

season_stats = teams_group[final_stats_cols].median().reset_index()

season_stats
tourney_res = bracket_res.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)

tourney_res = tourney_res.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_res = tourney_res.rename(columns={'Seed': 'WSeed'})

tourney_res = tourney_res.drop('TeamID', axis=1)

tourney_res = tourney_res.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_res = tourney_res.rename(columns={'Seed': 'LSeed'})

tourney_res = tourney_res.drop('TeamID', axis=1)



tourney_res['WSeed'] = tourney_res['WSeed'].map(lambda x: int(x[1:3]))

tourney_res['LSeed'] = tourney_res['LSeed'].map(lambda x: int(x[1:3]))
tourney_wins_stats = tourney_res.merge(season_stats[['Season', 'TeamID'] + list(stat_rename_win.keys())], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_wins_stats.rename(columns=stat_rename_win, inplace=True)

tourney_wins_stats = tourney_wins_stats.drop('TeamID', axis=1)

tourney_wins_stats = tourney_wins_stats[tourney_wins_stats[list(stat_rename_win.values())[0]].notnull()]

tourney_loss_stats = tourney_res.merge(season_stats[['Season', 'TeamID'] + list(stat_rename_loss.keys())], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_loss_stats.rename(columns=stat_rename_loss, inplace=True)

tourney_loss = tourney_loss_stats.drop('TeamID', axis=1)

tourney_loss_stats = tourney_loss_stats[tourney_loss_stats[list(stat_rename_loss.values())[0]].notnull()]

tourney_res = tourney_wins_stats.merge(tourney_loss_stats, on=['Season', 'WTeamID', 'LTeamID', 'WSeed', 'LSeed'])

tourney_res = tourney_res.drop('TeamID', axis=1)

tourney_res
tourney_wins = tourney_res.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

tourney_wins.rename(columns={'WSeed': 'Seed1', 'LSeed': 'Seed2'}, inplace=True)



tourney_loss = tourney_wins.copy()

tourney_loss['Seed1'] = tourney_wins['Seed2']

tourney_loss['Seed2'] = tourney_wins['Seed1']

for stat in stat_abbrevs.values():

    tourney_loss[stat+'1'] = tourney_wins[stat+'2']

    tourney_loss[stat+'2'] = tourney_wins[stat+'1']



tourney_wins['outcome'] = 1

tourney_loss['outcome'] = 0

train = pd.concat((tourney_wins, tourney_loss), sort=True).reset_index(drop=True)

train
test_df = sample_sub.copy()

test_df['Season'] = sample_sub['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = sample_sub['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = sample_sub['ID'].map(lambda x: int(x[10:14]))



test_df = test_df.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df = test_df.rename(columns={'Seed': 'Seed1'})

test_df = test_df.drop('TeamID', axis=1)

test_df = test_df.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df = test_df.rename(columns={'Seed': 'Seed2'})

test_df = test_df.drop('TeamID', axis=1)



test_df = test_df.merge(season_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns=stat_rename_win, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = test_df.merge(season_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns=stat_rename_loss, inplace=True)

test_df = test_df.drop('TeamID', axis=1)



test_df['Seed1'] = test_df['Seed1'].map(lambda x: int(x[1:3]))

test_df['Seed2'] = test_df['Seed2'].map(lambda x: int(x[1:3]))

test = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test
X = train.drop('outcome', axis=1)

y = train.outcome
from sklearn import preprocessing

df = pd.concat([X, test], axis=0, sort=False).reset_index(drop=True)

df_log = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(df),

    columns=df.columns,

    index=df.index

)

train_log, test_log = df_log.iloc[:len(X), :], df_log.iloc[len(X):,:].reset_index(drop=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# scaled

#clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.75, random_state=2).fit(train_log, y)

clf = LogisticRegression(penalty='l1', solver='saga', random_state=2).fit(train_log, y)

probs = clf.predict_proba(test_log)



team1_probs = [x[1] for x in probs]



res_df_orig = pd.DataFrame({'Season': test_df['Season'], 'Team1ID': test_df['WTeamID'], 'Team2ID': test_df['LTeamID'], 'Pred': team1_probs})

res_df_orig['ID'] = res_df_orig.apply(lambda row: str(int(row['Season'])) + '_' + str(int(row['Team1ID'])) + '_' + str(int(row['Team2ID'])), axis=1)

res_df = res_df_orig[['ID', 'Pred']]

res_df

res_df.to_csv('submission.csv', index=False)

res_df['Pred'].hist()
from sklearn.metrics import log_loss



check_df = bracket_res[bracket_res['Season'] >= 2015].merge(res_df_orig, right_on=['Season', 'Team1ID', 'Team2ID'], left_on=['Season', 'WTeamID', 'LTeamID'], how='left')

check_df2 = bracket_res[bracket_res['Season'] >= 2015].merge(res_df_orig, right_on=['Season', 'Team1ID', 'Team2ID'], left_on=['Season', 'LTeamID', 'WTeamID'], how='left')



check_df = check_df[check_df['Team1ID'].notnull()]

check_df['outcome'] = 1

check_df2 = check_df2[check_df2['Team1ID'].notnull()]

check_df2['outcome'] = 0

checker = pd.concat((check_df, check_df2), sort=True)

checker.drop(['DayNum', 'LScore', 'NumOT', 'WLoc', 'WScore'], axis=1, inplace=True)

log_loss(checker['outcome'], checker['Pred'])
i = 0

for col in train_log.columns:

    print(col, ': ', clf.coef_[0][i], '%')

    i += 1