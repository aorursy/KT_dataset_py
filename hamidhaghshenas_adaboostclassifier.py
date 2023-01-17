import numpy as np

import pandas as pd



from sklearn.metrics import log_loss

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import AdaBoostClassifier
teams = pd.read_csv('../input/wdatafiles/WTeams.csv')

teams2 = pd.read_csv('../input/wdatafiles/WTeamSpellings.csv', encoding='latin-1')

season_cresults = pd.read_csv('../input/wdatafiles/WRegularSeasonCompactResults.csv')

season_dresults = pd.read_csv('../input/wdatafiles/WRegularSeasonDetailedResults.csv')

tourney_cresults = pd.read_csv('../input/wdatafiles/WNCAATourneyCompactResults.csv')

tourney_dresults = pd.read_csv('../input/wdatafiles/WNCAATourneyDetailedResults.csv')

slots = pd.read_csv('../input/wdatafiles/WNCAATourneySlots.csv')

seeds = pd.read_csv('../input/wdatafiles/WNCAATourneySeeds.csv')

seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}

seeds = {**seeds, **{k.replace('2018_','2019_'):seeds[k] for k in seeds if '2018_' in k}}

cities = pd.read_csv('../input/wdatafiles/WCities.csv')

gcities = pd.read_csv('../input/wdatafiles/WGameCities.csv')

seasons = pd.read_csv('../input/wdatafiles/WSeasons.csv')

sub = pd.read_csv('../input/WSampleSubmissionStage1.csv')
teams2 = teams2.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()

teams2.columns = ['TeamID', 'TeamNameCount']

teams = pd.merge(teams, teams2, how='left', on=['TeamID'])

del teams2
season_cresults['ST'] = 'S'

season_dresults['ST'] = 'S'

tourney_cresults['ST'] = 'T'

tourney_dresults['ST'] = 'T'



games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)

games.reset_index(drop=True, inplace=True)

games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)

games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)

games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)

games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)

games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)

games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)
games['ScoreDiff'] = games['WScore'] - games['LScore']

games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)

games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)

games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed'] 

games = games.fillna(-1)
#Competitiveness, includes more game options - overfitting for now in Tournaments

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',

 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',

 'LBlk', 'LPF']

c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']

gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()

gb.columns = [''.join(c) + '_c_score' for c in gb.columns]



#for now

games = games[games['ST']=='T']
sub['WLoc'] = 3

sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])

sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])

sub['Season'] = sub['Season'].astype(int)

sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])

sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])

sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)

sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)

sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)

sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] 

sub = sub.fillna(-1)
games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col]
model = AdaBoostClassifier(n_estimators=200, learning_rate=1.4)

model.fit(games[col].fillna(-1), games['Pred'])

predictions = model.predict(games[col].fillna(-1)).clip(0,1)

print('Log Loss:', log_loss(games['Pred'], predictions))
sub['Pred'] = model.predict(sub[col].fillna(-1)).clip(0,1)

sub[['ID', 'Pred']].to_csv('submission.csv', index=False)