import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
# Input data files are available in the "../input/" directory.

import os



file_count = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    filenames.sort()

    for filename in filenames:

        print(os.path.join(dirname, filename))

        file_count += 1

print(f'Total {file_count} files!')
datadir = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament')

stage1dir = datadir/'MDataFiles_Stage1'
teams_df = pd.read_csv(stage1dir/'MTeams.csv')



print('teams_df', teams_df.shape)

teams_df.head()
tmp_df = teams_df[['TeamName', 'FirstD1Season', 'LastD1Season']].copy()

tmp_df.columns = ['Task', 'Start', 'Finish']



# Only plot first 20 teams

fig = ff.create_gantt(tmp_df.iloc[:20])

py.plot(fig, filename='gannt.html')

# fig.show()  # It causes kaggle kernel error when committed somehow...
seasons_df = pd.read_csv(stage1dir/'MSeasons.csv')
print(seasons_df.shape)

seasons_df.head()
tourney_seeds_df = pd.read_csv(stage1dir/'MNCAATourneySeeds.csv')

tourney_seeds_df
regular_season_results_df = pd.read_csv(stage1dir/'MRegularSeasonCompactResults.csv')

tournament_results_df = pd.read_csv(stage1dir/'MNCAATourneyCompactResults.csv')
regular_season_results_df.head()
tournament_results_df.head()
print('regular season', regular_season_results_df.shape, 'tournament', tournament_results_df.shape)
sample_submission = pd.read_csv(datadir/'MSampleSubmissionStage1_2020.csv')

sample_submission
regular_season_detailed_results_df = pd.read_csv(stage1dir/'MRegularSeasonDetailedResults.csv')

tournament_detailed_results_df = pd.read_csv(stage1dir/'MNCAATourneyDetailedResults.csv')
regular_season_detailed_results_df.head()
tournament_detailed_results_df.head()
print('regular', regular_season_detailed_results_df.shape, 'tournament', tournament_detailed_results_df.shape)
cities_df = pd.read_csv(stage1dir/'Cities.csv')

mgame_cities_df = pd.read_csv(stage1dir/'MGameCities.csv')
cities_df
mgame_cities_df
massey_df = pd.read_csv(stage1dir/'MMasseyOrdinals.csv')

massey_df
massey_df['SystemName'].unique()
tmp_df = massey_df[(massey_df['Season'] == 2003) & (massey_df['RankingDayNum'] == 35) & (massey_df['SystemName'] == 'SEL')][['TeamID', 'OrdinalRank']]

# Only shows first 20 teams.

tmp_df.sort_values('OrdinalRank').iloc[:20].plot(kind='barh', x='TeamID', y='OrdinalRank')
event2015_df = pd.read_csv(datadir/'MEvents2015.csv')

# event2016_df = pd.read_csv(datadir/'MEvents2016.csv')

# event2017_df = pd.read_csv(datadir/'MEvents2017.csv')

# event2018_df = pd.read_csv(datadir/'MEvents2018.csv')

# event2019_df = pd.read_csv(datadir/'MEvents2019.csv')
event2015_df.head(10)
players_df = pd.read_csv(datadir/'MPlayers.csv')

players_df
team_coaches_df = pd.read_csv(stage1dir/'MTeamCoaches.csv')



print('team_coaches_df', team_coaches_df.shape)

team_coaches_df.iloc[80:85]
conferences_df = pd.read_csv(stage1dir/'Conferences.csv')

team_conferences_df = pd.read_csv(stage1dir/'MTeamConferences.csv')
conferences_df
team_conferences_df.head()
team_conferences_df[team_conferences_df['TeamID'] == 1102]
conference_tourney_games_df = pd.read_csv(stage1dir/'MConferenceTourneyGames.csv')

conference_tourney_games_df
secondary_tourney_teams_df = pd.read_csv(stage1dir/'MSecondaryTourneyTeams.csv')

secondary_tourney_teams_df
secondary_tourney_results_df = pd.read_csv(stage1dir/'MSecondaryTourneyCompactResults.csv')

secondary_tourney_results_df
# it seems there's encoding problem... which encoding can be used to open this file?

# team_spellings_df = pd.read_csv(stage1dir/'MTeamSpellings.csv')



# encoding="cp1252" works!

team_spellings_df = pd.read_csv(stage1dir/'MTeamSpellings.csv',encoding="cp1252")
team_spellings_df.head()
teams_df.query("TeamID == 1394").TeamName
tourney_slots_df = pd.read_csv(stage1dir/'MNCAATourneySlots.csv')

tourney_seed_round_slots_df = pd.read_csv(stage1dir/'MNCAATourneySeedRoundSlots.csv')
tourney_slots_df
tourney_slots_df[(tourney_slots_df['Season'] == 1985) & (tourney_slots_df['Slot'].str.startswith('R1W'))]
tourney_seed_round_slots_df
tournament_results2015_df = tournament_results_df.query("Season >= 2015")

tournament_results2015_df
sample_submission.head()


for key, row in tournament_results2015_df.iterrows():

    if row['WTeamID'] < row['LTeamID']:

        # Check season_win_lost type

        id_name = str(row['Season']) + '_' + str(row['WTeamID']) + '_' + str(row['LTeamID'])

        sample_submission.loc[sample_submission['ID'] == id_name, 'Pred'] = 1.0

    else:

        # Check season_lost_win type

        id_name = str(row['Season']) + '_' + str(row['LTeamID']) + '_' + str(row['WTeamID'])

        sample_submission.loc[sample_submission['ID'] == id_name, 'Pred'] = 0.0
sample_submission.to_csv('submission.csv', index=False)
sample_submission['Pred'].hist()