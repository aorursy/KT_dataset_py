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
!ls /kaggle/input
%%time



datadir = Path('/kaggle/input/ncaa-march-madness-2020-womens')

stage1dir = datadir / 'WDataFiles_Stage1'



# --- read data ---

event_2015_df = pd.read_feather(datadir / 'WEvents2015.feather')

event_2016_df = pd.read_feather(datadir / 'WEvents2016.feather')

event_2017_df = pd.read_feather(datadir / 'WEvents2017.feather')

event_2018_df = pd.read_feather(datadir / 'WEvents2018.feather')

event_2019_df = pd.read_feather(datadir / 'WEvents2019.feather')

players_df = pd.read_feather(datadir / 'WPlayers.feather')

sample_submission = pd.read_feather(datadir / 'WSampleSubmissionStage1_2020.feather')



cities_df = pd.read_feather(stage1dir / 'Cities.feather')

conferences_df = pd.read_feather(stage1dir / 'Conferences.feather')

# conference_tourney_games_df = pd.read_feather(stage1dir / 'WConferenceTourneyGames.feather')

game_cities_df = pd.read_feather(stage1dir / 'WGameCities.feather')

# massey_ordinals_df = pd.read_feather(stage1dir / 'WMasseyOrdinals.feather')

tourney_compact_results_df = pd.read_feather(stage1dir / 'WNCAATourneyCompactResults.feather')

tourney_detailed_results_df = pd.read_feather(stage1dir / 'WNCAATourneyDetailedResults.feather')

# tourney_seed_round_slots_df = pd.read_feather(stage1dir / 'WNCAATourneySeedRoundSlots.feather')

tourney_seeds_df = pd.read_feather(stage1dir / 'WNCAATourneySeeds.feather')

tourney_slots_df = pd.read_feather(stage1dir / 'WNCAATourneySlots.feather')

regular_season_compact_results_df = pd.read_feather(stage1dir / 'WRegularSeasonCompactResults.feather')

regular_season_detailed_results_df = pd.read_feather(stage1dir / 'WRegularSeasonDetailedResults.feather')

seasons_df = pd.read_feather(stage1dir / 'WSeasons.feather')

# secondary_tourney_compact_results_df = pd.read_feather(stage1dir / 'WSecondaryTourneyCompactResults.feather')

# secondary_tourney_teams_df = pd.read_feather(stage1dir / 'WSecondaryTourneyTeams.feather')

# team_coaches_df = pd.read_feather(stage1dir / 'WTeamCoaches.feather')

team_conferences_df = pd.read_feather(stage1dir / 'WTeamConferences.feather')

teams_df = pd.read_feather(stage1dir / 'WTeams.feather')

%%time



datadir = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament')

stage1dir = datadir/'WDataFiles_Stage1'



event_2015_df = pd.read_csv(datadir / 'WEvents2015.csv')

event_2016_df = pd.read_csv(datadir / 'WEvents2016.csv')

event_2017_df = pd.read_csv(datadir / 'WEvents2017.csv')

event_2018_df = pd.read_csv(datadir / 'WEvents2018.csv')

event_2019_df = pd.read_csv(datadir / 'WEvents2019.csv')

players_df = pd.read_csv(datadir / 'WPlayers.csv')

sample_submission = pd.read_csv(datadir / 'WSampleSubmissionStage1_2020.csv')



cities_df = pd.read_csv(stage1dir / 'Cities.csv')

conferences_df = pd.read_csv(stage1dir / 'Conferences.csv')

# conference_tourney_games_df = pd.read_csv(stage1dir / 'WConferenceTourneyGames.csv')

game_cities_df = pd.read_csv(stage1dir / 'WGameCities.csv')

# massey_ordinals_df = pd.read_csv(stage1dir / 'WMasseyOrdinals.csv')

tourney_compact_results_df = pd.read_csv(stage1dir / 'WNCAATourneyCompactResults.csv')

tourney_detailed_results_df = pd.read_csv(stage1dir / 'WNCAATourneyDetailedResults.csv')

# tourney_seed_round_slots_df = pd.read_csv(stage1dir / 'WNCAATourneySeedRoundSlots.csv')

tourney_seeds_df = pd.read_csv(stage1dir / 'WNCAATourneySeeds.csv')

tourney_slots_df = pd.read_csv(stage1dir / 'WNCAATourneySlots.csv')

regular_season_compact_results_df = pd.read_csv(stage1dir / 'WRegularSeasonCompactResults.csv')

regular_season_detailed_results_df = pd.read_csv(stage1dir / 'WRegularSeasonDetailedResults.csv')

seasons_df = pd.read_csv(stage1dir / 'WSeasons.csv')

# secondary_tourney_compact_results_df = pd.read_csv(stage1dir / 'WSecondaryTourneyCompactResults.csv')

# secondary_tourney_teams_df = pd.read_csv(stage1dir / 'WSecondaryTourneyTeams.csv')

# team_coaches_df = pd.read_csv(stage1dir / 'WTeamCoaches.csv')

team_conferences_df = pd.read_csv(stage1dir / 'WTeamConferences.csv')

teams_df = pd.read_csv(stage1dir / 'WTeams.csv')
