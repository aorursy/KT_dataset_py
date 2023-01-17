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
%%time



datadir = Path('/kaggle/input/ncaa-march-madness-2020-mens')

stage1dir = datadir/'MDataFiles_Stage1'



event_2015_df = pd.read_feather(datadir / 'MEvents2015.feather')

event_2016_df = pd.read_feather(datadir / 'MEvents2016.feather')

event_2017_df = pd.read_feather(datadir / 'MEvents2017.feather')

event_2018_df = pd.read_feather(datadir / 'MEvents2018.feather')

event_2019_df = pd.read_feather(datadir / 'MEvents2019.feather')

players_df = pd.read_feather(datadir / 'MPlayers.feather')

sample_submission = pd.read_feather(datadir / 'MSampleSubmissionStage1_2020.feather')



cities_df = pd.read_feather(stage1dir / 'Cities.feather')

conferences_df = pd.read_feather(stage1dir / 'Conferences.feather')

conference_tourney_games_df = pd.read_feather(stage1dir / 'MConferenceTourneyGames.feather')

game_cities_df = pd.read_feather(stage1dir / 'MGameCities.feather')

massey_ordinals_df = pd.read_feather(stage1dir / 'MMasseyOrdinals.feather')

tourney_compact_results_df = pd.read_feather(stage1dir / 'MNCAATourneyCompactResults.feather')

tourney_detailed_results_df = pd.read_feather(stage1dir / 'MNCAATourneyDetailedResults.feather')

tourney_seed_round_slots_df = pd.read_feather(stage1dir / 'MNCAATourneySeedRoundSlots.feather')

tourney_seeds_df = pd.read_feather(stage1dir / 'MNCAATourneySeeds.feather')

tourney_slots_df = pd.read_feather(stage1dir / 'MNCAATourneySlots.feather')

regular_season_compact_results_df = pd.read_feather(stage1dir / 'MRegularSeasonCompactResults.feather')

regular_season_detailed_results_df = pd.read_feather(stage1dir / 'MRegularSeasonDetailedResults.feather')

seasons_df = pd.read_feather(stage1dir / 'MSeasons.feather')

secondary_tourney_compact_results_df = pd.read_feather(stage1dir / 'MSecondaryTourneyCompactResults.feather')

secondary_tourney_teams_df = pd.read_feather(stage1dir / 'MSecondaryTourneyTeams.feather')

team_coaches_df = pd.read_feather(stage1dir / 'MTeamCoaches.feather')

team_conferences_df = pd.read_feather(stage1dir / 'MTeamConferences.feather')

teams_df = pd.read_feather(stage1dir / 'MTeams.feather')

%%time



datadir = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament')

stage1dir = datadir/'MDataFiles_Stage1'



event_2015_df = pd.read_csv(datadir/'MEvents2015.csv')

event_2016_df = pd.read_csv(datadir/'MEvents2016.csv')

event_2017_df = pd.read_csv(datadir/'MEvents2017.csv')

event_2018_df = pd.read_csv(datadir/'MEvents2018.csv')

event_2019_df = pd.read_csv(datadir/'MEvents2019.csv')

players_df = pd.read_csv(datadir/'MPlayers.csv')

sample_submission = pd.read_csv(datadir/'MSampleSubmissionStage1_2020.csv')



cities_df = pd.read_csv(stage1dir/'Cities.csv')

conferences_df = pd.read_csv(stage1dir/'Conferences.csv')

conference_tourney_games_df = pd.read_csv(stage1dir/'MConferenceTourneyGames.csv')

game_cities_df = pd.read_csv(stage1dir/'MGameCities.csv')

massey_ordinals_df = pd.read_csv(stage1dir/'MMasseyOrdinals.csv')

tourney_compact_results_df = pd.read_csv(stage1dir/'MNCAATourneyCompactResults.csv')

tourney_detailed_results_df = pd.read_csv(stage1dir/'MNCAATourneyDetailedResults.csv')

tourney_seed_round_slots_df = pd.read_csv(stage1dir/'MNCAATourneySeedRoundSlots.csv')

tourney_seeds_df = pd.read_csv(stage1dir/'MNCAATourneySeeds.csv')

tourney_slots_df = pd.read_csv(stage1dir/'MNCAATourneySlots.csv')

regular_season_compact_results_df = pd.read_csv(stage1dir/'MRegularSeasonCompactResults.csv')

regular_season_detailed_results_df = pd.read_csv(stage1dir/'MRegularSeasonDetailedResults.csv')

seasons_df = pd.read_csv(stage1dir/'MSeasons.csv')

secondary_tourney_compact_results_df = pd.read_csv(stage1dir/'MSecondaryTourneyCompactResults.csv')

secondary_tourney_teams_df = pd.read_csv(stage1dir/'MSecondaryTourneyTeams.csv')

team_coaches_df = pd.read_csv(stage1dir/'MTeamCoaches.csv')

team_conferences_df = pd.read_csv(stage1dir/'MTeamConferences.csv')

teams_df = pd.read_csv(stage1dir/'MTeams.csv')