# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import time

import csv

# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

from sklearn import preprocessing

# Suppr warning

import warnings

warnings.filterwarnings("ignore")



import itertools

from scipy import interp

# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams

from matplotlib import cm

#import ggridges

# Any results you write to the current directory are saved as output.

import seaborn as sns

import statsmodels.api as sm
folder_path = '.../input/nhl-game-data/'

games_data = pd.read_csv(f'/kaggle/input/nhl-game-data/game.csv')

game_team_stats = pd.read_csv(f'/kaggle/input/nhl-game-data/game_teams_stats.csv')

game_goalie_stats = pd.read_csv(f'/kaggle/input/nhl-game-data/game_goalie_stats.csv')

team_info = pd.read_csv(f'/kaggle/input/nhl-game-data/team_info.csv')

game_shifts = pd.read_csv(f'/kaggle/input/nhl-game-data/game_shifts.csv')

game_plays = pd.read_csv(f'/kaggle/input/nhl-game-data/game_plays.csv')

player_info = pd.read_csv(f'/kaggle/input/nhl-game-data/player_info.csv')

game_skater_stats = pd.read_csv(f'/kaggle/input/nhl-game-data/game_skater_stats.csv')

game_plays_players = pd.read_csv(f'/kaggle/input/nhl-game-data/game_plays_players.csv')
games_data.head()
game_team_stats.head()
game_goalie_stats.head()
team_info.head()
game_shifts.head()
game_plays.head()
player_info.head()
game_skater_stats.head()
game_plays_players.head()
nhl_team_game_stats = pd.merge(game_team_stats,team_info,on='team_id', how='left')

# merged data together, but kills kernel



nhl_team_game_stats['team_id'].is_unique
nhl_team_game_stats.head(10)
games_data.get_dtype_counts()
df = pd.DataFrame(games_data)

df.head(10)
df2 = pd.DataFrame(nhl_team_game_stats)

df2.head(10)
sns.set(style='whitegrid')

axe = sns.barplot(x='abbreviation',y='goals',data=nhl_team_game_stats)

plt.xticks(rotation=90)
sns.set(style='whitegrid')

axe = sns.barplot(x='abbreviation',y='shots',data=nhl_team_game_stats)

plt.xticks(rotation=90)
nhl_team_game_stats.columns
nhl_team_game_stats.describe()
nhl_team_game_stats.isnull().any()