# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
teams = pd.read_csv('../input/Teams.csv')

regular_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
frames = []

for season in regular_season.Season.unique():

    team_season = teams.copy()

    team_season['season'] = season

    team_season.index = team_season.Team_Id.apply(str) + team_season.season.apply(str)

    frames.append(team_season)



team_seasons = pd.concat(frames)

print(team_seasons.head())
loser_columns = ['Lscore', 'Lfgm', 'Lfga', 'Lfgm3', 'Lfga3','Lftm', 'Lfta', 'Lor', 'Ldr', 'Last', 'Lto', 'Lstl', 'Lblk', 'Lpf']

winner_columns = ['Wscore', 'Wfgm', 'Wfga', 'Wfgm3', 'Wfga3', 'Wftm', 'Wfta', 'Wor', 'Wdr', 'Wast', 'Wto', 'Wstl', 'Wblk', 'Wpf']

fixed_columns = [column[1:] for column in winner_columns]
season_averages = []

for index, row in team_seasons.iterrows():

    wins = regular_season.Wteam == row.Team_Id

    loses = regular_season.Lteam == row.Team_Id

    right_season = regular_season.Season == row.season



    won_games = regular_season[right_season & wins][winner_columns]

    won_games.columns = fixed_columns



    lost_games = regular_season[right_season & loses][loser_columns]

    lost_games.columns = fixed_columns



    games = pd.concat([won_games, lost_games]).mean()

    games['Team_Id'] = row.Team_Id

    games['Season'] = row.season

    games['Team_Name'] = row.Team_Name

    games = games.to_frame().transpose()



    season_averages.append(games)



season_averages = pd.concat(season_averages).dropna()
season_averages.head()