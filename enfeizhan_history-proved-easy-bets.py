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
rdat = pd.read_csv('../input/RegularSeasonCompactResults.csv')

tdat = pd.read_csv('../input/TourneyCompactResults.csv')

dat = pd.concat([rdat, tdat])
# only need to know who won and who lost

games = dat.loc[:, ['Wteam', 'Lteam']].copy()
# create head-to-head combination regardless of game result

games.loc[:, 'Min_team'] = games.loc[:, ['Wteam', 'Lteam']].min(axis=1)

games.loc[:, 'Max_team'] = games.loc[:, ['Wteam', 'Lteam']].max(axis=1)

games.loc[:, 'game'] = games.Min_team.apply(str) + ', ' + games.Max_team.apply(str)
games.head()
# create head-to-head combination that contains info of game results

games.loc[:, 'win_lose'] = games.Wteam.apply(str) + ', ' + games.Lteam.apply(str)
games.head()
gpd_win = games.groupby(['game', 'win_lose'])

gpd_game = games.groupby('game')
game_counts = gpd_game.size()

win_counts = gpd_win.size()
game_counts.tail()
win_counts.tail()
min_ncontests = 20

win_rate = win_counts.div(game_counts.loc[game_counts >= min_ncontests], level=0).dropna() * 100
win_rate.tail()
always_win = win_rate.loc[win_rate >= 100]

always_win.name = 'percentage'

always_win = always_win.reset_index()

always_win
always_win.loc[:, 'Wteam'] = always_win.win_lose.str.extract(r'(\d+), \d+', expand=False)

always_win.loc[:, 'Lteam'] = always_win.win_lose.str.extract(r'\d+, (\d+)', expand=False)
always_win
teams = pd.read_csv('../input/Teams.csv').set_index('Team_Id')

teams.head()
always_win.loc[:, 'Wteam'] = always_win.Wteam.astype('int').map(teams.Team_Name)

always_win.loc[:, 'Lteam'] = always_win.Lteam.astype('int').map(teams.Team_Name)
always_win