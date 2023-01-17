# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
player = pd.read_csv('../input/atp-tennis/Player.csv')

stats = pd.read_csv('../input/atp-tennis/Stats.csv')

match = pd.read_csv('../input/atp-tennis/Match.csv')
player.head()
stats.head()
match.tail()
player.count()
player.isna().any()
player_subset = player

player_subset.dropna(subset=['hand'], how='all', inplace = True)
player_subset.isna().any()
player_subset.count()
sns.countplot(x='hand',data=player_subset)
#Most number of players from countries

#player.groupby('country',as_index=False,sort=False)['name'].count().head()

player['country'].value_counts().head()
#18 countries with only 1 player

player['country'].value_counts().tail(20)
match.isna().any()
match.count()
match_subset = match

match_subset.dropna(subset=['match_minutes'], how='all', inplace = True)
#Data for 4 tournaments available

match['tournament'].unique()
match.groupby('tournament')['match_minutes'].mean()
#All the tournaments follow similar format.

match.groupby(['tournament','round'],as_index=False)['match_id'].count()
#Win-Loss count for the player

def WinLoss (player_name):

    player_id = player.loc[player['name'] == player_name]['player_id'].iloc[0]

    return stats.loc[stats['player_id'] == player_id]['winner'].value_counts()
WinLoss('Roger Federer')
WinLoss('Rafael Nadal')
#Number of aces hit by the player 

def AcesInLife (player_name):

    player_id = player.loc[player['name'] == player_name]['player_id'].iloc[0]

    return stats.loc[stats['player_id'] == player_id]['aces'].sum()
AcesInLife('Roger Federer')
AcesInLife('Rafael Nadal')
#According to the dataset, first game and latest game a player played



def firstAndLastMatch (player_name):

    AllDates = []

    player_id = player.loc[player['name'] == player_name]['player_id'].iloc[0]

    match_ids = stats.loc[stats['player_id'] == player_id]['match_id'].values.tolist()

    for match_id in match_ids:

        dateOfTheMatch = match.loc[match['match_id'] == match_id]['date'].iloc[0]

        AllDates.append(dateOfTheMatch)

    print(f'Earliest Game in the dataset: {min(AllDates)} and Latest Game in the dataset: {max(AllDates)}')
firstAndLastMatch('Roger Federer')
firstAndLastMatch('Rafael Nadal')
firstAndLastMatch('Andre Agassi')
def Winner(year,tournament):

    match_id = match.loc[(match['year'] == year) & (match['tournament'] == tournament) & (match['round'] == 'The Final')]['match_id'].iloc[0]

    player_id = stats.loc[(stats['match_id'] == match_id) & (stats['winner'] == True)]['player_id'].iloc[0]

    player_name = player.loc[(player['player_id'] == player_id)]['name'].iloc[0]

    print(player_name)
Winner(2000,'Wimbledon')
Winner(2017,'French Open')
def HeadToHead (playerOne,playerTwo):

    player_id_one = player.loc[player['name'] == playerOne]['player_id'].iloc[0]

    player_id_two = player.loc[player['name'] == playerTwo]['player_id'].iloc[0]

    match_ids_one = stats.loc[stats['player_id'] == player_id_one]['match_id'].values.tolist()

    match_ids_two = stats.loc[stats['player_id'] == player_id_two]['match_id'].values.tolist()

    match_ids_common = set(match_ids_one) & set(match_ids_two)

    if(len(match_ids_common) == 0):

        print("Did not play matches against each other")

    else:

        winner_1 = 0

        winner_2 = 0

        for match_id in match_ids_common:

            winner_id = stats.loc[(stats['match_id'] == match_id) & stats['winner'] == True]['player_id'].iloc[0]

            if(winner_id == player_id_one):

                winner_1 = winner_1+1

            elif(winner_id == player_id_two):

                winner_2 = winner_2+1

        print(f" {playerOne}: {winner_1} & {playerTwo}: {winner_2}")
HeadToHead('Roger Federer','Giuseppe Merlo')
HeadToHead('Roger Federer','Rafael Nadal')