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
events2015 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2015.csv')

events2015.head()
events2015['EventType'].value_counts().index
events2015.loc[events2015['EventType'] == 'sub']
players = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')

teams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv')



players.shape
players.head()
teams.head()
!pip install sportsreference
from sportsreference.ncaab.roster import Player

from sportsreference.ncaab.teams import Teams
temp_player_df = players.loc[players['FirstName'] == 'Karl-Anthony'].loc[players['LastName'] == 'Towns'].iloc[0] #Looking at Kentucky star and No. 1 pick Karl-Anthony Towns 

print(temp_player_df)
def GetReferenceIDForPlayer(playerdf):

    teamid = playerdf['TeamID']

    players_team = teams.loc[teams['TeamID'] == teamid]

    allTeams = Teams(year=2015)

    for team in allTeams:

        if(team.name.startswith(players_team.iloc[0]['TeamName'])):

            print('Team found')

            print(team.name)

            roster = team.roster  # Gets each team's roster

            for player in roster.players:

                if(playerdf['FirstName'] in player.name and playerdf['LastName'] in player.name):

                    print(player.player_id)

                    print(player.player_efficiency_rating)

                    print(player.points)

                    return
GetReferenceIDForPlayer(temp_player_df)