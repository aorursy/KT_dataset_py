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



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

#print multiple outputs


MEvents2015 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2015.csv")

MEvents2016 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2016.csv")

MEvents2017 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2017.csv")

MEvents2018 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2018.csv")

MEvents2019 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2019.csv")

MPlayers = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv")

MSampleSubmissionStage1_2020 = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv")



Seasons = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSeasons.csv")

Conferences = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/Conferences.csv")

TeamConferences = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamConferences.csv")

Coaches = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamCoaches.csv")

Teams = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv")

Cities = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/Cities.csv")

Massey = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MMasseyOrdinals.csv")

TeamSpellings = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamSpellings.csv", encoding = "ISO-8859-1")

GameCities = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MGameCities.csv")

Seeds = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv")

Slots = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySlots.csv")

Round_Slots = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv")



NCAACompact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")

NCAADetailed = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv")

SeasonCompact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")

SeasonDetailed = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")

NITTeams = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSecondaryTourneyTeams.csv")

NITCompact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSecondaryTourneyCompactResults.csv")

ConfTourneyGames = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MConferenceTourneyGames.csv")
#process data

#Add Full Name to Conferences. 

#Create seednum column 

#Combine regular season + tourney detailed data after adding tourney (T/F) column to both frames

TeamConferences = (pd.merge(TeamConferences, Conferences, on='ConfAbbrev')

                  .rename({'Description': 'conf_descr'}, axis=1))

Seeds['seednum'] = Seeds ['Seed'].str.slice(1,3).astype(int)

SeasonDetailed['tourney'] = 0

NCAADetailed['tourney'] = 1

gamedata = pd.concat([SeasonDetailed, NCAADetailed])

gamedata
    # team1: team with lower id

gamedata['team1'] = (gamedata['WTeamID'].where(gamedata['WTeamID'] < gamedata['LTeamID'],

                                           gamedata['LTeamID']))

    # team2: team with higher id

gamedata['team2'] = (gamedata['WTeamID'].where(gamedata['WTeamID'] > gamedata['LTeamID'],

                                           gamedata['LTeamID']))

gamedata['score1'] = gamedata['WScore'].where(gamedata['WTeamID'] < gamedata['LTeamID'], gamedata['LScore'])

gamedata['score2'] = gamedata['WScore'].where(gamedata['WTeamID'] > gamedata['LTeamID'], gamedata['LScore'])



boxscore_stats = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',

                      'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF',]
gamedata