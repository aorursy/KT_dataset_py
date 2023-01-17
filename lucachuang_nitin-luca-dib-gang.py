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
MTeams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv')

print('Shape of MTeams = ', MTeams.shape)

MTeams.head()
#We might have to remove the teams where LastD1Season is not 2020.

MTeams.LastD1Season.value_counts()
MSeasons = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSeasons.csv')

print('Shape of MSeasons = ', MSeasons.shape)

MSeasons.head()
MNCAATourneySeeds = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')

print('Shape of MNCAATourneySeeds = ', MNCAATourneySeeds.shape)

MNCAATourneySeeds.head()
#W11ab, W16ab and X11ab, X16ab to play together in 2019 and give 4 out of 8 teams to make a total of 64 teams in 2019. 

#Similarly for other seasons, if any.

MNCAATourneySeeds[MNCAATourneySeeds.Season == 2019].shape
#From Day 1 to Day 132. Regular games for selection into main games

MRegularSeasonCompactResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

print('Shape of MRegularSeasonCompactResults = ', MRegularSeasonCompactResults.shape)

MRegularSeasonCompactResults.head()

MRegularSeasonCompactResults.WLoc.value_counts()
MNCAATourneyCompactResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

print('Shape of MNCAATourneyCompactResults = ', MNCAATourneyCompactResults.shape)

MNCAATourneyCompactResults[MNCAATourneyCompactResults.Season == 2019].head()
print(MNCAATourneyCompactResults[(MNCAATourneyCompactResults.Season == 2019) & ((MNCAATourneyCompactResults.DayNum == 134) | (MNCAATourneyCompactResults.DayNum == 135))].shape)

MNCAATourneyCompactResults[(MNCAATourneyCompactResults.Season == 2019) & ((MNCAATourneyCompactResults.DayNum == 134) | (MNCAATourneyCompactResults.DayNum == 135))].head()
print(MNCAATourneyCompactResults[(MNCAATourneyCompactResults.Season == 2019) & ((MNCAATourneyCompactResults.DayNum == 136) | (MNCAATourneyCompactResults.DayNum == 137))].shape)

MNCAATourneyCompactResults[(MNCAATourneyCompactResults.Season == 2019) & ((MNCAATourneyCompactResults.DayNum == 136) | (MNCAATourneyCompactResults.DayNum == 137))].head()
MRegularSeasonDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')

print('Shape of MRegularSeasonDetailedResults = ', MRegularSeasonDetailedResults.shape)

MRegularSeasonDetailedResults.head()
MNCAATourneyDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

print('Shape of MNCAATourneyDetailedResults = ', MNCAATourneyDetailedResults.shape)

MNCAATourneyDetailedResults.head()
MGameCities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MGameCities.csv')

print('Shape of MGameCities = ', MGameCities.shape)

MGameCities.head()
MMasseyOrdinals = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MMasseyOrdinals.csv')

print('Shape of MMasseyOrdinals = ', MMasseyOrdinals.shape)

MMasseyOrdinals.head()
MEvents2015 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2015.csv')

MEvents2016 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2016.csv')

MEvents2017 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2017.csv')

MEvents2018 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2018.csv')

MEvents2019 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MEvents2019.csv')

print('Shape of MEvents2015 = ', MEvents2015.shape)

MEvents2015.head()
MPlayers = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MPlayers.csv')

print('Shape of MPlayers = ', MPlayers.shape)

MPlayers.head()
MTeamCoaches = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamCoaches.csv')

print('Shape of MTeamCoaches = ', MTeamCoaches.shape)

MTeamCoaches.head()
Conferences = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/Conferences.csv')

print('Shape of Conferences = ', Conferences.shape)

Conferences.head()
MTeamConferences = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamConferences.csv')

print('Shape of MTeamConferences = ', MTeamConferences.shape)

MTeamConferences.head()
MConferenceTourneyGames = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MConferenceTourneyGames.csv')

print('Shape of MConferenceTourneyGames = ', MConferenceTourneyGames.shape)

MConferenceTourneyGames.head()
MSecondaryTourneyTeams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSecondaryTourneyTeams.csv')

print('Shape of MSecondaryTourneyTeams = ', MSecondaryTourneyTeams.shape)

MSecondaryTourneyTeams.head()

MSecondaryTourneyCompactResults = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSecondaryTourneyCompactResults.csv")

print('Shape of MSecondaryTourneyCompactResults = ', MSecondaryTourneyCompactResults.shape)

MSecondaryTourneyCompactResults.head()
MTeamSpellings = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamSpellings.csv", engine='python')

print("Shape of MTeamSpellings = ", MTeamSpellings.shape)

MTeamSpellings.head()
MNCAATourneySlots = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySlots.csv')

print('Shape of MNCAATourneySlots = ', MNCAATourneySlots.shape)

MNCAATourneySlots.head()

MNCAATourneySeedRoundSlots = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv')

print('Shape of MNCAATourneySeedRoundSlots = ', MNCAATourneySeedRoundSlots.shape)

MNCAATourneySeedRoundSlots.head()