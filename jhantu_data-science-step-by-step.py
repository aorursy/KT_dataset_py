# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
world_cup_raw_data= pd.read_csv("../input/WorldCups.csv")
world_cup_raw_data.head()
world_cup_raw_data.shape

world_cup_raw_data.isnull().sum()
world_cup_raw_data.describe()
world_cup_raw_data.hist()
world_cup_raw_data['Winner'].describe()
#winner and times
world_cup_raw_data.Winner.value_counts().plot(kind='bar')
world_cup_raw_data['Winner'].hist()
##point to the winner in year
import matplotlib.pyplot as plt
t=world_cup_raw_data.iloc[:,[0,2]]
plt.figure(figsize=(5,5))
plt.scatter(t.Year,t.Winner,color='red')
plt.show()

#what years that Brazil is Winner
brazil_winner=world_cup_raw_data[world_cup_raw_data['Winner']=='Brazil']
brazil_winner.Year.value_counts().plot(kind='bar')
#Country is Winner?
country_is_winner= world_cup_raw_data[(world_cup_raw_data['Winner']==world_cup_raw_data['Country'])]
country_is_winner
country_is_winner.Winner.value_counts().plot(kind='bar')
world_cup_raw_data['Runners-Up'].describe()
#country is runners-up
country_is_runnersup=world_cup_raw_data[(world_cup_raw_data['Runners-Up']==world_cup_raw_data['Country'])]
country_is_runnersup
country_is_runnersup.Country.value_counts().plot(kind='bar')
world_cup_raw_data['Third'].describe()
#country is third
country_is_third = world_cup_raw_data[(world_cup_raw_data['Third']==world_cup_raw_data['Country'])]
country_is_third
country_is_third.Country.value_counts().plot(kind='bar')
world_cup_raw_data['Fourth'].describe()
#country is fourth
country_is_fourth = world_cup_raw_data[(world_cup_raw_data['Fourth']==world_cup_raw_data['Country'])]
country_is_fourth
country_is_fourth.Country.value_counts().plot(kind='bar')
world_cup_matches_raw_data= pd.read_csv("../input/WorldCupMatches.csv")
world_cup_matches_raw_data.head()
world_cup_matches_raw_data.shape
world_cup_matches_raw_data.isnull().sum()
world_cup_matches_raw_data.describe()
#drop 3722 missing data rows will condition "all missing columns"
#world_cup_matches_filter_data: 852 rows, Attendance column is missing 2 rows
world_cup_matches_filter_data= world_cup_matches_raw_data.dropna(how='all')
world_cup_matches_filter_data.describe()
world_cup_matches_filter_data.shape
world_cup_matches_filter_data.isnull().sum()
world_cup_matches_filter_data[world_cup_matches_filter_data['Attendance'].isnull()]
#first year winner is brasil 1958
brazil_winner_1958= world_cup_matches_filter_data[world_cup_matches_filter_data.Year==1958]
brazil_winner_1958[(brazil_winner_1958['Home Team Name']=='Brazil') | (brazil_winner_1958['Away Team Name']=='Brazil')]

world_cup_players_raw_data= pd.read_csv("../input/WorldCupPlayers.csv")
world_cup_players_raw_data.head()
world_cup_players_raw_data.shape
world_cup_players_raw_data.isnull().sum()
world_cup_matches_raw_data.describe()
#MatchID 1343.0( final Brazill-Sweeden 1958)
#brazil players
world_cup_players_raw_data[(world_cup_players_raw_data['MatchID']==1343) & (world_cup_players_raw_data['Team Initials'] == 'BRA')]
#MatchID 1343.0( final Brazill-Sweeden 1958)
#sweeden players
world_cup_players_raw_data[(world_cup_players_raw_data['MatchID']==1343) & (world_cup_players_raw_data['Team Initials'] == 'SWE')]