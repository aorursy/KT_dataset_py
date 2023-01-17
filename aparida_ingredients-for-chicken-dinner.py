# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/test.csv')
data.head()
## Id 
#search for duplicates
any(data['Id'].duplicated())

## Id 
#total no of players
len(data['Id'])
## groupId
#Check NaN
data[data['groupId'].isnull()]
#No nan present
## groupId
#No. of people per group
groupIdData=pd.DataFrame(data['groupId'].value_counts())
groupIdData.reset_index(level=0, inplace=True)
groupIdData.columns = ['groupId', 'Members']
groupIdData.head()
## groupId
#Basic Stats on the members in each group
groupIdData['Members'].describe()
## groupId
# removing invalid groups where members more than 4 / could be just "useless" bots
groupIdDataValid=groupIdData[groupIdData['Members']<=4]
groupIdDataValid.head()
## groupId
#Basic Stats on the members in each VALID group
groupIdDataValid['Members'].describe()
## matchId 
# Total no. people in a match
matchIdData=pd.DataFrame(data['matchId'].value_counts())
matchIdData.reset_index(level=0, inplace=True)
matchIdData.columns = ['matchId', 'Players']
matchIdData.head()
## matchId 
# Total no. of matches
len(matchIdData)
## matchId
#Basic Stats on the players in each match
matchIdData['Players'].describe()
## matchId
# removing invalid matches where players are equal to 10 or less
# we need good comepition to identify most import fratures for a win 
matchIdDataValid=matchIdData[matchIdData['Players']>10]
matchIdDataValid.tail()
## matchId
#Basic Stats on the members in each VALID group
matchIdDataValid['Players'].describe()
## Main DataSet
# remove invalid groups from further analysis
groupIdDataValidList=list(groupIdDataValid['groupId'])
data=data[data['groupId'].isin(groupIdDataValidList)]
matchIdDataValidList=list(matchIdDataValid['matchId'])
data=data[data['matchId'].isin(matchIdDataValidList)]
len(data['Id'])
## assists
#Basic Stats on the player assists in each match
data['assists'].describe()
## boosts
#Basic Stats on the player boosts in each match
data['boosts'].describe()
## damageDealt 
#Basic Stats on the player damage dealt in each match
data['damageDealt'].describe()
## Killing Stats
# Basic Stats on player headshotKills, kills, roadKills and friendlyKills 
killing=data[['kills','headshotKills','roadKills','teamKills']]
killing.describe(include='all')
## heals 
#Basic Stats on the player healing items used in each match
data['heals'].describe()
## revives
# Basic Stats on the player reviving another player  in a match
data['revives'].describe()
## weaponsAcquired
# Basic Stats on the no. of weapon picked up a player
data['weaponsAcquired'].describe()
## numGroups
# Basic Stats on the no. of groups joining a game 
data['numGroups'].describe()
## killPlace

#Basic Stats on the player rank based on her/his kills in the match
# Just checking for a  min max limits else it is not useful
data['killPlace'].describe()
## Travel 
# Basic descriptive analysis of player travel distance on foot, vehicle and swim
# All values are in 'm' 
data['totalDistance']=data.walkDistance+data.rideDistance+data.swimDistance
travel=data[['walkDistance','rideDistance','swimDistance','totalDistance']]
travel.describe(include='all')
## Elo Rating
# basic description of Kill and win Elo rating of each players
Elo=data[['winPoints','killPoints']]
Elo.describe(include='all')

### Does this makes sense as Elo rating evolves with time and same player can increase/decrease so mean and all may not be meaningful 

# Some rating for group participation
groupIdDataList=list(set(data['groupId']))
for group in groupIdDataList:
    #if (i+1)%100 ==0:
      #  print(i+1,'/',len(groupIdDataList))
        
    data.loc[data['groupId']==group,'totalTeamsKills']=data[data['groupId']==group]['kills'].mean()
    data.loc[data['groupId']==group,'totalTeamWinPoints']=data[data['groupId']==group]['winPoints'].mean()
    data.loc[data['groupId']==group,'totalTeamKillPoints']=data[data['groupId']==group]['killPoints'].mean()

# Some elo based expectation caluation
matchIdDataList=list(set(data['matchId']))

for match in matchIdDataList:
    matchData=data[data['matchId']== match]
    
    groupsMatchList=list(set(matchData['groupId']))
    
    for group in groupsMatchList:
        data.loc[data['groupId']==group,'ExpectedWinPoints']=1/(1+10**(-abs(matchData[matchData['groupId']==group]['totalTeamWinPoints'].mean()-matchData['totalTeamWinPoints'].mean())/400))
        data.loc[data['groupId']==group,'ExpectedKillPoints']=1/(1+10**(-abs(matchData[matchData['groupId']==group]['totalTeamKillPoints'].mean()-matchData['totalTeamKillPoints'].mean())/400))
        
dropCols = ['Id', 'groupId', 'matchId']
# These have no outcome on the game;
#'maxPlace'=='numGroups'
#data=data.drop(['maxPlace'], axis=1)
keepCols = [col for col in data.columns if col not in dropCols]
corr = data[keepCols].corr()
plt.figure(figsize=(15,10))
plt.title("Correlation Heat Map of Data")
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    annot=True,
    cmap="RdYlGn",
)
plt.show()
data.to_csv('../working/cleanedTrain.csv')
print(os.listdir("../working"))
