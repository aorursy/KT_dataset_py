# Import the necessary libraries needed for Data Exploration and Visualisation

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
# Read the dataset from csv file
inputdata=pd.read_csv('../input/originalDataset.csv')

# Total number of ODI Matches data present in the dataset
len(inputdata.index)

## Dataset comprises of 3932 odi matches details
# Count of Number of Unique Teams  in the Team 1 Column Dataset
len(inputdata['Team 1'].unique().tolist())
#Number of Unique Teams  in the Team 2 Column Dataset
team1List=inputdata['Team 1'].unique().tolist()
team1List
# Count of Number of Unique Teams  in the Team 2 Column Dataset
len(inputdata['Team 2'].unique().tolist())
# Number of Unique Teams in the Team 2 Column Dataset
team2List=inputdata['Team 2'].unique().tolist()
team2List
#Convert team1List and team2List as Set and perform Union Opertion to get the Total Number of Unique Teams and dispaly as list
TotalTeams=list(set(team1List).union(set(team2List)))
TotalTeams
#Remove Teams Formed for the Pupose of Charity and Exhibition Matches and get the final list of Teams
Teams_FinalList=[]
for i in TotalTeams:
    if 'XI' not in i:
        Teams_FinalList.append(i)
Teams_FinalList
#Consider the Records of ODI Matches Played Between Cricketing Nations 
Odi_matches=inputdata.loc[(inputdata['Team 1'].isin(Teams_FinalList)) & (inputdata['Team 2'].isin(Teams_FinalList)),:]
Odi_matches
Odi_matches.rename(columns={'Scorecard':'Odi_No'},inplace=True)
#Set Scorecard as index for the Records as it is unique
Odi_matches.set_index('Odi_No')
# Number of Matches won by Each Team in Ascending Order
Odi_matches[['Odi_No','Winner']].groupby('Winner').agg('count').sort_values(ascending=False,by='Odi_No')
Odi_matches_wins=pd.DataFrame(Odi_matches[['Odi_No','Winner']].groupby('Winner').agg('count').sort_values(ascending=False,by='Odi_No'))
# Adding Team as Column as we have to do label based indexing operations
Odi_matches_wins['Team']=Odi_matches_wins.index
# Renaming the Column to Number of Wins
Odi_matches_wins.rename(columns={'Odi_No':'Number of Wins'},inplace=True)
# Dropping the Rows Associated with Result and Not Tied
Odi_matches_wins=Odi_matches_wins.drop(['tied','no result'])
# Visualise the data of matches won by each team using bar plot
plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
sns.barplot(x='Number of Wins',y='Team',data=Odi_matches_wins)
plt.title('Matches won by Each Country')
plt.show()
#Total Number of Matches Played by each Counry as Team 2
Matches_Team1=pd.DataFrame(Odi_matches.groupby('Odi_No').sum().groupby('Team 1').count())
Matches_Team1.index.name='Team'
Matches_Team1=pd.DataFrame(Matches_Team1.loc[:,'Team 2'])
Matches_Team1
#Total Number of Matches Played by each Counry as Team 1
Matches_Team2=pd.DataFrame(Odi_matches.groupby('Odi_No').sum().groupby('Team 2').count())
Matches_Team2.index.name='Team'
Matches_Team2=pd.DataFrame(Matches_Team2.loc[:,'Team 1'])
Matches_Team2
#Total number of Matches Played by each Country

totalnumberofmatches_pernation=pd.merge(Matches_Team1,Matches_Team2,how='outer',on='Team')

## Replace NAN Values with 0
totalnumberofmatches_pernation.fillna(0,inplace=True)

## Convert Team 2 Column of type float 64 to int64
totalnumberofmatches_pernation['Team 2']=totalnumberofmatches_pernation['Team 2'].astype(np.int64,inplace=True)

## Add the values of Columns Team 2 and Team 1
totalnumberofmatches_pernation['TotalMatchesPlayed']=totalnumberofmatches_pernation['Team 2'].add(totalnumberofmatches_pernation['Team 1'])

## Add the Column Country 

totalnumberofmatches_pernation['Country']=totalnumberofmatches_pernation.index

## Sort the Value Based on the Descending Order
totalnumberofmatches_pernation.sort_values(ascending=False,by='TotalMatchesPlayed')
# Visualise the data using the Bar plot
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
sns.barplot(x='TotalMatchesPlayed',y='Country',data=totalnumberofmatches_pernation.sort_values(ascending=False,by='TotalMatchesPlayed'))
plt.title('Total Matches played by Each Country')
plt.show()
# Win Percentage of Each Country
totalnumberofmatches_pernation_columnsfiltered=totalnumberofmatches_pernation[['Country','TotalMatchesPlayed']]
Odi_matches_wins
Teams_withwinpercentage=pd.merge(Odi_matches_wins,totalnumberofmatches_pernation_columnsfiltered,how='inner',left_on='Team',right_on='Country')
Teams_withwinpercentage=Teams_withwinpercentage[['Country','Number of Wins','TotalMatchesPlayed']]
Teams_withwinpercentage['WinPercentage']=100*(Teams_withwinpercentage['Number of Wins']/Teams_withwinpercentage['TotalMatchesPlayed'])
Teams_withwinpercentage.sort_values(ascending=False,by='WinPercentage')
# Plotting the Win Percentage of each team on Bar PLot
plt.figure(num=None,figsize=(12,8),dpi=100,facecolor='w',edgecolor='w')
sns.barplot(x='WinPercentage',y='Country',data=Teams_withwinpercentage.sort_values(ascending=False,by='WinPercentage'))
plt.title('Teams with Win Percentage')
plt.show()
#Filter the data with Teams Playing More than 100 Matches
Teams_withwinpercentage_min100matchesPlayed=Teams_withwinpercentage[Teams_withwinpercentage.TotalMatchesPlayed>100]
Teams_withwinpercentage_min100matchesPlayed.sort_values(ascending=False,by='WinPercentage')
# Plotting the Win Percentage of each team on Bar PLot with mimimum 50 matches played
plt.figure(num=None,figsize=(12,8),dpi=100,facecolor='w',edgecolor='w')
sns.barplot(x='WinPercentage',y='Country',data=Teams_withwinpercentage_min100matchesPlayed.sort_values(ascending=False,by='WinPercentage'))
plt.title('Teams with Win Percentage (min 50 matches played)')
plt.show()
