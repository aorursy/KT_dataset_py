# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#lets load dataframe and mathematical libraries as well as plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns',None)
#loading General.csv which contains gneralised overview of a match 
general_dataset = pd.read_csv("../input/datasetcsv/General.csv")
general_dataset.head()
general_dataset.shape
general_dataset.columns
#loading Rounds.csv which contains data for every round in every game
round_dataset = pd.read_csv("../input/datasetcsv/Rounds.csv")
round_dataset.head()
round_dataset.shape
round_dataset.columns
#loading Players.csv which contains data of performance of each players in every game 
players_dataset = pd.read_csv("../input/datasetcsv/Players.csv")
players_dataset.head()
players_dataset.shape
players_dataset.columns
#loading Kills.csv which contains data for every kill in every round in every game
kills_dataset = pd.read_csv("../input/datasetcsv/Kills.csv")
kills_dataset.head()
kills_dataset.shape
kills_dataset.columns
#loading EKTeams.csv which contains data for entry kills for every Team for terrorist side 
EKTeams_dataset = pd.read_csv("../input/datasetcsv/EKTeams.csv")
EKTeams_dataset.head()
EKTeams_dataset.shape
EKTeams_dataset.columns
#loading EKRounds.csv which containns data for entry kills for every round in terrorist side
EKRounds_dataset = pd.read_csv("../input/datasetcsv/EKRounds.csv")
EKRounds_dataset.head()
EKRounds_dataset.shape
EKRounds_dataset.columns
#loading EKPlayers.csv which contains data for entry kills for every player in terrorist side
EKPlayers_dataset= pd.read_csv("../input/datasetcsv/EKPlayers.csv")
EKPlayers_dataset.head()
EKPlayers_dataset.shape
EKPlayers_dataset.columns
#loading EHKTeams.csv which contains data for hold kills for every Team for counter terrorist side
EHKTeams_dataset = pd.read_csv("../input/datasetcsv/EHKTeams.csv")
EHKTeams_dataset.head()
EHKTeams_dataset.shape
EHKTeams_dataset.columns
#loading EHKRounds.csv which containns data for hold kills for every round in counter terrorist side
EHKRounds_dataset = pd.read_csv("../input/datasetcsv/EHKRounds.csv")
EHKRounds_dataset.head()
EHKRounds_dataset.shape
EHKRounds_dataset.columns
#loading EHKPlayers.csv which contains data for hold kill for every player in counter terrorist side
EHKPlayers_dataset = pd.read_csv("../input/datasetcsv/EHKPlayer.csv")
EHKPlayers_dataset.head()
EHKPlayers_dataset.shape
EHKPlayers_dataset.columns
#loading KillMatrix.csv which contains nxn kill matrix for every player against another players for every game in competition
killmatrix_data = pd.read_csv("../input/datasetcsv/KillMatrix.csv")
killmatrix_data.head(15)
killmatrix_data.shape
killmatrix_data.columns
#loading FMPlayers.csv which contains nxn flash matrix for every players againt his opponent for every match
FMP_data = pd.read_csv("../input/datasetcsv/FMPlayers.csv")
FMP_data.head()
FMP_data.shape
FMP_data.columns
#loading FMTeams.csv which contains nxn flash matrix for one team against its opponents for every game
FMT_data = pd.read_csv("../input/datasetcsv/FMTeams.csv")
FMT_data.head()
FMT_data.shape
FMT_data.columns
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
general_data = general_dataset.drop(['Filename', 'Type', 'Source', 'Hostname', 'Client', 'Server Tickrate', 'Framerate', 'Flashbang', 'Smoke', 'HE', 'Decoy', 'Molotov', 'Incendiary', 'Shots', 'Comment', 'Cheater'], axis=1)
general_data.head()
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
players_data = players_dataset.drop(['SteamID', 'Rank', 'VAC', 'OW'], axis=1)
players_data.head()
#players_data.shape
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
kills_data = kills_dataset.drop(['Killer SteamID', 'Killer bot', 'Victim SteamId', 'Victim bot', 'Assister SteamID', 'assister bot'], axis=1)
#kills_data.head()
kills_data.shape
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
EKRounds_data = EKRounds_dataset.drop(['Killer SteamID', 'Victim SteamID'], axis=1)
EKRounds_data.columns
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
EKPlayers_data = EKPlayers_dataset.drop(['SteamID'], axis=1)
EKPlayers_data.columns
EKPlayers_data.shape
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
EHKRounds_data = EHKRounds_dataset.drop(['Killer SteamID', 'Victim SteamID'], axis=1)
EHKRounds_data.shape
#here we drop columns which does not hold meaningful data extra ids and source names and null columns or columns with same data
EHKPlayers_data = EHKPlayers_dataset.drop(['SteamID'], axis=1)
EHKPlayers_data.shape
#here we drop null rows in columns
EKRounds_data = EKRounds_data.dropna()
#EKRounds_data.head(10)
EKRounds_data.shape
#here we drop null rows in columns
EHKRounds_data = EHKRounds_data.dropna()
EHKRounds_data.head(10)
#here we are normalizing several columns so that data will be reliable and well structured
#winner_team = []
#looser_team= []
winner_team = np.where(general_data['Score team 1'] > general_data['Score team 2'], general_data['Score team 1'], general_data['Score team 2'])
#print(winner_team)    
looser_team = np.where(general_data['Score team 1'] < general_data['Score team 2'], general_data['Score team 1'], general_data['Score team 2'])
#print(looser_team) 
winner_team_name = np.where(general_data['Score team 1'] > general_data['Score team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(winner_team_name )
looser_team_name = np.where(general_data['Score team 1'] < general_data['Score team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(looser_team_name)
first_half_winner = np.where(general_data['Score 1st half team 1'] > general_data['Score 1st half team 2'], general_data['Score 1st half team 1'], general_data['Score 1st half team 2'])
#print(first_half_winner)
first_half_looser = np.where(general_data['Score 1st half team 1'] < general_data['Score 1st half team 2'], general_data['Score 1st half team 1'], general_data['Score 1st half team 2'])
#print(first_half_looser)
first_half_winner_team_name = np.where(general_data['Score 1st half team 1'] > general_data['Score 1st half team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(first_half_winner_team_name)
first_half_looser_team_name = np.where(general_data['Score 1st half team 1'] < general_data['Score 1st half team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(first_half_looser_team_name)
second_half_winner = np.where(general_data['Score 2nd half team 1'] > general_data['Score 2nd half team 2'], general_data['Score 2nd half team 1'], general_data['Score 2nd half team 2'])
#print(second_half_winner)
second_half_looser = np.where(general_data['Score 2nd half team 1'] < general_data['Score 2nd half team 2'], general_data['Score 2nd half team 1'], general_data['Score 2nd half team 2'])
#print(second_half_looser)
second_half_winner_team_name = np.where(general_data['Score 2nd half team 1'] > general_data['Score 2nd half team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(second_half_winner_team_name)
second_half_looser_team_name = np.where(general_data['Score 2nd half team 1'] < general_data['Score 2nd half team 2'], general_data['Name team 1'], general_data['Name team 2'])
#print(second_half_looser_team_name)

general_data['Winner_team'] = winner_team
general_data['Looser_team'] = looser_team
general_data['Winner_team_name'] = winner_team_name
general_data['Looser_team_name'] = looser_team_name
general_data['First_half_winner'] = first_half_winner
general_data['First_half_looser'] = first_half_looser
general_data['First_half_winner_team_name'] = first_half_winner_team_name
general_data['First_half_looser_team_name'] = first_half_looser_team_name
general_data['Second_half_winner'] = second_half_winner
general_data['Second_half_looser'] = second_half_looser
general_data['Second_half_winner_team_name'] = second_half_winner_team_name
general_data['Second_half_looser_team_name'] = second_half_looser_team_name
general_data.head()
general_data.drop_duplicates(subset=['ID'], keep='first', inplace=True)
general_data.shape
#counting unique maps and how many time they were played
test = general_data['Map'].value_counts()
print(test)
sns.countplot(x="Map", data=general_data);
plt.xticks(rotation='vertical');
#collecting name of all the team who participated in competition challenger stage
general_data['Name team 1'].unique()
general_data['Map'].unique()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_dust2'])
grouped['de_dust2'].plot.bar();
#grouped.dtypes
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_dust2'])
grouped['de_dust2'].plot.bar();
#plotting winninng teams on de_inferno
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_inferno'])
grouped['de_inferno'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_inferno'])
grouped['de_inferno'].plot.bar()
#plotting winninng teams on de_mirage
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_mirage'])
grouped['de_mirage'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_mirage'])
grouped['de_mirage'].plot.bar()
#plotting winninng teams on de_train
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_train'])
grouped['de_train'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_train'])
grouped['de_train'].plot.bar()
#plotting winninng teams on de_overpass
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_overpass'])
grouped['de_overpass'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_overpass'])
grouped['de_overpass'].plot.bar()
#plotting winninng teams on de_nuke
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_nuke'])
grouped['de_nuke'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_nuke'])
grouped['de_nuke'].plot.bar()
#plotting winninng teams on de_vertigo
grouped = general_data.groupby("Map")['Winner'].value_counts()
print(grouped['de_vertigo'])
grouped['de_vertigo'].plot.bar()
#finding and plotting which map has how many teams winning that map also how many times
grouped = general_data.groupby("Map")['Looser_team_name'].value_counts()
print(grouped['de_vertigo'])
grouped['de_vertigo'].plot.bar()
gg = general_data.groupby('Winner')['Map'].value_counts()
print(gg['Astralis'])
general_data['Name team 1'].unique()
general_data['Name team 2'].unique()
test=[]
teams = ['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']
for i in range(len(teams)):
    #print(teams[i])
    temp1=np.where(general_data['Name team 1'].isin([teams[i]]), 1, 0 )
    temp2=np.where(general_data['Name team 2'].isin([teams[i]]), 1, 0 )
    gg1 = temp1.sum()
    gg2 = temp2.sum()
    gg = gg1+gg2
    #print(teams[i],gg)
    test.append(gg)
    print(teams[i] , gg)
   
#print(sorted(test, reverse=True))
#print(test)
sns.countplot(x='Winner', data=general_data);
plt.xticks(rotation='vertical');
sns.countplot(x='Looser_team_name', data=general_data);
plt.xticks(rotation='vertical');
#lets move to duration column and check its 5 point summary 
general_data['Duration'].describe()
sns.distplot(general_data['Duration']);
#let's groupby duaration wrt to map and see which map has what duration of games in general
grouped = general_data.groupby('Map')['Duration'].describe()
print(grouped)
sns.distplot(grouped);
# does match duration varies with team?? 
grouped = general_data.groupby('Winner')['Duration'].describe()
print(grouped)
# does match duration varies with team?? 
grouped = general_data.groupby('Looser_team_name')['Duration'].describe()
print(grouped)
test = general_data.loc[(general_data['Name team 1'] == 'Avangar') | (general_data['Name team 2'] == 'Avangar')]['Map'].value_counts()
print(test)
teams = ['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']
for i in range(len(teams)):
    #print(teams[i])
    test = general_data.loc[(general_data['Name team 1'] == teams[i]) | (general_data['Name team 2'] == teams[i])]['Map'].value_counts()
    print(teams[i])
    print(test)
    
#print(test)
test = []
teams = ['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']
for i in range(len(teams)):
    #print(teams[i])
    temp = general_data.loc[(general_data['Name team 1'] == teams[i]) | (general_data['Name team 2'] == teams[i])]['Map'].value_counts()
    test.append(temp)
    #print(teams[i])
    #print(test)
    
print(test[0])
gg = []
#finding and plotting which map has how many teams winning that map also how many times
gg = general_data.groupby("Map")['Winner'].value_counts()
print(gg[0])
win_ratio =[] 
#print(test[0])
win_ratio = gg[0]/test[0]
print(win_ratio[0])
count = general_data.groupby(['Map', 'Name team 1', 'Name team 2','Winner']).size()
print(count)
count=[]
teams = ['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']
count = general_data.groupby(['Name team 1', 'Name team 2'])['Duration'].mean()
#print(count)
#print(count['Avangar'])

for i in range(len(teams)):
    print(teams[i])
    print(count[teams[i]])
    count[teams[i]].plot.bar();
    

general_data['Score_diff'] = general_data['Winner_team'] - general_data['Looser_team']
print(len(general_data))
print(general_data['ID'].value_counts())
#t = input("input team name:")
t='Astralis'
testing = general_data[general_data['Winner'] == t]
testing.groupby(['Map','Winner', 'Looser_team_name','Score_diff'])['Map','Winner','Looser_team_name','Score_diff'].size()
#t = input("input team name:")
t='Astralis'
testing = general_data[general_data['Looser_team_name'] == t]
testing.groupby(['Map','Looser_team_name', 'Winner', 'Score_diff']).size()
general_data['Output_enable'] = np.where(general_data['Winner_team'] > 16, 1, 0)
general_data.head(20)
graph = general_data.groupby('Map')['Output_enable'].value_counts()
print(graph)
graph.plot.bar();
graph=general_data.groupby("Winner")['Output_enable'].value_counts()
print(graph)
graph.plot.bar();
first_half_score_diff = general_data['First_half_winner'] - general_data['First_half_looser']
print(first_half_score_diff)
graph = general_data.groupby(['Winner_team_name'])['Output_enable'].value_counts()
print(graph)
graph.plot.bar();
graph = general_data.groupby(['Looser_team_name'])['Output_enable'].value_counts()
print(graph)
graph.plot.bar();
general_data['First_half_score_diff'] = first_half_score_diff
general_data.head(5)
#t= input('put team name:')
t='Astralis'
testing = general_data[general_data['First_half_winner_team_name'] == t]
testing.groupby(['Map','First_half_winner_team_name', 'First_half_looser_team_name', 'First_half_score_diff']).size()
#t= input('put team name:')
t='Astralis'
testing = general_data[general_data['First_half_looser_team_name'] == t]
testing.groupby(['Map','First_half_looser_team_name', 'First_half_winner_team_name', 'First_half_score_diff']).size()
plt.figure(figsize=(10,10))
sns.boxplot(y='First_half_winner_team_name', x='First_half_score_diff', data=general_data, orient='h')
plt.show()
plt.figure(figsize=(10,10))
sns.boxplot(y='First_half_looser_team_name', x='First_half_score_diff', data=general_data, orient='h')
plt.show()
Second_half_score_diff = general_data['Second_half_winner'] - general_data['Second_half_looser']
print(Second_half_score_diff)
general_data['Second_half_score_diff'] = Second_half_score_diff
general_data.head()
#t= input('put team name:')
t='Astralis'
testing = general_data[general_data['Second_half_winner_team_name'] == t]
testing.groupby(['Map','Second_half_winner_team_name', 'Second_half_looser_team_name', 'Second_half_score_diff']).size()
#t= input('put team name:')
t='Astralis'
testing = general_data.loc[(general_data['Second_half_looser_team_name'] == t)]
testing.groupby(['Map','Second_half_looser_team_name', 'Second_half_winner_team_name', 'Second_half_score_diff']).size()
plt.figure(figsize=(10,10))
sns.boxplot(y='Second_half_winner_team_name', x='Second_half_score_diff', data=general_data, orient='h')
plt.show()
plt.figure(figsize=(10,10))
sns.boxplot(y='Second_half_looser_team_name', x='Second_half_score_diff', data=general_data, orient='h')
plt.show()
#t= input('put team name:')
t='Astralis'
testing = general_data[general_data['Winner'] == t]
print(testing[['Winner','First_half_winner_team_name','Second_half_winner_team_name']])
twist = general_data.loc[general_data['Winner'] != general_data['First_half_winner_team_name']]
print(len(twist))
gg = twist.groupby(['Map','Looser_team_name','Winner','First_half_score_diff','Second_half_score_diff']).size()
print(gg)
kills_per_round = general_data['Kills'] / general_data['Round']
print(kills_per_round)
general_data['Kills_per_round'] = kills_per_round
general_data.head(20)
general_data['Kills_per_round'].describe()
general_data.groupby('Map')['Kills_per_round'].describe()
plot = general_data.groupby('Map')['Kills_per_round'].mean()
print(plot)
plot.plot.bar();
general_data.groupby('Winner')['Kills_per_round'].describe()
plot = general_data.groupby('Winner')['Kills_per_round'].mean()
print(plot)
plot.plot.bar();
general_data.groupby('Looser_team_name')['Kills_per_round'].describe()
plot = general_data.groupby('Looser_team_name')['Kills_per_round'].mean()
print(plot)
plot.plot.bar();
Normalized_assists = general_data['Assists'] / general_data['Round']
print(Normalized_assists)
general_data['Normalized_assists'] = Normalized_assists
general_data.head()
general_data['Normalized_assists'].describe()
general_data.groupby('Map')['Normalized_assists'].describe()
plot = general_data.groupby('Map')['Normalized_assists'].mean()
print(plot)
plot.plot.bar();
general_data.groupby('Winner')['Normalized_assists'].describe()
plot = general_data.groupby('Winner')['Normalized_assists'].mean()
print(plot)
plot.plot.bar();
general_data.groupby('Looser_team_name')['Normalized_assists'].describe()
plot = general_data.groupby('Looser_team_name')['Normalized_assists'].mean()
print(plot)
plot.plot.bar();
general_data['Clutch'].describe()
sns.distplot(general_data['Clutch'])
general_data.groupby(['Map','Winner', 'Looser_team_name'])['Clutch'].value_counts()
general_data['Bomb Planted'].describe()
gg = general_data.groupby('Map')['Bomb Planted'].describe()
print(gg)
sns.distplot(gg);
general_data['Round'].describe()
general_data.groupby('Map')['Round'].describe()
general_data.groupby('Winner')['Round'].describe()
general_data.groupby('Looser_team_name')['Round'].describe()
general_data.head()
round_dataset.head()
round_dataset.shape
round_dataset.columns
test = general_data[['Source.Name','Winner','Looser_team_name','Map','Name team 1', 'Name team 2','First_half_winner_team_name','Second_half_winner_team_name']]
test.head()
round_data = pd.merge(round_dataset, test, on='Source.Name')
round_data.head(40)
round_data['Side'] = round_data['Side'].fillna('NA')
round_data['Team'] = round_data['Team'].fillna('NA')
round_data.head(25)
looser_team_name = np.where(round_data['Winner Clan Name'] == round_data['Winner_y'], round_data['Looser_team_name'] , round_data['Winner_y'])

round_data['Looser Clan Name'] = looser_team_name
round_data.head(50)
Looser = np.where(round_data['Winner_x'] == 'T','CT','T')
#for i in range (len(Looser)):
    #print(Looser[i])
    
round_data['Looser'] = Looser
round_data.head(50)
g = np.where((round_data['Number']>15) & (round_data['Number']<31), round_data['Name team 2'], round_data['Name team 1'])
h = np.where((round_data['Number']>15) & (round_data['Number']<31), round_data['Name team 1'], round_data['Name team 2'])       

round_data['Name team 1'] = g
round_data['Name team 2'] = h

round_data.head(60)
round_data.columns
round_dataset['Duration (s)'].describe()
f = round_dataset.groupby('Type')['Duration (s)'].describe()
f
gg = round_dataset[(round_dataset['Duration (s)'] > 0) & (round_dataset['Duration (s)'] < 250)]
print(len(gg))
print(gg['Duration (s)'])
(len(gg)/len(round_data))*100
plt.figure(figsize=(10,10))
sns.barplot(data=round_data, x='Type', y='Duration (s)', hue='Type');
#t= input('put team name:')
t='Astralis'
testing = round_data[round_data['Winner Clan Name']==t]
testing.groupby(['Winner Clan Name','Map','Winner_x','Type'])['Duration (s)'].describe()
#t= input('put team name:')
t='Astralis'
testing = round_data[round_data['Looser Clan Name']==t]
testing.groupby(['Looser Clan Name','Map','Looser','Type'])['Duration (s)'].describe()
round_data.groupby(['Map','Winner_x']).size()
r=[]
#t= input('put team name:')
t='Astralis'
testing = round_data[round_data['Winner Clan Name']==t]
r.append(testing.groupby(['Winner Clan Name','Map'])['Winner_x'].value_counts())
r
t='Astralis'
overall = round_data.loc[(round_data['Name team 2']==t)]
overall.groupby(['Map'])['Name team 2'].count()
t= 'Astralis'
maps =['de_inferno', 'de_dust2', 'de_vertigo', 'de_nuke', 'de_overpass',
       'de_train', 'de_mirage']

side1 = 'CT'
side2= 'T'
for m in maps:
    ovrdct = round_data.loc[(round_data['Name team 1']==t) & (round_data['Map'] == m)]['Name team 1'].count()
    ovrdt = round_data.loc[(round_data['Name team 2']==t) & (round_data['Map'] == m)]['Name team 2'].count()
    rdct  = round_data.loc[(round_data['Winner Clan Name'] == t) & (round_data['Winner_x'] == side1 ) & (round_data['Map'] == m)]['Winner_x'].count()
    rdt  = round_data.loc[(round_data['Winner Clan Name'] == t) & (round_data['Winner_x'] == side2 ) & (round_data['Map'] == m)]['Winner_x'].count()
    def result(p1,p2):
        try:
            #suppose that number2 is a float
            return (p1/p2)*100
        except ZeroDivisionError:
            return 0
    print('{} winning % of {} rounds on {} is'.format(t,side1,m))
    percent1 = result(rdct,ovrdct)
    print(percent1)
    print('{} winning % of {} rounds on {} is'.format(t,side2,m))
    percent2 = result(rdt,ovrdt)
    print(percent2)    
#t= input('put team name:')
t='Astralis'

testing = round_data[round_data['Looser Clan Name']==t]
testing.groupby(['Looser Clan Name','Map'])['Looser'].value_counts()
#t= input('put team name:')
t='Astralis'
testing = round_data[round_data['Winner Clan Name']==t]
testing.groupby(['Winner Clan Name','Map','Winner_x'])['Duration (s)'].describe()
#t= input('put team name:')
t='Astralis'
testing = round_data[round_data['Looser Clan Name']==t]
testing.groupby(['Looser Clan Name','Map','Looser'])['Duration (s)'].describe()
plt.figure(figsize=(10,8))
sns.lineplot(data=round_data, x='Number', y='Start money team 1', hue='Type');
plt.figure(figsize=(10,8))
sns.lineplot(data=round_data, x='Number', y='Start money team 2', hue='Type');
a=[]
temp = []
t='Astralis'
count=0
for i in round_data['Winner Clan Name']:
    if t==i:
        count+=1
    elif t!=i:
        temp.append(count)
        count=0
        
for i in range(len(temp)):
    if (temp[i]>2):
        a.append(temp[i])
        
print(a)     
print(len(a))
a=[]
temp = []
t='Astralis'
count=0
for i in round_data['Looser Clan Name']:
    if i==t:
        count+=1
        #print(count)
    elif t!=i:
        temp.append(count)
        count=0
        
    
for i in range(len(temp)):
    if (temp[i]>2):
        a.append(temp[i])
        
        
print(a)     
print(len(a))
t='Astralis'
m = 'de_inferno'
s='CT'
count=0
temp=[]
a=[]
test= round_data[(round_data['Map'] == m) & (round_data['Winner_x'] == s)]
for i in (test['Winner Clan Name']):
    #print(i)
    if i== t:
        count=count+1
    else:
        temp.append(count)
        count=0
        
for i in range(len(temp)):
    if (temp[i]>2):
        a.append(temp[i])
        
print(a)     
print(len(a))
a=[]
temp = []
t='Astralis'
o= 'Avangar'
count=0
test= round_data[round_data['Winner_y'] == t]
for i in test['Looser Clan Name']:
    #print(i)
    if o == i:
        count+=1
    elif o != i:
        temp.append(count)
        count=0
        
for i in range(len(temp)):
    if (temp[i]>2):
        a.append(temp[i])
        
print(a)     
print(len(a))
for i in (round_data.index):
    if (round_data.iloc[i , 5] != round_data.iloc[i-1, 5]):
        print('current round winner:',round_data.iloc[i,[1,5,7]])
        print('previous round winner:',round_data.iloc[i-1,[1,5,7]])
round_data['Type'].value_counts()
teams=['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']

maps =['de_inferno', 'de_dust2', 'de_vertigo', 'de_nuke', 'de_overpass',
       'de_train', 'de_mirage']

sides = ['T','CT']

rounds = ['Force buy', 'Eco','Semi-Eco']



#t ='Astralis'
#m = 'de_dust2'
#s = 'CT'
#r= 'Pistol round'

for t in (teams):
    for m in (maps):
        for s in (sides):
            for r in (rounds):
                test1 = round_data.loc[((round_data['Winner Clan Name']==t) & (round_data['Map']==m) & (round_data['Winner_x'] == s) & (round_data['Side']== s) & (round_data['Type'] == r))]['Type'].count()
                #print(t,m,s,r)
                #print(test1)
                #print('\n')
                #print('len temp1 is :',len(temp1))
                test2 = round_data.loc[((round_data['Looser Clan Name']==t) & (round_data['Map']==m) & (round_data['Looser'] == s) & (round_data['Side']== s) & (round_data['Type'] == r))]['Type'].count()
                #print(t,m,s,r)
                #print(test2)
                #print('\n')
                #print('len temp2 is :',len(temp2))
                result = test1/(test1+test2)*100
                print(t,m,s,r)
                print(result,'%')

teams=['Avangar', 'Astralis', 'CR4ZY', 'Team Liquid', 'NRG',
       'compLexity Gaming', 'G2 Esports', 'Renegades', 'Syman Gaming',
       'Vitality', 'DreamEaters', 'forZe eSports', 'ENCE', 'MIBR',
       'FaZe Clan', 'FURIA', 'mousesports', 'Grayhound Gaming',
       'INTZ E-SPORTS CLUB', 'HellRaisers', 'Natus Vincere',
       'Ninjas in Pyjamas', 'North', 'Tyloo']

maps =['de_inferno', 'de_dust2', 'de_vertigo', 'de_nuke', 'de_overpass',
       'de_train', 'de_mirage']

sides = ['T','CT']

rounds = ['Pistol round','Normal']

for t in (teams):
    for m in (maps):
        for s in (sides):
            for r in (rounds):
                test3 = round_data.loc[((round_data['Winner Clan Name']==t) & (round_data['Map']==m) & (round_data['Winner_x'] == s) & (round_data['Type'] == r))]['Type'].count()
                print(test3)
                #print('\n')
                
                test4 = round_data.loc[((round_data['Looser Clan Name']==t) & (round_data['Map']==m) & (round_data['Looser'] == s) &  (round_data['Type'] == r))]['Type'].count()
                print(test4)
                #print('\n')
                result=test3/(test3+test4)*100
                print(t,m,s,r)
                print(result)
t='Astralis'
test = round_data[round_data['Winner Clan Name']==t]
ax = sns.countplot(x='End reason', data=test);
plt.xticks(rotation = 'vertical');
t='Astralis'
test = round_data[(round_data['Winner Clan Name']==t) & (round_data['Team']==t)]
ax = sns.countplot(x='Type', data=test);
plt.xticks(rotation = 'vertical');
round_data.groupby(['Type','Side', 'End reason']).size()
t='Astralis'
m='de_inferno'
test = round_data[(round_data['Map'] == m) & (round_data['Winner Clan Name'] == t)]
test.groupby(['Type', 'End reason']).size()
t='Astralis'
m='de_inferno'
test = round_data.loc[(round_data['Map'] == m) & (round_data['Winner Clan Name'] == t) & (round_data['Winner_x'] == round_data['Side'])]
test.groupby(['Type', 'End reason']).size()
t='Astralis'
m='de_inferno'
test = round_data[(round_data['Map'] == m) & (round_data['Looser Clan Name'] == t) & (round_data['Looser'] == round_data['Side'])]
test.groupby(['Type', 'End reason']).size()
t='Astralis'
m='de_inferno'
test = round_data.loc[(round_data['Map'] == m) & (round_data['Looser Clan Name'] == t) & (round_data['Looser'] != round_data['Side'])]
test.groupby(['Type', 'End reason']).size()
#use this for visualization
#use this for hypothesis testing
#use this to make assumptions and check anomlies
sns.distplot(round_data['Kills'])
round_data.groupby("Kills")['Winner_x'].value_counts()
round_data.groupby('Type')['Kills'].sum()
round_data.groupby('Trade kill')['Winner_x'].value_counts()
round_data.groupby('Type')['ADP'].mean()
round_data.groupby('Winner_x')['Bomb planted'].value_counts()
round_data.groupby('Type')['Bomb planted'].value_counts()
t='T'
t_eco = round_data.loc[round_data['Side'] == t]
plot = t_eco.groupby('Type')['Bomb planted'].value_counts()
sns.countplot(x=plot, data=plot);
plot
test = round_data[(round_data['Winner_x'] == 'T') & (round_data['Side']  == "T")]
test.groupby('Type')['Bomb planted'].value_counts()
t= 'Astralis'
test= round_data[(round_data['Winner Clan Name'] == t) & (round_data['Side'] == round_data['Winner_x'])]
test.groupby(['Map','Type','End reason'])['Bomb planted'].value_counts()
t= 'Astralis'
test= round_data[(round_data['Looser Clan Name'] == t) & (round_data['Side'] == round_data['Looser'])]
test.groupby(['Map','Type','End reason'])['Bomb planted'].value_counts()
t_eco = round_data.loc[round_data['Side']=='CT']
t_eco.groupby('Type')['Start money team 1'].describe()
t_eco = round_data.loc[round_data['Side']=='T']
t_eco.groupby('Type')['Start money team 2'].describe()
round_data['Start money team 1'].describe()
round_data['Start money team 2'].describe()
sns.distplot(round_data['Start money team 1']);
sns.distplot(round_data['Start money team 2']);
t='Astralis'
test = round_data[(round_data['Name team 1'] == t) & (round_data['Winner Clan Name']== t)]['Start money team 1']
test.describe()
#sns.distplot(test);
t='Astralis'
test = round_data[(round_data['Name team 1'] == t) & (round_data['Looser Clan Name']== t)]['Start money team 1']
test.describe()
#sns.distplot(test);
t='Astralis'
test = round_data[(round_data['Name team 2'] == t) & (round_data['Winner Clan Name']== t)]['Start money team 2']
test.describe()
#sns.distplot(test);
t='Astralis'
test = round_data[(round_data['Name team 2'] == t) & (round_data['Looser Clan Name']== t)]['Start money team 2']
test.describe()
#sns.distplot(test);
t='Astralis'
test = round_data[(round_data['Winner_x']=='T') & (round_data['Winner Clan Name'] == t) & (round_data['Side'] == 'T')]
test.groupby(['Map','Type'])['Start money team 2'].describe()
t='Astralis'
test = round_data[(round_data['Looser']=='T') & (round_data['Looser Clan Name'] == t) & (round_data['Side'] == 'T')]
test.groupby(['Map','Type'])['Start money team 2'].describe()
t='Astralis'
test = round_data[(round_data['Winner_x']=='CT') & (round_data['Winner Clan Name'] == t) & (round_data['Side'] == 'CT')]
test.groupby(['Map','Type'])['Start money team 1'].describe()
t='Astralis'
test = round_data[(round_data['Looser']=='CT') & (round_data['Looser Clan Name'] == t) & (round_data['Side'] == 'CT')]
test.groupby(['Map','Type'])['Start money team 1'].describe()
t='Astralis'
test = round_data[(round_data['Looser']=='T') & (round_data['Looser Clan Name'] == t) & (round_data['Side'] == 'T')]
test.groupby(['Map','Type'])['Start money team 2'].describe()
t='Astralis'
test = round_data.loc[round_data['Name team 1'] == t]['Start money team 1']
test.describe()
sns.distplot(test);
t='Astralis'
test = round_data.loc[round_data['Name team 2'] == t]['Start money team 2']
test.describe()
sns.distplot(test);
t='Astralis'
team_economy=[]
oppo_economy = []
for i in round_data.index:
    if(round_data.iloc[i,38] == t):
       team_economy.append(round_data.iloc[i,25])
       oppo_economy.append(round_data.iloc[i,24])
    elif (round_data.iloc[i,37] == t):
        team_economy.append(round_data.iloc[i,24])
        oppo_economy.append(round_data.iloc[i,25])
team_df = round_data.loc[(round_data['Name team 1'] == t) | (round_data['Name team 2'] == t)][['Source.Name', 'Number','Winner Clan Name',
       'Winner_x', 'End reason', 'Type', 'Side', 'Team', 'Bomb Exploded', 'Bomb planted', 'Bomb defused','Winner_y', 'Looser_team_name', 'Map',
       'Name team 1', 'Name team 2', 'Looser Clan Name', 'Looser']]
team_df['team economy'] = team_economy
team_df['oppo economy']= oppo_economy

team_df.head(50)

sns.distplot(team_df['team economy']);
mean=[]
number=[]
std=[]
for i in range(1,31):
    if i <= 30:
        temp = team_df.loc[team_df['Number']== i]['team economy']
        number.append(i)
        std.append(temp.std())
        mean.append(temp.mean())
        

#print(std)
#sns.distplot(mean);
#sns.lineplot(number,mean);
std = np.asarray(std)
mean = np.asarray(mean)
plt.plot(number,mean, label='team mean economy')
plt.fill_between(number, mean-std, mean+std, color="#DDDDDD")
#plt.show()

temp2 = team_df.loc[team_df['Team'] == t]
temp2.groupby('Type')['team economy'].agg([np.mean ,np.std])
t = 'Astralis'
mean=[]
number=[]
std=[]
for i in range(1,31):
    if i <= 30:
        temp = team_df.loc[team_df['Number']== i]['oppo economy']
        number.append(i)
        std.append(temp.std())
        mean.append(temp.mean())
        

#print(std)
#sns.distplot(mean);
#sns.lineplot(number,mean);
std = np.asarray(std)
mean = np.asarray(mean)
plt.plot(number,mean, label='oppo mean economy');
plt.fill_between(number, mean-std, mean+std, color="#DDDDDD");
#plt.show()


t='Astralis'
r='Pistol round'
temp = round_data.loc[(round_data['Winner Clan Name'] == t) & (round_data['Type'] == r)]
temp.groupby(['Number','Winner_x'])['Winner_y'].value_counts()
round_data.columns
r = 'Pistol round'
win = round_data.loc[round_data['Type'] == r]
win.groupby('Map')['Winner_x'].value_counts()
r = 'Pistol round'
win = round_data.loc[round_data['Type'] == r]
temp=(win.groupby('Map')['Winner_x'].value_counts())
for i in range(0,len(temp) ,2):
    x=temp[i]
    y=temp[i+1]
    CT=x/(x+y)*100
    T=y/(x+y)*100
    print('CT side win % are',CT)
    print('T side win % are',T)
t='Astralis'
r = 'Pistol round'
win = round_data.loc[(round_data['Type'] == r) & (round_data['Winner Clan Name']== t)]
win.groupby('Map')['First_half_winner_team_name'].value_counts()
t='Astralis'
r='Pistol round'
temp = round_data.loc[(round_data['Winner Clan Name'] == t) & (round_data['Type'] == r) & (round_data['First_half_winner_team_name'] == t)]
temp.groupby(['Number', 'Winner_x'])['First_half_winner_team_name'].value_counts()
t='Astralis'
r='Pistol round'
temp = round_data.loc[(round_data['Looser Clan Name'] == t) & (round_data['Type'] == r)]
temp.groupby(['Number', 'Looser'])['First_half_winner_team_name'].value_counts()
t='Astralis'
r='Pistol round'
temp = round_data.loc[(round_data['Type'] == r) & (round_data['Looser_team_name'] == t)]
temp.groupby(['Number', 'Winner_x'])['Looser Clan Name'].value_counts()
t='NRG'
m = 'de_dust2'
temp = round_data.loc[(round_data['Map'] == m) & (round_data['Winner Clan Name'] == t)]
temp.groupby(['Type','Winner_x']).size()
r = 'Pistol round'
temp = round_data.loc[round_data['Type'] == r]
temp.groupby(['Map','Winner_x']).size()
y=[]
test = round_data[(round_data['Type'] == 'Pistol round') & (round_data['Winner_x'] == "T")]
#print(len(test))
for i in test.index:
    x = round_data.iloc[i+1,24]
    y.append(round_data.iloc[i+1,7])
    #print('2nd round economy', x)
print(len(y))
#print(y.count('Force buy'))
#print(y.count('Eco'))
#print(y.count('Semi-Eco'))
#print(y.count('Normal'))

test2 = round_data.loc[(round_data['Type'] == 'Normal') & ((round_data['Number']==2) | (round_data['Number']==17))]
test2.groupby('Type')['Winner_x'].value_counts()

maps =['de_inferno', 'de_dust2', 'de_vertigo', 'de_nuke', 'de_overpass',
       'de_train', 'de_mirage']

sides = ['T','CT']

for m in maps:
    for s in sides:
        ab = round_data[(round_data['Type'] == 'Normal') & (round_data['Map']== m)]
        a =len(ab)
        print(a)
        bc = ab.loc[(ab['Start money team 1']<23000) & (ab['Winner_x'] == s)]
        b= len(bc)
        print(b)
        c=(b/a)*100
        print("{}'s win % on {} when they < 23000 is {}".format(s,m,c))
avg = round_data[(round_data['Type'] == 'Normal') & (round_data['Start money team 1']<23000)]
avg.groupby(['Winner_x']).size()
x=round_data.iloc[:,4,]
x
from collections import Counter
a=[]
b=[]
for i in round_data.index:
    x=(round_data.iloc[i-1,[1,4,5]])
    y=(round_data.iloc[i-2,[1,4,5]])
    z=(round_data.iloc[i,[1,4,5]])
    u=(round_data.iloc[i,4,])
    #a.append(np.where(((x['Winner Clan Name'] == y['Winner Clan Name']) & (y['Winner Clan Name'] == z['Winner Clan Name'])),z[['Number','Winner Clan Name','Winner_x']],0))
    b.append(np.where(((x['Winner Clan Name'] == y['Winner Clan Name']) & (y['Winner Clan Name'] == z['Winner Clan Name'])),u,0))

print(len(b))
print(b.count('Avangar'))


players_dataset['Source.Name'] = players_dataset['Source.Name'].replace(['ASTALIS VS AVANGAR BO3-INFERNO-ASTRALIS.xlsx'], 'ASTRALIS VS AVANGAR BO3-INFERNO-ASTRALIS.xlsx')
players_dataset['Source.Name'].value_counts()
test = general_data[['Source.Name','Winner','Looser_team_name','Map','Name team 1', 'Name team 2']]
test.head()
players_data = pd.merge(players_data, test, on='Source.Name')
players_data.head(40)
players_data.shape
players_data.groupby(['Team','Name'])['Kills'].sum().sort_values(ascending=False).head(10)
df = []
teams=['Astralis', 'Avangar', 'DreamEaters', 'G2 Esports', 'Team Liquid',
       'NRG', 'mousesports', 'Renegades', 'Syman Gaming', 'HellRaisers',
       'CR4ZY', 'forZe eSports', 'ENCE', 'Vitality', 'FaZe Clan',
       'compLexity Gaming', 'INTZ E-SPORTS CLUB', 'Grayhound Gaming',
       'Natus Vincere', 'MIBR', 'Ninjas in Pyjamas', 'North', 'FURIA',
       'Tyloo']
for t in teams:
    sum_kills = players_data.loc[players_data['Team']==t]
    df.append(sum_kills.groupby(['Team','Name'])[['Name','Kills']].sum().sort_values(by='Kills', ascending=False))
    #test = pd.DataFrame(data=df,columns=['Name','Kills'])
    #test.head(50)
    
print(len(df[0]))
round_data.head(60)
t='Astralis'
test = round_data.loc[(round_data['Map'] == 'de_dust2') & (round_data['Number']<=15) & (round_data['Name team 1'] == t)]
l1 = len(test)
#print(l1)
test
t='Astralis'
m='de_inferno'
test = round_data.loc[(round_data['Winner Clan Name'] == t) & (round_data['Map']==m) & (round_data['Number']<=15) & (round_data['Name team 2'] == t) & (round_data['First_half_winner_team_name']==t)]
l1 = len(test)
print(l1)
t='G2 Esports'
m='de_inferno'
test = round_data.loc[(round_data['Map'] == m) & (round_data['Number']<=15) & (round_data['Name team 1'] == t) & (round_data['First_half_winner_team_name']==t)]
l1=test['First_half_winner_team_name'].value_counts()
p1=len(l1)
print(p1)
t='G2 Esports'
m='de_inferno'
test = round_data.loc[(round_data['Map'] == m) & (round_data['Number']<=15) & (round_data['Name team 1'] == t)]
l2=test['First_half_winner_team_name'].value_counts()
p2=len(l2)
print(p2)
p1/p2
zero=0
result=[]
teams=['Astralis', 'Avangar', 'DreamEaters', 'G2 Esports', 'Team Liquid',
       'NRG', 'mousesports', 'Renegades', 'Syman Gaming', 'HellRaisers',
       'CR4ZY', 'forZe eSports', 'ENCE', 'Vitality', 'FaZe Clan',
       'compLexity Gaming', 'INTZ E-SPORTS CLUB', 'Grayhound Gaming',
       'Natus Vincere', 'MIBR', 'Ninjas in Pyjamas', 'North', 'FURIA',
       'Tyloo']

side = ['Name team 1', 'Name team 2']
m='de_inferno'

for t in teams:
    for s in side:
        test1 = round_data.loc[(round_data['Map'] == m) & (round_data['Number']<=15) & (round_data[s] == t) & (round_data['First_half_winner_team_name']==t)]
        l1=test1['First_half_winner_team_name'].value_counts()
        p1=len(l1)
        test2 = round_data.loc[(round_data['Map'] == m) & (round_data['Number']<=15) & (round_data[s] == t)]
        l2=test2['First_half_winner_team_name'].value_counts()
        p2=len(l2)
        def result(p1,p2):
            try:
                #suppose that number2 is a float
                return p1/p2*100
            except ZeroDivisionError:
                return 0
            
        print('{} winning % on {} side on {} in first half is'.format(t,s,m))
        percent = result(p1,p2)
        print(percent)











































