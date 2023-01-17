# Importing libraries

import numpy as np

import pandas as pd

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings('ignore')
# Importing the dataset

df = pd.read_csv('../input/results.csv')

df.head()
df.isnull().any().any()
df = df.dropna(axis=0)

for i in range(len(df)):

    if df.iat[i, 0].find('-') != -1:

        df.iat[i, 0] = int(df.iat[i, 0][:str(df.iat[i,0]).find('-')])

df['Year Scale'] = df['date'].apply(lambda x:(('Old Match(<1976)' , 'Middling Match(1976<.<2000)')[x > 1976],'Modern Match(>2000)')[x > 2000])
# The Sorted Played Times Count of Different Team

home_team_Count = df.groupby(by=['home_team'])['tournament'].agg({'Count': np.size})

home_team_Count['Count'] = home_team_Count['Count'].astype(int)

home_team_Count = home_team_Count.sort_values(by = 'Count', ascending=False)

home_team_Count.head()
# The Sorted Played Times Count of Different Team

AwayTeamCount = df.groupby(by=['away_team'])['tournament'].agg({'Count': np.size})

AwayTeamCount['Count'] = AwayTeamCount['Count'].astype(int)

AwayTeamCount = AwayTeamCount.sort_values(by = 'Count', ascending=False)

AwayTeamCount.head()
#Visualization for Count

TopHomeTeam = np.array(home_team_Count.head(15).index)

TopHomeTeamData = df[df['home_team'].isin(TopHomeTeam)]

TopAwayTeam = np.array(AwayTeamCount.head(15).index)

TopAwayTeamData = df[df['away_team'].isin(TopAwayTeam)]

f, axes = plt.subplots(2, 1, figsize=(14,23))

plt.sca(axes[0])

plt.title('The Played Times as Home Team(Top15)', fontsize = 20, weight = 'bold')

ax = sns.countplot(data=TopHomeTeamData, y='home_team', order=TopHomeTeam, hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Match Count', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)



plt.sca(axes[1])

plt.title('The Played Times as Away Team(Top15)', fontsize = 20, weight = 'bold')

ax = sns.countplot(data=TopAwayTeamData, y='away_team', order=TopAwayTeam, hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Match Count', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()
# Visualization for Score

fig = plt.figure(1, figsize=(15,11))

plt.title('Score as Home Team for top 15 played Times', fontsize = 20, weight = 'bold')

ax = sns.lvplot(data=TopHomeTeamData, x='home_ft', y='home_team',order=TopHomeTeam,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()
# Visualization for Score

fig = plt.figure(1, figsize=(15,11))

plt.title('Score as Away Team for top 15 played Times', fontsize = 20, weight = 'bold')

ax = sns.lvplot(data=TopAwayTeamData, x='away_ft', y='away_team',order=TopAwayTeam,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()
# The Count of Different Tournament

TournamentCount = df.groupby(by=['tournament'])['away_team'].agg({'Count': np.size})

TournamentCount['Count'] = TournamentCount['Count'].astype(int)

TournamentCount = TournamentCount.sort_values(by = 'Count', ascending=False)

TournamentCount.head()
#Visualization for Count

fig = plt.figure(1, figsize=(15,13))

TopTournament = np.array(TournamentCount.head(15).index)

TopTournamentData = df[df['tournament'].isin(TopTournament)]

ax = sns.countplot(data= TopTournamentData, y ='tournament',hue = 'Year Scale',order=TopTournament)

plt.title('The Count of Different Tournaments(Top15)', fontsize = 20, weight = 'bold')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Tournament count', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()
#Visualization foe Score in Different Tournament

f, axes = plt.subplots(2, 1, figsize=(14,20))

plt.sca(axes[0])

TopTournament = np.array(TournamentCount.head(15).index)

TopTournamentData = df[df['tournament'].isin(TopTournament)]

plt.title('The Home Score for Different Tournement', fontsize = 20, weight = 'bold')

ax = sns.lvplot(data=TopTournamentData, x='home_ft', y='tournament',order=TopTournament,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Home Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)



plt.sca(axes[1])

plt.title('The Away Score for Different Tournement', fontsize = 18, weight = 'bold')

ax = sns.lvplot(data=TopTournamentData, x='away_ft', y='tournament',order=TopTournament,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Away Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()
# Count for Cities

CityCount = df.groupby(by=['city'])['tournament'].agg({'Count': np.size})

CityCount['Count'] = CityCount['Count'].astype(int)

CityCount = CityCount.sort_values(by = 'Count', ascending=False)

CityCount.head()
# Visualization for Count 

fig = plt.figure(1, figsize=(15,11))

TopCity = np.array(CityCount.head(15).index)

TopCityData = df[df['city'].isin(TopCity)]

ax = sns.countplot(data= TopCityData, y ='city',hue = 'Year Scale',order=TopCity)

plt.title('The Count of Matches of Different Cities(Top15)', fontsize = 20, weight = 'bold')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

ax.yaxis.label.set_visible(False)

plt.xlabel('Match count', fontsize=16, weight = 'bold', labelpad=10)

plt.show()
# Visualization for score

f, axes = plt.subplots(2, 1, figsize=(14,20))

plt.sca(axes[0])

plt.title('The Home Score in Different Cities', fontsize = 20, weight = 'bold')

ax = sns.lvplot(data=TopCityData, x='home_ft', y='city',order=TopCity,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Home Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)



plt.sca(axes[1])

plt.title('The Away Score in Different Cities', fontsize = 20, weight = 'bold')

ax = sns.lvplot(data=TopCityData, x='away_ft', y='city',order=TopCity,hue='Year Scale')

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

plt.xlabel('Away Score', fontsize=16, weight = 'bold', labelpad=10)

ax.yaxis.label.set_visible(False)

plt.show()