import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
file_path = '../input/matches.csv'
df = pd.read_csv(file_path)
df.head()
print (df.shape)
print (df.columns)
df.dtypes
df.isnull().sum()
df.drop(['id','umpire3'],axis=1,inplace=True)
df.replace('Rising Pune Supergiant','Rising Pune Supergiants', inplace=True)
all_teams = np.unique(df['team1'])
print (all_teams)
all_city = df['city'].unique()
all_city = all_city[:-1]
print (all_city)
all_stadium = np.unique(df['venue'])
print (all_stadium)
all_umpires = set(df['umpire1']).union(set(df['umpire2']))
all_umpires = list(all_umpires)
all_umpires = all_umpires[1:]
print (all_umpires)
print ("Total matches played : {}\n".format(df.shape[0]))
print ("Number of different cities : {}\n".format(len(all_city)))
print ("Number of different venues : {}\n".format(len(all_stadium)))
print ("Total umpires : {}".format(len(all_umpires)))
df = df[df['result'] != 'no result']
df.shape
df_umpires = pd.concat([df['umpire1'], df['umpire2']]) 
df_umpires.value_counts().head(10).plot.bar(figsize=(10,6),colormap='Accent')

plt.xticks(fontsize=12)
plt.xlabel('Umpire',fontsize=14)
plt.ylabel('Number of matches',fontsize=16)
plt.title('Top 10 umpires',fontsize=18)
df_stadium = df['venue'].value_counts()
print ("Number of matches played at different stadiums :\n\n{}\n".format(df_stadium))

df_stadium.sort_values(ascending=True).plot.barh(figsize=(10,12), colormap='summer')

plt.yticks(fontsize=12)
plt.xlabel('Number of matches played',fontsize=14)
plt.ylabel('Stadiums',fontsize=16)
plt.title('Number of matches played at different stadiums',fontsize=18)
df_toss_decision = 100 * (df['toss_decision'].value_counts()) / df.shape[0]
print ("Toss decisions in terms of % :\n{}\n".format(df_toss_decision))
sns.set(rc={'figure.figsize':(10,6)})
plt.xticks(fontsize=14)
sns.countplot(x='season', hue='toss_decision', data=df)
df_team_matches = (df.team1.value_counts() + df.team2.value_counts()).sort_values(ascending=False)
print ("Number of matches played by each team :\n\n{}\n".format(df_team_matches))

df_team_matches.plot(kind='bar',figsize=(10,6))

plt.xticks(fontsize=12)
plt.xlabel('Teams',fontsize=14)
plt.ylabel('Number of matches played',fontsize=16)
plt.title('Number of matches played by each team',fontsize=18)
df_winner = df['winner'].value_counts().sort_values(ascending=False)
df_winner_per = 100 * df_winner[all_teams]/ df_team_matches[all_teams]

print ("Number of matches won by each team :\n\n{}\n".format(df_winner))
print ("Percentage of matches won by each team :\n\n{}\n".format(df_winner_per))

df_winner_per.sort_values(ascending=False).plot(kind='bar',figsize=(10,6), colormap='summer')

plt.xticks(fontsize=12)
plt.ylim(0,100)
plt.xlabel('Teams',fontsize=14)
plt.ylabel('% of matches won',fontsize=16)
plt.title('Percentage of matches won by each team',fontsize=18)
df_toss = df.toss_winner.value_counts()
df_toss_num = df_toss[all_teams].sort_values(ascending=False)
df_toss_per = 100 * df_toss[all_teams]/df_team_matches[all_teams].values

print ("Number of times a team has won the toss : \n")
print (df_toss_num)

print ("\nPercentage a team has won the toss : \n")
print (df_toss_per)

df_toss_per.sort_values(ascending=False).plot(kind='bar',figsize=(10,6))

plt.xticks(fontsize=12)
plt.xlabel('Teams',fontsize=16)
plt.ylim(0,100)
plt.ylabel('%',fontsize=14)
plt.title('Percetange a team has won the toss',fontsize=18)
df_toss_winner = df[df.toss_winner == df.winner]['winner']
df_toss_winner_count = df_toss_winner.value_counts()
df_toss_winner_per = 100 * df_toss_winner.value_counts()[all_teams]/df_team_matches[all_teams].values
toss_match_win_per = df_toss_winner.count()/float(df.shape[0])

print ("Percentage of matches won by teams after winning the toss : {}\n".format(toss_match_win_per))

print ("Number of matches won by teams after winning the toss : \n")
print (df_toss_winner_count)

print ("\nPercentage of matches won by teams after winning the toss : \n")
print (df_toss_winner_per)

df_toss_winner_per.sort_values(ascending=False).plot(kind='bar',figsize=(10,6),colormap='Dark2')

plt.xticks(fontsize=12)
plt.xlabel('Teams',fontsize=16)
plt.ylim(0,100)
plt.ylabel('%',fontsize=14)
plt.title('Percentage of matches won after winning the toss',fontsize=18)
df_toss_winner_field = df[(df.toss_winner == df.winner) & (df.toss_decision == 'field')]['winner']
df_toss_winner_field_count = df_toss_winner_field.value_counts()
df_toss_winner_field_per = 100 * df_toss_winner_field.value_counts()[all_teams]/df_team_matches[all_teams].values
df_toss_winner_field_total_per = df_toss_winner_field.count()/float(df.shape[0])

print ("Percentage of matches won by teams after winning the toss and electing to field first : {}\n".format(df_toss_winner_field_total_per))

print ("Number of matches won by teams after winning the toss and electing to field first: \n")
print (df_toss_winner_field_count)

print ("\nPercentage of matches won by teams after winning the toss and electing to field first: \n")
print (df_toss_winner_field_per)

df_toss_winner_field_per.sort_values(ascending=False).plot(kind='bar',figsize=(10,6))

plt.xticks(fontsize=12)
plt.xlabel('Teams',fontsize=16)
plt.ylim(0,100)
plt.ylabel('%',fontsize=14)
plt.title('Percentage of matches won after winning the toss and electing to field first',fontsize=18)
df_toss_winner_bat = df[(df.toss_winner == df.winner) & (df.toss_decision == 'bat')]['winner']
df_toss_winner_bat_count = df_toss_winner_bat.value_counts()
df_toss_winner_bat_per = 100 * df_toss_winner_bat.value_counts()[all_teams]/df_team_matches[all_teams].values
df_toss_winner_bat_total_per = df_toss_winner_bat.count()/float(df.shape[0])

print ("Percentage of matches won by teams after winning the toss and electing to bat first : {}\n".format(df_toss_winner_bat_total_per))

print ("Number of matches won by teams after winning the toss and electing to bat first: \n")
print (df_toss_winner_bat_count)

print ("\nPercentage of matches won by teams after winning the toss and electing to bat first: \n")
print (df_toss_winner_bat_per)

df_toss_winner_bat_per.sort_values(ascending=False).plot(kind='bar',figsize=(10,6),colormap='Set2')

plt.xticks(fontsize=12)
plt.xlabel('Teams',fontsize=16)
plt.ylim(0,100)
plt.ylabel('%',fontsize=14)
plt.title('Percentage of matches won after winning the toss and electing to bat first',fontsize=18)
total_teams = len(all_teams)
heatmap_scores = np.zeros((total_teams, total_teams))
for i,team1 in enumerate(all_teams):
    for j,team2 in enumerate(all_teams):
        matches_played = 0
        if team1 != team2:
            df_winner = df[(df.team1 == team1) & (df.team2 == team2)]['winner']
            matches_played = df_winner.shape[0]
            #print team1, team2, matches_played
            heatmap_scores[i,j] += matches_played
            heatmap_scores[j,i] += matches_played
        
fig, ax = plt.subplots(1, 1)
fig.set_figheight(7.5)
fig.set_figwidth(10)

ax1 = sns.heatmap(heatmap_scores, xticklabels = all_teams, yticklabels = all_teams, linewidths = 0.5, annot = True, cmap="OrRd", ax = ax)

ax1.set_title("No. of Matches played TeamA vs TeamB",fontsize=16)

fig.tight_layout()
df['player_of_match'].value_counts().head(10).plot.bar(figsize=(10,6), color='R')

plt.xticks(fontsize=12)
plt.ylim(0,30)
plt.xlabel('Player',fontsize=14)
plt.ylabel('Number of matches',fontsize=16)
plt.title('Top 10 players with most no. of MoMs',fontsize=18)