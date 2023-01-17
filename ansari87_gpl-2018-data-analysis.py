import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
gpl_data=pd.read_csv("../input/cleaned-data.csv")
# Displaying first 5 records
gpl_data.head(5)
%matplotlib inline
# calculating bowling average
gpl_data['bowling_average']=gpl_data['runs_conceded']/gpl_data['wickets']
# Innings and player count in each inning
sns.countplot(gpl_data['inning'])
# Observations
# Not a single player played 4 innings
# Another one is that, no of players' who played total 6 innings are greater than who played total 5 innings
# Both of these anomalies can be explained by, how the macthes were scheduled
# Runs scored by players and innings count
sns.lmplot(x='runs_scored', y='inning', data=gpl_data)
# Observations
# 1. Some players played more than 3 innings but scored no run--> they must be bowlers
# 2. Players who scored more than 60 runs, played at least 5 innings, with exception of one player who scored more than 80 in just 2 innings
# 3. Players who played just 1 inning were not able to score more than 40 runs
# 4. Points on top right represent outstanding batsmen
# Runs scored distribution
ax = sns.factorplot(x='runs_scored',data=gpl_data,kind='count', size=6,aspect=2)
ax.set_xticklabels(fontsize=10, rotation=45)
plt.show()
# Observations
# 1. 16 players were not able to score a single run
# 2. 2 Players scored more than 100 runs
# 3. Total 13 Players scored exactly 1 run
# Runs conceded by players and innings count
sns.lmplot(x='runs_conceded', y='inning', data=gpl_data)
# Observations
# 1. Most of the players who played 3 innings or less, conceded 40 or less runs
# 2. Points on the top left reprsent some outstanding bowlers who conceded very less runs in most no. of innings
# Runs conceded distribution
ax = sns.factorplot(x='runs_conceded',data=gpl_data,kind='count', size=6,aspect=2)
ax.set_xticklabels(fontsize=10, rotation=45)
plt.show()
# Observations
# 1. 2 players conceded 0 runs --> they must be batsmen
# 2. 7 runs were conceded by more than 10 bowlers
# 3. 76 is the most run conceded by a bowler
# Wickets taken by players and innings count
sns.lmplot(x='wickets', y='inning', data=gpl_data)
# Observations
# 1. Most of the players who played 3 or less innings were able to take at most 2 wickets,
# with exception of players(s) who palyed 2 innings and took 4 wickets
# Distribution of Wickets taken by player
ax = sns.factorplot(x='wickets',data=gpl_data,kind='count', size=6,aspect=2)
ax.set_xticklabels(fontsize=15, rotation=0)
plt.show()
# Observations
# 1. More than 60 players were not able to take a single wicket
# 2. More than 30 players took 1 wicket
# 3. 10 is the most number of wickets taken by a player
# Getting top players' who scored most runs
top_scorer=gpl_data.nlargest(10,'runs_scored').sort_values(['runs_scored'],ascending=0).reset_index(drop=True)
# Top scorers(in descending order)
ax=sns.barplot(top_scorer['player'],top_scorer['runs_scored'])
ax.set_xticklabels(labels=top_scorer['player'], fontsize=14, rotation=90)
plt.show()
# Runs and innings relation
ax1 = sns.factorplot(x='inning',data=top_scorer,kind='count', size=4,aspect=1)
ax1.ax.set_title("No. of innings played by top scorers")
ax1.set(xlabel='Inning(s)', ylabel='No. of player(s)')

plt.figure(figsize=(14,8))
ax=sns.barplot(top_scorer['inning'],top_scorer['runs_scored'], hue=top_scorer['player'])
plt.show()
# Observations
# 1. Top scorer players mostly played 5 or 6 innings.
# 2. With exception of sagar.pansare, he scored 93 in just 2 innings
# Wickets taken by top scorers
ax1 = sns.factorplot(x='wickets',data=top_scorer,kind='count', size=4,aspect=1)
ax1.ax.set_title("No. of wickets taken by top scorers")
ax1.set(xlabel='Wicket(s)', ylabel='No. of player(s)')
# Observations
# 1. Out of 10 top scorer players, 3 were not able to take a single wicket
# 2. 4 top scorer were able to take 3 or more wickets
# Overview of top scorer players
ax= top_scorer[['runs_scored','runs_conceded','wickets','inning']].plot.bar(figsize=(16,4))
labels = []
for item in top_scorer['player']:
    labels.append(item[:10] + '...')
ax.set_xticklabels(labels, rotation=45, fontsize=10)
plt.show()
# Observations 
# 1. Out of 10 top scorer players, 2 players conceded more runs than they scored
# Finding players who took most no of wickets
top_wicket_taker = gpl_data.nlargest(10,'wickets').sort_values(['wickets'],ascending=0)
# ShowinMost no. of wickets taken by a player
ax=sns.barplot(top_wicket_taker['player'],top_wicket_taker['wickets'])
ax.set_xticklabels(labels=top_wicket_taker['player'], fontsize=14, rotation=90)
plt.show()
# Wickets taken and innings relation
plt.figure(figsize=(14,6))
ax=sns.barplot(top_wicket_taker['inning'],top_wicket_taker['wickets'], hue=top_wicket_taker['player'])
plt.show()
# Observataions
# 1. Most top scorer players played at least 5 innings, with an exception of arif.mulla, he took 4 wickets in just 2 innings
# Bowling average
plt.figure(figsize=(10,6))
ax=sns.barplot(top_wicket_taker['player'],top_wicket_taker['bowling_average'])
ax.set_xticklabels(labels=top_wicket_taker['player'], fontsize=16, rotation=90)
ax.set_ylabel("Bowling Average")
plt.show()
# Observations
# Bowling average = total runs conceded / no of wickets taken
# 1. omika.ingle is not only the highest wicket taker but she has one of the best bowling average too
# Overview of top wicket taker bowlers
ax= top_wicket_taker[['runs_scored','runs_conceded','wickets','inning']].plot.bar(figsize=(16,4))
labels = []
for item in top_wicket_taker['player']:
    labels.append(item[:15] + '...')
ax.set_xticklabels(labels, rotation=30, fontsize=10)
plt.show()
# Filtering players who have bowling average
player_with_bowling_average = gpl_data[~gpl_data.isin([np.nan, np.inf, -np.inf, np.NAN, np.NaN]).any(1)]
# Wickets taken by players and innings count
sns.lmplot(x='wickets', y='bowling_average', data=player_with_bowling_average)
# Observations
# NOTE : Lower bowling average is better
# 1. Players who took more wickets, tend to have less bowling average(which is a sign of a good bowler)
# 2. Apparently there is a bowler who took a wicket without conceding a single run
# 3. The bowlers with highest bowling average were able to take only a single wicket
# Top bowlers based on bowling average
best_bowlers = gpl_data.nsmallest(10,'bowling_average').sort_values(['bowling_average'],ascending=1) 
# Bowlers with bowling average, wickets and innings
ax=sns.barplot(best_bowlers['player'],best_bowlers['bowling_average'])
ax.set_xticklabels(labels=best_bowlers['player'], fontsize=14, rotation=90)
ax.set_title("Fig.1 - Players' Bowling Average")
plt.show()

ax1=sns.barplot(best_bowlers['player'],best_bowlers['inning'])
ax1.set_xticklabels(labels=best_bowlers['player'], fontsize=14, rotation=90)
ax1.set_title("Fig.2 - Players' innings count")
plt.show()

ax2=sns.barplot(best_bowlers['player'],best_bowlers['wickets'])
ax2.set_xticklabels(labels=best_bowlers['player'], fontsize=14, rotation=90)
ax2.set_title("Fig.3 - Players' wicket count")
plt.show()
# Observations
# NOTE - Lower bowling average better 
# 1. ashwini.surve is the best bowler with 0 average, and took 1 wicket
# 2. 7 female members with best bowling average
# 3. Fig.2 - Most of the best bowlers(based on bowling average) played 1 or 2 innings only
# 4. Fig.3 - Most of the best bowlers took 1 or 2 wickets only(coz they played at most 2 innings)
# Finding players who are in both top scorer and top wicket taker list
pd.merge(top_scorer, top_wicket_taker, how='inner', on=['player','inning','runs_scored','runs_conceded','wickets', 'bowling_average'])
# Best bowlers and top wicket takers
pd.merge(top_wicket_taker, best_bowlers, how='inner', on=['player','inning','runs_scored','runs_conceded','wickets', 'bowling_average'])
# Observations
# 1. avdhesh.tiwari, nikhil.katekhaye and kapil.uddharwar not only scored top runs but also took most wickets in thetournament.
# 2. omika.ingle and shweta.chauhan not only took most wickets but they also have one of the best bowling average in the tournament.