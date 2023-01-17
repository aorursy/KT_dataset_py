# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
matches = pd.read_csv('../input/matches.csv')
deliveries = pd.read_csv('../input/deliveries.csv')
matches.head()
matches.info()
deliveries.head()
deliveries.info()
matches['team1'].unique()
matches.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
deliveries.replace('Rising Pune Supergiant','Rising Pune Supergiants',inplace=True)
matches['team1'].unique()
type(matches['date'].iloc[0])
matches['date'] = pd.to_datetime(matches['date'])
type(matches['date'].iloc[0])

most_wins = matches[matches['result'] == 'normal']['winner'].value_counts()
most_home_wins = matches[matches['result'] == 'normal'][matches['team1'] == matches['winner']]['team1'].value_counts()
most_matches = matches['team1'].append(matches['team2']).value_counts()
most_runs = deliveries[['batsman','batsman_runs']].groupby('batsman').sum().sort_values('batsman_runs',ascending=False)
umpire_most_matches = matches['umpire1'].append(matches['umpire2']).append(matches['umpire3']).value_counts()
most_player_of_match = matches['player_of_match'].value_counts()

deliveries['dismissal_kind'].unique()
#if dismiss kind is 'run out','retired hurt' or 'obstructing the field', it is not a bwloer's wicket.
dismiss_kind = ['caught', 'bowled', 'lbw', 'caught and bowled', 'stumped', 'hit wicket']
def check_kind(kind):
    if kind in dismiss_kind:
        return kind

most_wickets = deliveries[deliveries['dismissal_kind'].apply(check_kind).notnull()]['bowler'].value_counts()
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = most_wins.head().index, y = most_wins.head().values, units = most_wins.head().index, color = 'darkslategrey')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Teams').set_size(20)
ax.set_ylabel('Number of wins').set_size(20)
ax.set_title('TEAM WITH THE MOST WINS').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = most_home_wins.head().index, y = most_home_wins.head().values, units = most_home_wins.head().index, color='indigo')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Teams').set_size(20)
ax.set_ylabel('Number of wins').set_size(20)
ax.set_title('TEAM WITH THE MOST HOME WINS').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = most_matches.head().index, y = most_matches.head().values, units = most_matches.head().index, color='limegreen')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Teams').set_size(20)
ax.set_ylabel('Matches').set_size(20)
ax.set_title('TEAMS THAT PLAYED THE MOST MATCHES').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = most_runs.head().index, y = most_runs['batsman_runs'].head(), units = most_runs.head().index, color='darkorange')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Player').set_size(20)
ax.set_ylabel('Runs').set_size(20)
ax.set_title('HIGHEST RUN SCORERS').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = umpire_most_matches.head().index, y = umpire_most_matches.head().values, units = umpire_most_matches.head().index, color='springgreen')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Umpire').set_size(20)
ax.set_title('MOST MATCHES AS AN UMPIRE').set_size(20)
ax.set_yticklabels(())
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x=most_player_of_match.head().index,y=most_player_of_match.head().values, color='cornflowerblue')
plt.tight_layout()
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Player').set_size(20)
ax.set_yticklabels(())
title = ax.set_title("MOST 'PLAYER OF THE MATCH' WINNERS").set_size(20)
ax.autoscale()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
ax = sns.barplot(x = most_wickets.head().index, y = most_wickets.head().values, units = most_wickets.head().index, color='limegreen')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Player').set_size(20)
ax.set_ylabel('Wickets').set_size(20)
ax.set_title('HIGHEST WICKET TAKERS').set_size(20)
plt.tight_layout()
for p in ax.patches:
        p.set_width(0.5)
        text = ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        
        text.set_fontsize(15)
probability_of_winning_if_win_toss = ((sum(matches[matches['toss_winner'] == matches['winner']]['toss_winner'].value_counts()))/(matches['id'].count()))*100
print("There is a {}% chance of the team winning the match if they win the toss.".format(probability_of_winning_if_win_toss.round(2)))
matches_no_result = matches[matches['result'] == 'no result']['id'].count()
print("There were {} matches with No Result.".format(matches_no_result))
matches_interrupted_with_result = matches[matches['dl_applied'] == 1]['dl_applied'].count()
print("There were {} matches that were interrupted but had a result.".format(matches_interrupted_with_result))
overs_with_the_highest_wickets = deliveries[deliveries['player_dismissed'].notnull()]['over'].value_counts()
plt.figure(figsize=(10,8))
ax = sns.barplot(x = overs_with_the_highest_wickets.index, y = overs_with_the_highest_wickets.values,palette='BuGn_d')
ax.set_title('OVERS WITH THE HIGHEST WICKETS').set_size(20)
ax.set_xlabel('Over').set_size(20)
ax.set_ylabel('Wickets').set_size(20)
plt.tight_layout()
win_percent = ((most_wins/most_matches)*100).round(2)
home_win_percent = ((most_home_wins/most_matches)*100).round(2)
team_win_percentage = pd.concat([most_matches,most_wins,win_percent,most_home_wins,home_win_percent],axis=1)
team_win_percentage.columns=['Total Matches','Won','Win Percentage','Home Wins', 'Home Win Percentage']
team_win_percentage.style.apply(lambda x: ['background: lightsteelblue' for i in x])
plt.figure(figsize=(15,8))
sns.set_style("whitegrid")
ax = sns.barplot(x=team_win_percentage.index,y='Total Matches',data=team_win_percentage, color='darkgrey', label="Total Matches")
plt.tight_layout()
ax = sns.barplot(x=team_win_percentage.index,y='Won',data=team_win_percentage, color='orchid', label="Total Wins")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Teams').set_size(20)
ax.set_ylabel('Matches').set_size(20)
title = ax.set_title('WIN PERCENTAGE').set_size(20)
ax.legend(loc=0)
ax.autoscale()
for val,p in zip(win_percent.values,ax.patches):
        text = ax.annotate("%.2f" % val + "%", (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        text.set_fontsize(15)
plt.figure(figsize=(15,8))
sns.set_style("whitegrid")
ax = sns.barplot(x=team_win_percentage.index,y='Total Matches',data=team_win_percentage, color='darkgrey', label="Total Matches")
plt.tight_layout()
ax = sns.barplot(x=team_win_percentage.index,y='Home Wins',data=team_win_percentage, color='royalblue', label="Home Wins")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('Teams').set_size(20)
ax.set_ylabel('Matches').set_size(20)
title = ax.set_title('HOME WIN PERCENTAGE').set_size(20)
ax.legend(loc=0)
ax.autoscale()
for val,p in zip(home_win_percent.values,ax.patches):
        text = ax.annotate("%.2f" % val + "%", (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', rotation=0, xytext=(0, 20), textcoords='offset points')
        text.set_fontsize(15)

overs, number = np.unique(np.concatenate(deliveries.groupby(['match_id','inning'])['over'].unique().values), return_counts=True)
average_runs_in_each_over = ((deliveries.groupby(['over'])['total_runs'].sum())/(number)).round(2)
plt.figure(figsize=(15,5))
sns.set_style("whitegrid")
ax = sns.barplot(x=average_runs_in_each_over.index,y=average_runs_in_each_over.values,palette='Blues_d')
ax.set_xlabel("Overs").set_size(20)
ax.set_ylabel("Runs").set_size(20)
ax.set_title("AVERAGE RUNS IN EACH OVER").set_size(20)
finalists = matches[matches['id'].apply(lambda id: id in matches.groupby('season')['id'].max().values)].sort_values(by=['season'])[['season','city','date','team1','team2','toss_winner','toss_decision','player_of_match', 'winner']]
finalists.style.apply(lambda x: ['background: royalblue' if x.name == 'winner' else 'background: lightsteelblue' for i in x])
