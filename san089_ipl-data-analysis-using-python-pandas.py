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
df = pd.read_csv('../input/matches.csv')
#Selecting top 3 rows from the dataframe
df.head(3)
# Fetching number of rows in the dataframe
df.shape[0]
df['winner'].sort_values().value_counts()
df.loc[df['winner'] == 'Rising Pune Supergiants', 'winner'] = 'Rising Pune Supergiant'
#But we have to replace in other series as well like 'team1', 'team2',etc.
df[['team1','team2','toss_winner','winner']] = df[['team1','team2','toss_winner','winner']].apply(lambda val : val.str.replace('Rising Pune Supergiants','Rising Pune Supergiant') )
df['winner'].sort_values().value_counts()
# Grouping the number of matched won by teams and sorting it based on the count of the matches.
winners_df = df.groupby('winner', sort = False).count().id.sort_values(ascending = False)
winners_df
# A simple plot for the winners.
winners_df.plot(kind = 'bar', title = 'Number of matches won in IPL', label = 'Number of wins' )
# case where team won the toss and also won the match
winning_toss_winning_match = df[(df['toss_winner'] == df['winner'])]
# cases where team won the toss but lost the match
winning_toss_losing_match = df[(df['toss_winner'] != df['winner'])]
new_df = pd.DataFrame({"Team won the toss":[ winning_toss_winning_match.shape[0],winning_toss_losing_match.shape[0]  ]}, index = ['Won match', 'Lost match']  )
#Let's plot the above obervations
new_df.plot( kind = 'bar' , legend = False , title = "Number of time a team won the toss and the match result of that match")
# Same as above, now we will check how many times team winning the toss and  batting first won and how many times team bowling first won
# For this we have to check if the team won the toss and toss_decision is bat/field and winning team is toss_winner
# NOTE : in this type of analysis we are considering that the team has won the toss and its their decision to select to bat or bowl
team_batting_first_won = df.loc[(df['toss_winner'] == df['winner']) & (df['toss_decision'] == 'bat'), ['id', 'winner']]
team_fielding_first_won = df.loc[(df['toss_winner'] == df['winner']) & (df['toss_decision'] == 'field'),  ['id', 'winner']]
team_fielding_first_won_df= team_fielding_first_won['winner'].value_counts()
team_batting_first_won_df = team_batting_first_won['winner'].value_counts()
team_batting_first_won_df.plot(kind = 'bar', title = 'Team winning toss, batting first and winning the match.')
team_fielding_first_won_df.plot(kind = 'bar', title = 'Team winning toss, fielding first won the match. ')
df['city'].value_counts()
#Two ways to do the same
df.loc[df['city'] == 'Bangalore', 'city'] = 'Bengaluru'
#Another way 
#Changing city name from Bangalore to Bengaluru
df['city'] = df['city'].str.replace('Bangalore','Bengaluru')
#Creating a dataframe, which holds the records where the match result is decided by "Duckworth-Lewis method".
new_df = df.loc[ df['dl_applied'] != 0 ,['id','team1','team2','toss_winner','toss_decision','dl_applied','winner']]
# Now we have to create a new column to decide which team has batted first for each match. I am gonna create a function for it and apply 
# that function to the dataframe ***

def set_batting_first(team1, team2, toss_winner, toss_decision):
    if toss_decision == 'bat':
        return toss_winner
    else:
        if team1 == toss_winner:
            return team2
        else:
            return team1  
for items in new_df.loc[: , ['team1','team2','toss_winner','toss_decision','bat_first']].itertuples():
    new_df.loc[items[0] , 'bat_first'] = set_batting_first(items[1], items[2], items[3], items[4])
#Now we are gonna create 1 more column named "is_bat_first_win" which will have 1 if the team batting first won and 0 if team batting 1 lost.
new_df['is_bat_first_win'] = 0
new_df.loc[new_df['bat_first'] == new_df['winner'],'is_bat_first_win'] = 1
#Lets create a new dataframe with just two results as below.
dl_bat_first_win = pd.DataFrame(new_df['is_bat_first_win'].value_counts())
dl_bat_first_win.index = ['field_first_wins','bat_first_wins']
dl_bat_first_win.plot(kind = 'bar')
# New dataframe where DL method is not applied and win margin is 100 or more runs or more than 8 wickets.
new_df = df.loc[ (df['dl_applied'] == 0) & ((df['win_by_runs'] > 99) | (df['win_by_wickets'] > 8)) , ['id','team1','team2','winner','win_by_runs','win_by_wickets']]
#creating a new column to track the losing team 
new_df['losing_team'] = np.nan
# I will create a method to fetch the losing team from the dataframe
def check_losing_team(row):
    if row['team1'] == row['winner']:
        row['losing_team'] = row['team2']
        return row['team2']
    else:
        row['losing_team'] = row['team1']
        return row['team1']
new_df['losing_team'] = new_df.apply(check_losing_team, axis = 1)
new_df.losing_team.value_counts().plot(kind = 'bar', title = 'Number of times team losing with big margin')
# number of times the playes got Man of the match
mom_players = df.player_of_match.value_counts()
# we want to analyse only those players who got the award atleast 10 times
mom_players[mom_players >= 10 ].plot(kind = 'bar', title = 'Number of times a player got Man of the Match')