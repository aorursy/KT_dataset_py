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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
print(os.listdir("../input"))
matches_data = pd.read_csv(r"../input/WorldCupMatches.csv")
players_data = pd.read_csv(r"../input/WorldCupPlayers.csv")
cups_data = pd.read_csv(r"../input/WorldCups.csv")
matches_data.describe() #20 cols
#players_data.info() #9 cols
#cups_data.info()    #10 cols
matches_data.columns
players_data.columns
matches_data.head()
#cups_data.head()
sns.heatmap(matches_data.corr())
matches_data.corr()
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('Germany FR','Germany')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('Germany FR','Germany')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('C�te d\'Ivoire','Côte d\'Ivoire')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('C�te d\'Ivoire','Côte d\'Ivoire')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('rn">Trinidad and Tobago','Trinidad and Tobago')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('rn">Trinidad and Tobago','Trinidad and Tobago')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('rn">United Arab Emirates','United Arab Emirates')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('rn">United Arab Emirates','United Arab Emirates')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('rn">Serbia and Montenegro','Serbia and Montenegro')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('rn">Serbia and Montenegro','Serbia and Montenegro')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('rn">Bosnia and Herzegovina','Bosnia and Herzegovina')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('rn">Bosnia and Herzegovina','Bosnia and Herzegovina')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('German DR','Germany')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('German DR','Germany')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('IR Iran','Iran')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('IR Iran','Iran')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('rn">Republic of Ireland','Republic of Ireland')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('rn">Republic of Ireland','Republic of Ireland')
matches_data['Home Team Name'] = matches_data['Home Team Name'].replace('Korea DPR','Korea Republic')
matches_data['Away Team Name'] = matches_data['Away Team Name'].replace('Korea DPR','Korea Republic')
total_cups_won = cups_data["Winner"].value_counts().reset_index()
plt.figure(figsize=(11,11))
plt.subplot(211)
explode = (0.1, 0, 0, 0,0.0, 0, 0, 0,0)
plt.pie(x=total_cups_won['Winner'],labels=total_cups_won['index'],autopct='%1.1f%%',explode=explode,startangle=90,shadow=True)
plt.title("Pie Chart showing Leading Team with most world cup wins")
plt.show()

plt.figure(figsize=(12,10))
plt.subplot(212)
sns.barplot(data=total_cups_won,x = 'index',y= 'Winner')
plt.xlabel("Country")
plt.ylabel("number of wins")
plt.title("Bar Plot showing Leading Team with most world cup wins")
plt.show()
cups_data[cups_data['Country'] == cups_data['Winner']][['Year','Country','Winner']]
matches_data=matches_data.drop_duplicates(subset=['MatchID'])#as there are duplicate matches 
matches_data=matches_data.dropna(axis=0,subset=['Year'])
matches_attendance = matches_data.groupby("Year")["Attendance"].sum().reset_index()
plt.figure(figsize=(15,7))
sns.set_style(style='darkgrid')
plt.grid(b=True,axis='both') 
sns.barplot(data=matches_attendance,x='Year',y='Attendance')
plt.title("Number of People attended per Year")
plt.show()
players_dataNoPor = players_data[~(players_data['Team Initials'] == 'POR')]#Removed POR team as Ronaldo (name) is in both BRA and POR Teams
max_played = players_dataNoPor['Player Name'].value_counts().reset_index().head()
plt.figure(figsize=(7,5))
sns.barplot(data=max_played,x='index',y = 'Player Name')
plt.xlabel("Player Names")
plt.ylabel("matches played")
plt.title("Top 5 players who played maximum matches in world cup")
plt.show()
winners_home = (matches_data['Home Team Goals'] > matches_data['Away Team Goals'])
winners_away = (matches_data['Home Team Goals'] < matches_data['Away Team Goals'])

matches_equal_goals = matches_data[matches_data['Home Team Goals']==matches_data['Away Team Goals']]

draw_matches = matches_equal_goals[matches_equal_goals['Win conditions'].str.len()<=1]
winners_overall = winners_home | winners_away

winners_other = matches_data['Win conditions'].str.len()>1 #.str converts series to string element wise

win_team_home = np.where(winners_home, matches_data['Home Team Name'], '') # parameters are condition, Incase if true, incase if false
win_team_away = np.where(winners_away, matches_data['Away Team Name'], '')
not_draw_matches = matches_equal_goals[matches_equal_goals['Win conditions'].str.len()>1]
wl = not_draw_matches['Win conditions'].str.split('(').str[1]
Home_goals_Penalty = wl.str.split('-').str[0]
second = wl.str.split('-').str[1]
Away_goals_Penalty= second.str.split("\)").str[0]

# splitted the win conditions column to find out who won beacuse the home team goals and away team goals are equal.so they are
# not a draw mathces 

not_draw_matches = pd.concat([not_draw_matches,Home_goals_Penalty,Away_goals_Penalty],axis=1)
not_draw_matches.columns = ['Year', 'Datetime', 'Stage', 'Stadium', 'City', 'Home Team Name',
       'Home Team Goals', 'Away Team Goals', 'Away Team Name',
       'Win conditions', 'Attendance', 'Half-time Home Goals',
       'Half-time Away Goals', 'Referee', 'Assistant 1', 'Assistant 2',
       'RoundID', 'MatchID', 'Home Team Initials', 'Away Team Initials',
       'Home_goals_Penalty', 'Away_goals_Penalty']#renaming the column names of the peanlty goals

winner_by_penalty_home = not_draw_matches['Home_goals_Penalty'].astype(str).astype(int)>not_draw_matches['Away_goals_Penalty'].astype(str).astype(int)
winners_home_Penalty = not_draw_matches[winner_by_penalty_home]['Home Team Name']
not_draw_matches['winner'] = winners_home_Penalty


winner_by_penalty_away= not_draw_matches['Home_goals_Penalty'].astype(str).astype(int)<not_draw_matches['Away_goals_Penalty'].astype(str).astype(int)
winners_away_Penalty= not_draw_matches[winner_by_penalty_away]['Away Team Name']

losers_home_Penalty = not_draw_matches[winner_by_penalty_home]['Away Team Name']
not_draw_matches['loser'] = losers_home_Penalty

losers_away_Penalty = not_draw_matches[winner_by_penalty_away]['Away Team Name']
not_draw_matches['loser'] = losers_away_Penalty

matches_data['winner'] = not_draw_matches['winner']
matches_data['winner'] = matches_data['winner'].fillna(value = matches_data[winners_home]['Home Team Name'] )
matches_data['winner'] = matches_data['winner'].fillna(value = matches_data[winners_away]['Away Team Name'] )
matches_data['winner'] = matches_data['winner'].fillna(value = 'Draw' )

matches_data['loser'] = not_draw_matches['loser']
matches_data['loser'] = matches_data['loser'].fillna(value = matches_data[winners_home]['Away Team Name'] )
matches_data['loser'] = matches_data['loser'].fillna(value = matches_data[winners_away]['Home Team Name'] )
matches_data['loser'] = matches_data['loser'].fillna(value = 'Draw' )
most_matches_won = matches_data['winner'].value_counts().reset_index()
fig = plt.figure(figsize=(16,10))
plt.grid(b=True,axis='both')
sns.set_style("darkgrid", {'grid.linestyle': '-'})
win_plt = sns.barplot(data=most_matches_won,x='index',y='winner')
for item in win_plt.get_xticklabels():
    item.set_rotation(90)
plt.xlabel("Countries")
plt.ylabel("number of matches")
plt.title('Total matches won by Teams and matches that are a draw')
plt.show()
top_10_teams = most_matches_won.drop(0,axis=0).head(10)
fig = plt.figure(figsize=(12,12))
plt.subplot(211)
ax = sns.barplot("winner","index",data=top_10_teams,
                 linewidth=1,edgecolor="k"*25)
plt.grid(True)
plt.title("Teams with the most win matches")
for i,j in enumerate("Matches Won  : " + top_10_teams["winner"].astype(str)):
    ax.text(0.5,i,j,fontsize=15,color="white")
plt.xlabel("")
plt.ylabel("")
plt.show()

most_matches_lost = matches_data['loser'].value_counts().reset_index()
bottom_10_teams = most_matches_lost.drop(0,axis=0).head(10)
fig = plt.figure(figsize=(12,12))
plt.subplot(212)
ax = sns.barplot("loser","index",data=bottom_10_teams,palette="gnuplot_r",
                 linewidth=1,edgecolor="k"*25)
plt.grid(True)
plt.title("Teams with the most lost matches")
for i,j in enumerate("Matches Lost  : " + bottom_10_teams["loser"].astype(str)):
    ax.text(0.5,i,j,fontsize=15,color="white")
plt.xlabel("")
plt.ylabel("")
plt.show()
