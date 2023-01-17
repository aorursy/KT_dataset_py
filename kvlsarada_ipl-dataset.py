# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.getcwd()
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
os.getcwd()
matches = pd.read_csv('../input/matches.csv')
deliveries = pd.read_csv('../input/deliveries.csv')
matches.head()
deliveries.head()
#Checking columns in matches and deliveries
matches.columns
deliveries.columns
#Abbriviate team names - Replace the full names with abbriviations
teamNames = matches.team1.unique() 
teamAbbriviations = ['SRH', 'MI', 'GL', 'RPS', 'RCB',
                           'KKR', 'DD', 'KXIP', 'CSK', 'RR', 'DC',
                           'KTK', 'PW', 'RPS']
matches.replace(teamNames, teamAbbriviations, inplace = True)
deliveries.replace(teamNames,teamAbbriviations, inplace = True)
matches.head()
deliveries.head()
#Tracking the performance of Teams over seasons
#groupby(team).count(winner) = Number of times the team has been a winner
#bar plot of wins per seasons for top 10 teams
seasons = matches.season.unique()
matches["type"] = "pre-qualifier"
for year in range(2008, 2017):
    final_match_index = matches[matches['season']==year][-1:].index.values[0]
    matches = matches.set_value(final_match_index, "type", "Final")
    matches = matches.set_value(final_match_index-1, "type", "qualifier-2")
    matches = matches.set_value(final_match_index-2, "type", "eliminator")
    matches = matches.set_value(final_match_index-3, "type", "qualifier-1")
    
matches.head()
matches.groupby(['type']).size()
winteam = matches.groupby(['season','winner', 'type']).size().reset_index()
type(winteam)
winteam
winteam.columns = ['Season', "Team", 'Type','Wins']
plt.figure(figsize=(24,20))
sns.set_style("darkgrid")
winteam['Team'].value_counts().plot(kind='bar')
#sns.barplot(data=winteam, x='Season',y='Wins', hue='Team')

plt.figure(figsize=(24,20))
sns.set_style("darkgrid")
sns.barplot(data=winteam, x='Team',y='Wins')

ct = pd.crosstab(winteam.Season, winteam.Team)
ct
#plot stacked bar chart
plt.figure(figsize=(15,13))
sns.set_style("darkgrid")
ct.plot.bar(stacked=True)
plt.show()

#Player of the match

matches['player_of_match'].value_counts()[:10].plot(kind='bar')

#Best Batsmen
batsman_grp = deliveries.groupby(["match_id", "inning", "batting_team", "batsman"])
batsmen = batsman_grp['batsman_runs'].sum().reset_index()

#Ignoring wide runs
deliveries_faced = deliveries[deliveries.wide_runs==0]
deliveries_faced = deliveries_faced.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()
deliveries_faced.columns = ["match_id", "inning", "batsman", "balls_faced"]
deliveries_faced
batsmen = batsmen.merge(deliveries_faced, left_on=["match_id", "inning", "batsman"],
                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen
#Count the 4's and 6's
batsman_4s = deliveries[deliveries.batsman_runs==4]
batsman_6s = deliveries[deliveries.batsman_runs==6]

fours_per_batsman = batsman_4s.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()
sixes_per_batsman = batsman_6s.groupby(["match_id", "inning", "batsman"])["batsman_runs"].count().reset_index()

fours_per_batsman.columns = ['match_id', 'inning', 'batsman', 'fours']
sixes_per_batsman.columns = ['match_id', 'inning', 'batsman', 'sixes']

batsmen = batsmen.merge(fours_per_batsman, left_on=["match_id", "inning", "batsman"],
                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen = batsmen.merge(sixes_per_batsman, left_on=["match_id", "inning", "batsman"],
                        right_on=["match_id", "inning", "batsman"], how="left")
batsmen.head()
batsmen['strike_rate'] = np.round(batsmen.batsman_runs/batsmen.balls_faced *100, 2)
batsmen.head()
for col in ["batsman_runs", "fours", "sixes", "balls_faced", "strike_rate"]:
    batsmen[col] = batsmen[col].fillna(0)

batsmen = matches[['id','season']].merge(batsmen, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
batsmen.head(2)

runs_per_batsman = batsmen.groupby(['season', 'batting_team', 'batsman'])['batsman_runs'].sum().reset_index()
runs_per_batsman.head()

batsmen_runs_per_season =  runs_per_batsman.groupby(['season', 'batsman'])['batsman_runs'].sum().unstack().T
batsmen_runs_per_season['Totalruns'] = batsmen_runs_per_season.sum(axis=1)
batsmen_runs_per_season = batsmen_runs_per_season.sort_values(by='Totalruns', ascending=False).drop('Totalruns',1)
batsmen_runs_per_season.head()
#Batsmen with highest runs
plt.figure(figsize=(15,13))
sns.set_style("darkgrid")
ax = batsmen_runs_per_season[:5].T.plot()
#Batsmen with best strike rate
batsmen.head()
batsman_strike_rate = batsmen.groupby(['season', 'batting_team', 'batsman'])['strike_rate'].mean().reset_index()
batsman_strike_rate = batsman_strike_rate.sort_values(by='strike_rate', ascending=False)
batsman_strike_rate = batsman_strike_rate[:10]
plt.figure(figsize=(15,13))
sns.set_style("darkgrid")
sns.barplot(data=batsman_strike_rate, x='batsman', y='strike_rate', hue='season')
#batsman_strike_rate.plot(kind='bar', stacked=True)
#sns.set_palette("bright", len(batsman_strike_rate['season']))
#Best Bowlers
bowler_wickets = deliveries.groupby(['match_id', 'inning', 'bowling_team', 'bowler', 'over'])
bowler_runs = bowler_wickets['total_runs', 'wide_runs', 'bye_runs','legbye_runs', 'noball_runs'].sum().reset_index()
bowler_runs["runs"] = bowler_runs["total_runs"] - (bowler_runs["bye_runs"] + bowler_runs['legbye_runs'])
bowler_runs["extras"] = bowler_runs["wide_runs"] + bowler_runs['noball_runs']
del(bowler_runs["bye_runs"])
del(bowler_runs["legbye_runs"])
del(bowler_runs["total_runs"])
dismissal_kind_bowlers = ["bowled", "caught", "lbw", "stumped","caught and bowled", "hit wicket"]
dismissals = deliveries[deliveries["dismissal_kind"].isin(dismissal_kind_bowlers)]
dismissals.head()
dismissals = dismissals.groupby(["match_id", "inning", "bowling_team","bowler", "over"])["dismissal_kind"].count().reset_index()
dismissals.rename(columns={"dismissal_kind": "wickets"}, inplace=True)
bowler_runs = bowler_runs.merge(dismissals, left_on=["match_id", "inning", "bowling_team", "bowler", "over"], 
                        right_on=["match_id", "inning", "bowling_team", "bowler", "over"], how="left")
bowler_runs["wickets"] = bowler_runs["wickets"].fillna(0)
bowler_overs = bowler_runs.groupby(["match_id", "inning", "bowling_team","bowler"])["over"].count().reset_index()
bowler_runs = bowler_runs.groupby(['match_id', 'inning', 'bowling_team','bowler']).sum().reset_index()
