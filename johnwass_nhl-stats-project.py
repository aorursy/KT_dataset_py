# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#This data is all NHL data from the last 6 years. Important disclosure, the Vegas Golden Knights were only a team for 2018.
#VGK had a very successful 2018 therefore they will be included in the data
team_stats = pd.read_csv("../input/game_teams_stats.csv")
game_skater_stats = pd.read_csv("../input/game_skater_stats.csv")
NHL_Teams = pd.read_csv("../input/team_info.csv")
NHL_Players = pd.read_csv("../input/player_info.csv")
players = NHL_Players.merge(game_skater_stats, on='player_id')
teams = NHL_Teams.merge(team_stats, on='team_id')
teams
#1. Is there correlation between wins and any of the numerical (i.e goals, faceoff wins, shots etc) stats?
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
Teams_Numerical = teams.groupby('teamName').sum()
teams_numerical = Teams_Numerical.drop(['team_id','franchiseId', 'game_id'],axis = 1)
y = teams_numerical['won']
X = teams_numerical.drop('won',axis=1)
lm.fit(X,y)
LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm.coef_
#the above coefficients show decent correlation between 4 of the variables, negative correlation with 4 of them and strong correlation with one variable: goals)
#2. What does the correlation between goals and wins look like in a scatter plot?
import matplotlib.pyplot as plt
plt.scatter(teams_numerical.won,teams_numerical.goals)
plt.xlabel('Wins')
plt.ylabel('Goals')
teams.info()
numerical_features = teams.select_dtypes(include=[np.number]).columns
categorical_features = teams.select_dtypes(include=[np.object]).columns
numerical_features = numerical_features.drop('franchiseId')
numerical_features = numerical_features.drop('game_id')
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer, LabelEncoder
scaler = StandardScaler()
lb = LabelBinarizer()
#3.Is there a correlation between any of the numerical variables themselves? i.e. we know goals lead to wins...
#But do other variables eventually lead to goals which in turn lead to wins? Use a heatmap to show this.
plt.figure(figsize=(20,7))
sns.heatmap(teams[numerical_features].corr(), annot=True)
#There seems to be a correlation between shots and faceoff percentage and shots and goals (although these data are definitely related since 1 goal = 1 SOG)
#4. What does shots versus Faceoffwin percentage look like in a scatter plot?
FOWP = teams.groupby('teamName').mean()['faceOffWinPercentage']
plt.scatter(teams_numerical.shots,FOWP)
plt.xlabel('Shots')
plt.ylabel('Face Off Win %')
#5. There seems to be a correlation between shots and faceoff win percentage so which teams lead in FaceOff Percentage?
teams.groupby(['teamName'])["shots","faceOffWinPercentage"].apply(lambda x : x.astype(int).mean()).sort_values('faceOffWinPercentage',ascending=False)
#6. Which teams who have won the most games with the number of shots they took? 
teams_numerical.groupby(['teamName','shots']).sum().sort_values('won', ascending = False)['won']
#7. In hockey, shots lead to goals, so who has taken the most shots and scored the most goals per game on average?
teams.groupby(['teamName'])["shots","goals"].apply(lambda x : x.astype(int).mean()).sort_values('goals',ascending=False)
#8. Okay, so now it has been shown FaceOff Win % leads to shots, shots lead to goals and goals lead to wins. 
#Lets look closer at the data and determine if this data over 4 years will help show the best teams. 
#For reference the Stanley Cup Champiions each year are as follows:
#2015 - Blackhawks
#2016 and 2017 - Penguins
#2018 - Capitals
Best = teams.groupby('teamName').mean()
Best
#9. Is there a correlation between winning face offs and winning games?
plt.scatter(Best.won,Best.faceOffWinPercentage)
plt.xlabel('Wins')
plt.ylabel('Face Off Win %')
#There does not seem to be correlation between FOWP and Wins, something that was shown earlier. But by building though the FOWP > Shots > Goals > Wins it should be able to be shown 
#10. What are the top 20 teams by FaceOffWinPercentage:
BestFOWP = teams.groupby('teamName').mean().sort_values('faceOffWinPercentage',ascending = False)[:20]
BestFOWP
#11. What teams who shoot the most, using average shots per game, and select the top 15 teams:
BestShots = BestFOWP.groupby('teamName').mean().sort_values('shots',ascending = False)[:15]
BestShots
#12. As can be seen, the top last 4 stanley cup champions are still there, now find top 5 teams based on wins:
BestWins = BestShots.groupby('teamName').mean().sort_values('won',ascending = False)['won'][:5]
BestWins
#Now we know the top 5 teams from the last 4 years. Now we will look at players from those teams to determine the best players on the best teams
games = NHL_Teams.merge(game_skater_stats, on='team_id')
games
#13. Using the games dataset create a new dataset called best_teams, selecting only the game data for the 5 best teams - Penguins, Capitals, Ducks, Blackhawks, Bruins
best_teams = games[(games['teamName']=='Penguins')| (games['teamName']=='Capitals')| (games['teamName']=='Bruins')| (games['teamName']=='Blackhawks') | (games['teamName']=='Ducks')]
#14. Filter out any player who has not played at least 200 games:
best_teams = best_teams.groupby('player_id').filter(lambda x: len(x) >= 200)
#15. Find the average per game stats per player using the best_teams dataset - 
stats = best_teams.groupby('player_id').mean()
#16. Create a new column called player score - similar to fantasy hockey points per game, with the following guidelines:
#goals * 60
#assists * 40
#plusMinus * 20
#shots * 9
#hits * 5
#blocked * 10
stats['player_score']= stats['goals']*60 + stats['assists']*40 + stats['plusMinus']*20 + stats['shots']*9 + stats['hits']*5 + stats['blocked']*10
stats
#17. Arrange the score from highest to lowest and determine the top 10 best players and their socres by player id:
best_players = stats.groupby('player_id').sum().sort_values('player_score',ascending = False)['player_score'][:10]
best_players
#18.What are the top 5 players names and how many games have they played?
#18aTo determine players:
bestplayersever = players[(players['player_id']==8471214)| (players['player_id']==8471675)| (players['player_id']==8470612)| (players['player_id']==8471215) | (players['player_id']==8471724)]
#18b. How many games have the best players played in?
bestplayersever.groupby(['firstName','lastName']).size()
#20. What are the average goals per game of the 5 best players?
bestplayersever.groupby(['firstName','lastName']).mean().sort_values('goals',ascending = False)['goals']
#21. What teams are the 5 best players on?
bestplayersever.groupby(['firstName','lastName']).mean()['team_id']
#So based off the team id the teams are:
print("team id of 5 is the:", teams.loc[teams['team_id'] == 5, 'teamName'].iloc[0])
print("team id of 15 is the:", teams.loc[teams['team_id'] == 15, 'teamName'].iloc[0])
print("team id of 24 is the:", teams.loc[teams['team_id'] == 24, 'teamName'].iloc[0])
#This means the Evgeni Malkin, Kris Letang and Sidney Crosby were on the Penguins
#The Penguins have won 2 of the last 4 Stanley Cups
#This makes total sense if they have 3 of the top 5 players from the last 4 years