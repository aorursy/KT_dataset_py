# basic operations

import numpy as np

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-pastel')



# file path

import os

print(os.listdir("../input/premier-league-player-stats-data/"))
# reading the data and checking the run time



%time data = pd.read_csv('/kaggle/input/premier-league-player-stats-data/Premier League Player Stats.csv')



print(data.shape)
# checking the first 5 rows and columns

data.head()
# decsribing the data



data.describe()
# checking NULL value



data.isnull().sum()
data.loc[data['G'] > 0, 'SA%'] =  round(data['SOG']/data['SHOTS'],3)*100

data.loc[data['G'] > 0, 'GA%'] =  round(data['G']/data['SHOTS'],3)*100

data.loc[data['G'] > 0, 'MPG'] = round(data['MIN']/data['G'],1)
data.head()
# check for NaN values

data.isna().sum()
# fill NaN values with lowest accuracy (0) for SA% and GA% beca

data['SA%'] = data['SA%'].fillna(0)

data['GA%'] = data['GA%'].fillna(0)



# fill all NaN with max time for MPG

data['MPG'] = data['MPG'].fillna(data['MPG'].max())
# check whether nan value is filled or not

data.isna().sum()
# top 10 player with highest shot accuracy

data.sort_values(by=['SA%'], ascending=False).head(10)
# top 10 player with highest shot accuracy

data.sort_values(by=['GA%'], ascending=False).head(10)
# top 10 player with highest shot accuracy

data.sort_values(by=['MPG'], ascending=True).head(10)
def team(x):

    return data[data['TEAM'] == x]
arsenal = team('Arsenal')
# top 10 arsenal player with highest shot accuracy

arsenal.sort_values(by=['MPG'], ascending=True).head(10)
# top 10 arsenal player with highest goal accuracy

arsenal.sort_values(by=['GA%'], ascending=False).head(10)
x = data.GP

plt.figure(figsize = (12, 8))



ax = sns.distplot(x, bins = 50, kde = False, color = 'b')

ax.set_xlabel(xlabel = 'Game Played range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Histogram for the number of game played of the Players', fontsize = 20)

plt.show()

# To show Different overall scores of the players participating in the FIFA 2019



sns.set(style = "dark", palette = "deep", color_codes = True)

x = data.G

plt.figure(figsize = (12,8))

plt.style.use('ggplot')



ax = sns.distplot(x, bins = 50, kde = False, color = 'r')

ax.set_xlabel(xlabel = "Number of Goals", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Histogram of number of goals by players', fontsize = 20)

plt.show()
# To show Different nations participating in the premier league



plt.style.use('fivethirtyeight')

data['TEAM'].value_counts().plot.bar(color = 'orange', figsize = (20, 7))

plt.title('Different Nations Participating in Premier League', fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Country')

plt.ylabel('Count')

plt.show()
# plotting a correlation heatmap



sns.heatmap(data.corr(), annot = True)



plt.title('Correlation of the Dataset', fontsize = 30)

plt.show()
# picking up the countries with highest number of players to compare their overall scores



data['TEAM'].value_counts()
# Every Nations' Player and their goals



teams = ('West Ham United', 'Arsenal', 'Manchester United', 'Burnley', 'Norwich City', 'Liverpool', 'Watford', 'Tottenham Hotspur')

data_teams = data.loc[data['TEAM'].isin(teams) & data['G']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.violinplot(x = data_teams['TEAM'], y = data_teams['G'], palette = 'Reds')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Number of GOals', fontsize = 9)

ax.set_title(label = 'Distribution of Goals of players from different countries', fontsize = 20)

plt.show()
# Every Nations' Player and their overall scores



teams = ('West Ham United', 'Arsenal', 'Manchester United', 'Burnley', 'Norwich City', 'Liverpool', 'Watford', 'Tottenham Hotspur')

data_teams = data.loc[data['TEAM'].isin(teams) & data['SHOTS']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.barplot(x = data_teams['TEAM'], y = data_teams['SHOTS'], palette = 'spring')

ax.set_xlabel(xlabel = 'Team', fontsize = 9)

ax.set_ylabel(ylabel = 'Total Shots', fontsize = 9)

ax.set_title(label = 'Distribution of total shots of players from different countries', fontsize = 20)

plt.show()
# finding 20 top Players who have max number of goals



highest_goals = data.sort_values('G', ascending = False)[['PLAYER','TEAM','GP','GS','G']].head(20)

print(highest_goals)
# finding 20 top Players who have lowest number of goals



lowest_goals = data.sort_values('G', ascending = True)[['PLAYER','TEAM','GP','GS','G']].head(20)

print(lowest_goals)
sns.lineplot(data['GP'], data['G'])

plt.title('Game Played vs Number of goals', fontsize = 20)



plt.show()
sns.lineplot(data['GS'], data['G'])

plt.title('Game started vs Number of goals', fontsize = 20)



plt.show()
sns.lineplot(data['GP'], data['MIN'])

plt.title('Game Played vs Minutes played', fontsize = 20)



plt.show()
# total goals by each team

goals = data.groupby("TEAM")["G"].sum().reset_index().sort_values(by = "G",ascending = False)
plt.figure(figsize=(9,14))

ax = sns.barplot(x="G",y="TEAM",

                 data=goals,palette="rainbow",

                 linewidth = 1,edgecolor = "k"*30)

for i,j in enumerate(goals["G"][:20]):

    ax.text(.3,i,j,weight="bold",color = "k",fontsize =12)

plt.title("Teams with highest total goals ")

plt.show()