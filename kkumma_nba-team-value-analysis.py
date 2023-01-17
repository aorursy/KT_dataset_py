import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
nba_salary = pd.read_csv('../input/nba_2017_salary.csv')
nba_team_value = pd.read_csv('../input/nba_2017_team_valuations.csv')
nba_twitter = pd.read_csv('../input/nba_2017_twitter_players.csv')
nba_salary = nba_salary.rename(columns={'NAME' : 'PLAYER'})
nba_salary.head(2)
nba_team_value.sort_values(by = 'VALUE_MILLIONS', ascending = False, inplace = True)
nba_team_value.head(10)
nba_twitter.head(2)
team_salary = nba_salary.groupby(['TEAM']).mean().reset_index()
team_salary.sort_values(by = 'SALARY',ascending = False,inplace = True)
team_salary.head(10)
df = pd.merge(nba_salary, nba_twitter, on = 'PLAYER')
#avg salary based on position 
df.POSITION.replace(' SF','SF',inplace = True)
df.POSITION.replace(' PG','PG',inplace = True)
salary_position = df['SALARY'].groupby(df['POSITION']).mean().reset_index()
salary_position
team_twitter = df.groupby(df['TEAM']).agg({'SALARY':'mean','TWITTER_FAVORITE_COUNT':'mean','TWITTER_RETWEET_COUNT':'mean'}).reset_index()
full_data = pd.merge(nba_team_value, team_twitter, on = 'TEAM')
full_data.head(5)
plt.figure(figsize = (15,4))
sns.barplot(x=salary_position['POSITION'], y=salary_position['SALARY'], data=salary_position).set_title("Position Vs Salary")


plt.show()
plt.figure(figsize = (20,4))
sns.barplot(x=team_salary.head(10)['TEAM'], y=team_salary.head(10)['SALARY'], data=team_salary).set_title("Top 10 Team with highest average salary amount the league")


plt.show()
plt.figure(figsize = (20,4))
sns.barplot(x=nba_team_value.head(10)['TEAM'], y=nba_team_value.head(10)['VALUE_MILLIONS'], data=nba_team_value).set_title("Top 10 Teams with highest value amount the league ")


plt.show()
plt.subplots(figsize=(7,6))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (SALARY & TWITTER)")
corr = full_data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)