import pandas as pd

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
salary_df = pd.read_csv("../input/nba_2017_salary.csv");

salary_df.sort_values('SALARY',ascending=False)[:10]
#round salary to millions for easier to read, plot bar chart

salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)

sns.barplot(x='NAME', y='SALARY_MILLIONS', data=salary_df)
#distribution of salary

salary_df['SALARY_MILLIONS'].plot(kind='hist')
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");

twitter_df.head()
#Merge twitter table to play&salary table

salary_df.rename(columns={'NAME': 'PLAYER',}, inplace=True)

salary_twit_df=salary_df.merge(twitter_df);

salary_twit_df.head()

#Combinte twitter favorite and retweet

salary_twit_df['TOTAL_TWIT']=salary_twit_df['TWITTER_FAVORITE_COUNT']+salary_twit_df['TWITTER_RETWEET_COUNT'];

salary_twit_df.head()
#Check relationship between total twit and salary

sns.jointplot(x="SALARY_MILLIONS", y="TOTAL_TWIT", data=salary_twit_df)
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
#split and rename colomns

plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = []

for player in plus_minus_df["PLAYER"]:

    plyr, _ = player.split(",")

    players.append(plyr)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)

plus_minus_df["PLAYER"] = players

plus_minus_df.head()
#plot pairplot

sns.pairplot(plus_minus_df)
#merge win info with player&salary&twit

nba_players_df=salary_twit_df.merge(plus_minus_df,on="PLAYER",how="inner");

nba_players_df.head()
nba_players_df.drop(["TEAM_y"], inplace=True, axis=1);

nba_players_df.rename(columns={"TEAM_x":"TEAM"}, inplace=True)

nba_players_df.head()
#check relationship between salary and wins

p1=sns.jointplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_df)
#merge player info with salary

nba_players_with_salary_df = nba_players_df.merge(salary_df); 
#plot correlation heatmap 

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_df.corr()

sns.heatmap(corr, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#check regression for wins_rpm and salary

results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_df).fit();

print(results.summary())
#check regression for wins_rpm and salary

results = smf.ols('TOTAL_TWIT ~WINS_RPM', data=nba_players_df).fit();

print(results.summary())