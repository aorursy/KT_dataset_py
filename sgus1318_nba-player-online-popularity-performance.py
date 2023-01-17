import numpy as np 

import pandas as pd 



import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.cluster import KMeans



import matplotlib.pyplot as plt



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")

attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv")

salary_df.head()
pie_df = pd.read_csv("../input/nba_2017_pie.csv")

pie_df.head()
list(pie_df)
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv")

plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv")

br_stats_df.head()
list(br_stats_df)
### Remove Position Abbreviation from Name Field 



plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = []

for player in plus_minus_df["PLAYER"]:

    plyr, _ = player.split(",")

    players.append(plyr)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)

plus_minus_df["PLAYER"] = players

plus_minus_df.head()
nba_players_df = br_stats_df.copy()

nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)

nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)

nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")

nba_players_df.head()
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()

nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")

nba_players_df.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)

salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)

salary_df.head()
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))

len(diff)
nba_players_with_salary_df = nba_players_df.merge(salary_df)
nba_players_with_salary_df.isnull().sum()
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap="Blues")
print(set(nba_players_with_salary_df["POSITION"]))
sal_SF = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="SF","SALARY_MILLIONS"]

sal_SG = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="SG","SALARY_MILLIONS"]

sal_C = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="C","SALARY_MILLIONS"]

sal_PFC = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PFC","SALARY_MILLIONS"]

sal_PF = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PF","SALARY_MILLIONS"]

sal_PG = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PG","SALARY_MILLIONS"]
sns.countplot(x='POSITION',data=nba_players_with_salary_df, palette="Set2")

plt.show()
plt.figure(figsize=(15,8))

sns.kdeplot(sal_SF, color="darkturquoise", shade=False)

sns.kdeplot(sal_SG, color="lightcoral", shade=False)

sns.kdeplot(sal_C, color="forestgreen", shade=False)

sns.kdeplot(sal_PFC, color="dimgray", shade=False)

sns.kdeplot(sal_PF, color="gold", shade=False)

sns.kdeplot(sal_PF, color="darkorchid", shade=False)

sns.kdeplot(sal_PG, color="maroon", shade=False)

plt.legend(['SF', 'SG', 'C', 'PF-C', 'PF', 'PG'])

plt.title('Density Plot of Salary by Position')

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(x="SALARY_MILLIONS", y="POSITION",data=nba_players_with_salary_df, orient="h",palette="Set2")

plt.show()
print(set(nba_players_with_salary_df["TEAM"]))
plt.figure(figsize=(8,15))

sns.boxplot(x="SALARY_MILLIONS", y="TEAM",data=nba_players_with_salary_df, orient="h")

plt.show()
sns.kdeplot(nba_players_with_salary_df["AGE"], color="mediumpurple", shade=True)

plt.show()
sns.lmplot(x="AGE", y="SALARY_MILLIONS", data=nba_players_with_salary_df)

plt.show()
results1 = smf.ols('SALARY_MILLIONS ~ AGE', data=nba_players_with_salary_df).fit()

print(results1.summary())
sns.kdeplot(nba_players_with_salary_df["WINS_RPM"], color="darkmagenta", shade=True)

plt.show()
sns.lmplot(x="WINS_RPM", y="SALARY_MILLIONS", data=nba_players_with_salary_df)

plt.show()
results2 = smf.ols('SALARY_MILLIONS ~ WINS_RPM', data=nba_players_with_salary_df).fit()

print(results2.summary())
sns.kdeplot(nba_players_with_salary_df["MPG"], color="dodgerblue", shade=True)

plt.show()
sns.lmplot(x="MPG", y="SALARY_MILLIONS", data=nba_players_with_salary_df)

plt.show()
results3 = smf.ols('SALARY_MILLIONS ~ MPG', data=nba_players_with_salary_df).fit()

print(results3.summary())
sns.kdeplot(nba_players_with_salary_df["POINTS"], color="darkturquoise", shade=True)

plt.show()
sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=nba_players_with_salary_df)

plt.show()
results4 = smf.ols('SALARY_MILLIONS ~ POINTS', data=nba_players_with_salary_df).fit()

print(results4.summary())
from pydoc import help

from scipy.stats.stats import pearsonr
pearsonr(nba_players_with_salary_df["ORB"], nba_players_with_salary_df["SALARY_MILLIONS"])

## Returns the Pearson's correlation coefficient and the 2-tailed p-value
pearsonr(nba_players_with_salary_df["ORB"], nba_players_with_salary_df["WINS_RPM"])
pearsonr(nba_players_with_salary_df["STL"], nba_players_with_salary_df["SALARY_MILLIONS"])
pearsonr(nba_players_with_salary_df["STL"], nba_players_with_salary_df["WINS_RPM"])
from ggplot import *
p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
median_wiki_df = wiki_df.groupby("PLAYER").median()

median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]

median_wiki_df_small = median_wiki_df_small.reset_index()

nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df['name_length']=nba_players_with_salary_wiki_twitter_df['PLAYER'].str.len()
nba_players_with_salary_wiki_twitter_df.head()
sns.lmplot(x="name_length", y="PAGEVIEWS", data=nba_players_with_salary_wiki_twitter_df)

plt.show()
results_name = smf.ols('PAGEVIEWS ~ name_length', data=nba_players_with_salary_wiki_twitter_df).fit()

print(results_name.summary())
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = nba_players_with_salary_wiki_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap="Blues")

plt.show()