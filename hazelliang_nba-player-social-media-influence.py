import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
## Plus_minus
plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
plus_minus_df.head()
## br_stats
nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head()
## Pie
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head()
## Salary 
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)

nba_players_with_salary_df = nba_players_df.merge(salary_df); 
## Wiki
wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
## Twitter
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
## Null value
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_twitter_df.dropna(how = 'any')
nba_players_with_salary_wiki_twitter_df.head()
nba_analysis = nba_players_with_salary_wiki_twitter_df.loc[:,["PLAYER", "RPM", "WINS_RPM", "PIE", "W", "SALARY_MILLIONS", "PAGEVIEWS", "TWITTER_FAVORITE_COUNT", "TWITTER_RETWEET_COUNT"]]
nba_analysis.head()
## Salary 
nba_analysis['SALARY_MILLIONS'].hist(bins=70)
## Pageviews on Wiki
nba_analysis["PAGEVIEWS"].hist(bins=50)
##TWITTER_FAVORITE_COUNT
nba_analysis["TWITTER_FAVORITE_COUNT"].hist(bins=50)
## TWITTER_RETWEET_COUNT
nba_analysis["TWITTER_RETWEET_COUNT"].hist(bins=50)
## Social Impact
fig,axes=plt.subplots(1,1)
sns.distplot(np.log(nba_analysis["PAGEVIEWS"]),hist=False, kde_kws={"label":"PAGEVIEWS"})
sns.distplot(np.log(nba_analysis["TWITTER_FAVORITE_COUNT"]),hist=False, kde_kws={"label":"TWITTER_FAVORITE_COUNT"})
sns.distplot(np.log(nba_analysis["TWITTER_RETWEET_COUNT"]),hist=False, kde_kws={"label":"TWITTER_RETWEET_COUNT"})
plt.xlabel('Social Impact')
nba_analysis['LOG_PAGEVIEWS'] = np.log(nba_analysis['PAGEVIEWS'])
nba_analysis['LOG_TWITTER_FAVORITE_COUNT'] = np.log(nba_analysis['TWITTER_FAVORITE_COUNT'])
nba_analysis['LOG_TWITTER_RETWEET_COUNT'] = np.log(nba_analysis['TWITTER_RETWEET_COUNT'])
nba_analysis = nba_analysis.drop(['PAGEVIEWS', 'TWITTER_FAVORITE_COUNT', 'TWITTER_RETWEET_COUNT'], axis = 1)
nba_analysis = nba_analysis.replace(-np.inf, np.nan)
nba_analysis = nba_analysis.dropna(how = "any")
nba_analysis.head()
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & Social Influence)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.subplots(figsize=(10,5))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  Salary and Social Impact")
corr = nba_analysis.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)
!pip install yellowbrick
nba_analysis_noplayer = nba_analysis.drop(["PLAYER"], axis = 1)
from yellowbrick.features import Rank2D

visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(nba_analysis_noplayer)
visualizer.poof()
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_wiki_twitter_df)
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="WINS_RPM", data=nba_players_with_salary_wiki_twitter_df)
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="WINS_RPM", data=nba_players_with_salary_wiki_twitter_df)
sns.lmplot(x="PIE", y="WINS_RPM", data=nba_players_with_salary_wiki_twitter_df)
results_1 = smf.ols('W ~SALARY_MILLIONS', data=nba_analysis).fit()
print(results_1.summary())
results_2 = smf.ols('WINS_RPM ~SALARY_MILLIONS', data=nba_analysis).fit()
print(results_2.summary())
results_3 = smf.ols('WINS_RPM ~ LOG_PAGEVIEWS + LOG_TWITTER_FAVORITE_COUNT + LOG_TWITTER_RETWEET_COUNT', data=nba_analysis).fit()
print(results_3.summary())
results_4 = smf.ols('W ~ LOG_PAGEVIEWS + LOG_TWITTER_FAVORITE_COUNT ', data=nba_analysis).fit()
print(results_4.summary())
results_5 = smf.ols('WINS_RPM ~ PIE ', data=nba_analysis).fit()
print(results_5.summary())