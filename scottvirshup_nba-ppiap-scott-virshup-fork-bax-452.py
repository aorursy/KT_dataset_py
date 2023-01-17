import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


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

# merge the two dataframes

nba_players_with_salary_df = nba_players_df.merge(salary_df); 


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df)

results = smf.ols('W ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('W ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

from ggplot import *



p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)

median_wiki_df = wiki_df.groupby("PLAYER").median()



median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small.head()
median_wiki_df_small = median_wiki_df_small.reset_index()

median_wiki_df_small.head()
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)

nba_players_with_salary_wiki_twitter_df.head()


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = nba_players_with_salary_wiki_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Creates variable called "positive marks" which basically just combines favorites and retweets.



nba_players_with_salary_wiki_twitter_df['TWITTER_POSITIVE_MARK'] = round(nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'] + 2*(nba_players_with_salary_wiki_twitter_df['TWITTER_RETWEET_COUNT']))
# Scatter plot of salary and twitter positive marks

sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_POSITIVE_MARK", data=nba_players_with_salary_wiki_twitter_df)
nba_players_with_salary_wiki_twitter_df.head()
# Number of clusters

k_means = KMeans(n_clusters=3)



# Choose the columns that the clusters will be based upon

cluster_source = nba_players_with_salary_wiki_twitter_df.loc[:,["SALARY_MILLIONS", "W", "PAGEVIEWS"]]



# Create the clusters

kmeans = k_means.fit(cluster_source)



# Create a column, 'cluster,' denoting the cluster classification of each row

nba_players_with_salary_wiki_twitter_df['cluster'] = kmeans.labels_



# Create a scatter plot with colors based on the cluster

ax = sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", data=nba_players_with_salary_wiki_twitter_df,hue="cluster", size=12, fit_reg=False)

ax.set(xlabel='Wikipedia Pageviews', ylabel='Salary in millions', title="NBA player Wikipedia pageviews vs Salary in millions clustered on SALARY_MILLIONS, W, PAGEVIEWS:  2016-2017 Season")