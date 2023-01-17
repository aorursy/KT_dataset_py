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
attendance_valuation_elo_df.isnull().sum().sort_values(ascending=False)
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

salary_df.isnull().sum().sort_values(ascending=False)
pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
pie_df.isnull().sum().sort_values(ascending=False)
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
plus_minus_df.isnull().sum().sort_values(ascending=False)
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()
br_stats_df.isnull().sum().sort_values(ascending=False)
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

nba_players_with_salary_df = nba_players_df.merge(salary_df); 

nba_players_with_salary_df.head()
nba_players_with_salary_df.info()


plt.subplots(figsize=(20,20))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, square = True, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot = True)
df1=nba_players_with_salary_df.loc[:,['WINS_RPM','MPG','RPM','SALARY_MILLIONS','FTA','POINTS']]

sns.pairplot(df1)
results = smf.ols(formula='WINS_RPM ~ POINTS + RPM + ORPM + FTA', data=nba_players_with_salary_df).fit()

print(results.summary())
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(wiki_df)

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df.head()


plt.subplots(figsize=(30,30))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = nba_players_with_salary_wiki_twitter_df.corr()

sns.heatmap(corr, square = True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot = True)
results = smf.ols(formula='WINS_RPM ~ TWITTER_FAVORITE_COUNT + PAGEVIEWS', data=nba_players_with_salary_wiki_twitter_df).fit()

print(results.summary())