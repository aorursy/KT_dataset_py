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
salary_df = pd.read_csv('../input/nba_2017_salary.csv');

salary_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");

br_stats_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");

plus_minus_df.head()
pie_df = pd.read_csv("../input/nba_2017_pie.csv");

pie_df.head()
plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = []

for player in plus_minus_df["PLAYER"]:

    name, pos = player.split(",")

    players.append(name)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)

plus_minus_df["PLAYER"] = players

plus_minus_df.head()
nba_players_df = br_stats_df.copy();

nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "PPG"}, inplace=True)

nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)

nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")

nba_players_df.head()
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()

nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")

nba_players_df.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

salary_df['SALARY_MILLION'] = round(salary_df['SALARY'] / 1000000, 2)

salary_df.drop(['POSITION','TEAM','SALARY'], inplace=True, axis=1)

salary_df.head()
nba_players_with_salary_df = nba_players_df.merge(salary_df, how='inner', on='PLAYER')

nba_players_with_salary_df['FIVE_STANDARD_STATS'] = nba_players_with_salary_df['PPG'] + nba_players_with_salary_df['TRB'] + nba_players_with_salary_df['AST'] + nba_players_with_salary_df['STL'] + nba_players_with_salary_df['BLK']

nba_players_with_salary_df.head()
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Stats Correlation Heatmap:  2016-2017 Season")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            cmap="YlGnBu")
sns.lmplot(height=8, aspect=1.5, x="SALARY_MILLION", y="WINS_RPM", data=nba_players_with_salary_df)
sns.lmplot(height=8, aspect=1.5, x="SALARY_MILLION", y="PPG", data=nba_players_with_salary_df)
sns.lmplot(height=8, aspect=1.5, x="SALARY_MILLION", y="FIVE_STANDARD_STATS", data=nba_players_with_salary_df)
results_ppg = smf.ols('SALARY_MILLION ~PPG', data=nba_players_with_salary_df).fit()

print(results_ppg.summary())

results_five_stats = smf.ols('SALARY_MILLION ~FIVE_STANDARD_STATS', data=nba_players_with_salary_df).fit()

print(results_five_stats.summary())

result_ws = smf.ols('SALARY_MILLION ~WINS_RPM', data=nba_players_with_salary_df).fit();

print(result_ws.summary());