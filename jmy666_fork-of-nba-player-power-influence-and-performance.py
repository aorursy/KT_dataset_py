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

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from ggplot import *
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



nba_players_with_salary_df = nba_players_df.merge(salary_df); nba_players_with_salary_df.head()

nba_players_with_salary_df.columns



plt.subplots(figsize=(30,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns,

            yticklabels=corr.columns,annot =True)


p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS",shape = 'POSITION')) + geom_point()

p + facet_wrap('POSITION')+xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")

import numpy as np

from sklearn.model_selection import train_test_split

train,test = train_test_split(nba_players_with_salary_df,test_size=0.2, random_state=0)

train.head()

test.head()