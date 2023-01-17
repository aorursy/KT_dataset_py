import pandas as pd

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

sns.set_style("whitegrid")

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
## This data set will be our starting point

player_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv");player_df.head()
plt.figure(figsize=(15,7))

sns.regplot(x="MP", y="WINS_RPM", data=player_df)
minutes = smf.ols('WINS_RPM ~ MP', data=player_df).fit()

print(minutes.summary())
ax = sns.lmplot(x="MP", y="WINS_RPM", data=player_df, hue='POSITION', fit_reg=False, size=6, aspect=2, legend=False, scatter_kws={"s": 200})

ax.set(xlabel='Minutes Played', ylabel='RPM (Real Plus Minus)', title="Minutes Played vs RPM (Real Plus Minus) by Position: 2016-2017 Season")

plt.legend(loc='upper left', title='Position')
## Defense

player_df_def = player_df[["DRB","STL","BLK","WINS_RPM"]].copy();player_df_def.head()
## Offense

player_df_off = player_df[["eFG%","FT%","ORB","AST","POINTS","WINS_RPM"]].copy()

player_df_off.rename(columns={'eFG%': 'eFG','FT%':'FT'}, inplace=True)

player_df_off.head()
player_df_def.corr()
plt.subplots(figsize=(10,10))

sns.heatmap(player_df_def.corr(), xticklabels=player_df_def.columns.values, yticklabels=player_df_def.columns.values, cmap="Reds")
player_df_def.cov()
defense = smf.ols('WINS_RPM ~ DRB + STL', data=player_df_def).fit()

print(defense.summary())
fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_partregress_grid(defense, fig=fig)
fig = plt.figure(figsize=(12, 8))

fig = sm.graphics.plot_ccpr_grid(defense, fig=fig)
fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_regress_exog(defense, "STL", fig=fig)
sns.jointplot("STL", "WINS_RPM", data=player_df_def,size=10, ratio=3, color="r")
steals = smf.ols('WINS_RPM ~ STL', data=player_df_def).fit()

print(steals.summary())
player_df_off.corr()
plt.subplots(figsize=(10,10))

sns.heatmap(player_df_off.corr(), xticklabels=player_df_off.columns.values, yticklabels=player_df_off.columns.values, cmap="Greens")
player_df_off.cov()
offense = smf.ols('WINS_RPM ~ eFG + ORB + AST + POINTS', data=player_df_off).fit()

print(offense.summary())
fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_partregress_grid(offense, fig=fig)
fig = plt.figure(figsize=(12, 8))

fig = sm.graphics.plot_ccpr_grid(offense, fig=fig)
fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_regress_exog(offense, "POINTS", fig=fig)
sns.jointplot("POINTS", "WINS_RPM", data=player_df_off,size=10, ratio=3, color="g")
eFGs = smf.ols('WINS_RPM ~ POINTS', data=player_df_off).fit()

print(eFGs.summary())
## Final Variables

player_df_full = player_df[["PLAYER","STL","DRB","eFG%","ORB","AST","POINTS","WINS_RPM"]].copy()

player_df_full.rename(columns={'eFG%': 'eFG'}, inplace=True)

player_df_full.head()
combined = smf.ols('WINS_RPM ~ STL + DRB + eFG + AST + POINTS', data=player_df_full).fit()

print(combined.summary())