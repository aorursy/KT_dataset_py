import pandas as pd

import numpy as np

from numpy import arange

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
play_tw_sala = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv");play_tw_sala.head()
play_tw_sala.info()
play_tw_sala.info()
play_tw_sala["SALARY_MILLIONS"].describe()
fig, ax = plt.subplots()

ax.hist(play_tw_sala['SALARY_MILLIONS'])

plt.show()
fig, ax = plt.subplots()

ax.hist(np.log(play_tw_sala['SALARY_MILLIONS']+1))

plt.show()
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = play_tw_sala.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.lmplot(x="WINS_RPM", y="SALARY_MILLIONS", data=play_tw_sala)
sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=play_tw_sala)
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="SALARY_MILLIONS", data=play_tw_sala)
sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", data=play_tw_sala)
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", data=play_tw_sala)
play_tw_sala['ViewBand'] = pd.cut(play_tw_sala['PAGEVIEWS'], 3)

df = play_tw_sala[["ViewBand", "SALARY_MILLIONS"]].groupby(['ViewBand'], as_index=False).mean()

heights = df["SALARY_MILLIONS"]

position =arange(3) + 1

tick_positions = range(1,4)

fig, ax = plt.subplots()

ax.plot(position, heights, 0.5)

ax.set_xticks(tick_positions)

plt.xticks(rotation=60)

ax.set_xticklabels(df["ViewBand"].values)

plt.xlabel('ViewBand')

plt.ylabel('SALARY_MILLIONS')

plt.show()
play_tw_sala['likeBand'] = pd.cut(play_tw_sala['TWITTER_FAVORITE_COUNT'], 5)

df =play_tw_sala[['likeBand', 'SALARY_MILLIONS']].groupby(['likeBand'], as_index=False).mean()

heights = df["SALARY_MILLIONS"]

position =arange(5) + 3

tick_positions = range(3,8)

fig, ax = plt.subplots()

ax.bar(position, heights, 0.3)

ax.set_xticks(tick_positions)

plt.xticks(rotation=60)

ax.set_xticklabels(df["likeBand"].values)

plt.xlabel('likeBand')

plt.ylabel('SALARY_MILLIONS')

plt.show()
play_tw_sala['retwBand'] = pd.cut(play_tw_sala['TWITTER_RETWEET_COUNT'], 5)

df =play_tw_sala[['retwBand', 'SALARY_MILLIONS']].groupby(['retwBand'], as_index=False).mean()

heights = df["SALARY_MILLIONS"]

position =arange(5) + 3

tick_positions = range(3,8)

fig, ax = plt.subplots()

ax.bar(position, heights, 0.3)

ax.set_xticks(tick_positions)

plt.xticks(rotation=60)

ax.set_xticklabels(df["retwBand"].values)

plt.xlabel('retwBand')

plt.ylabel('SALARY_MILLIONS')

plt.show()
play_tw_sala['POSITION'].unique()
grid = sns.FacetGrid(play_tw_sala, row='POSITION', size=8, aspect=1.6)

grid.map(sns.pointplot, 'TWITTER_RETWEET_COUNT', 'SALARY_MILLIONS', palette='deep')

grid.add_legend()
subset= play_tw_sala[["PLAYER","WINS_RPM","POINTS", "POSITION", "SALARY_MILLIONS","PAGEVIEWS", "TWITTER_FAVORITE_COUNT","TWITTER_RETWEET_COUNT"]];subset.head()
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(subset, hue="POSITION")
results = smf.ols('SALARY_MILLIONS ~POINTS+WINS_RPM+PAGEVIEWS', data=play_tw_sala).fit()

print(results.summary())
