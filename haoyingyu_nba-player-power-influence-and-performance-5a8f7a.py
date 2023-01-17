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
influence_performance_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")

influence_performance_df.head()


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = influence_performance_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=influence_performance_df)

import math
g = sns.lmplot(x="PAGEVIEWS", y="WINS_RPM", data=influence_performance_df)

g = (g.set(xlim = (0,7500), ylim = (0, 25)))
g = sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="POINTS", data=influence_performance_df)

g = (g.set(xlim = (200,4000), ylim = (0, 45)))
from ggplot import *



p = ggplot(influence_performance_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")
p = ggplot(influence_performance_df,aes(x="PAGEVIEWS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("WIKIPAGEVIEWS") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  WIKIPAGEVIEWS, WINS REAL PLUS MINUS and SALARY")
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = influence_performance_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.pairplot(influence_performance_df, vars = ["FGA","POINTS","WINS_RPM","SALARY_MILLIONS","PAGEVIEWS"], size = 2)