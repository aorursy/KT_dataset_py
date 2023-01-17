# load python packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_regression
# load the dataset

players_stats = pd.read_csv('../input/nba_2017_players_stats_combined.csv')

players_stats.head()
# observe the data types

players_stats.dtypes
# remove column 'Unnamed: 0'

players_stats.drop('Unnamed: 0', axis=1, inplace=True)
# observe the correlation between each variables 

plt.subplots(figsize=(20,15))

axes = plt.axes()

axes.set_title("NBA Player Correlation Heatmap")

corr = players_stats.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# prepare dataset

players_stats.dropna(inplace=True)

data = players_stats.loc[:,'AGE':'W'].drop('TEAM',axis=1)
# select features using univariate measures

Select_f = SelectPercentile(f_regression, percentile=25)

Select_f.fit(data.drop('WINS_RPM',axis=1),data['WINS_RPM'])

feature = []

fscore = []

for n,s in zip(data.drop('WINS_RPM',axis=1).columns,Select_f.scores_):

    fscore.append(s)

    feature.append(n)
# combine Feature and F-score and show the results

feature_fscore = pd.DataFrame([feature,fscore]).T

feature_fscore.columns = ['Feature','F-score']

feature_fscore = feature_fscore.sort_values(by=['F-score'], ascending=False)

print(feature_fscore)
# pick variables that have F-score greater than 300 for the regression

results = smf.ols(formula='WINS_RPM ~ RPM+ORPM+POINTS+FTA+FG+FT+MPG+MP+DRB+FGA+STL+PIE', data=data).fit()

print(results.summary())