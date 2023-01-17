# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_all = pd.read_csv("../input/QBStats_all.csv")
# Remove known outliers for ypa

df_all = df_all[df_all.ypa < 15]



# Reduce the analysis to only QBs that have attempted at least 250 passes

scoring_qbs = df_all.groupby('qb').sum()



scoring_qbs = scoring_qbs[scoring_qbs.cmp > 250]

print(scoring_qbs.shape)



for index, row in df_all.iterrows():

    if "t" in str(row.lg):

        lg = str(row.lg[:-1])

        df_all.set_value(index,'lg',lg)

    if row.qb not in scoring_qbs.index:

        df_all = df_all.drop(index)



df_all.lg = df_all.lg.astype(float)

df_all = df_all.dropna()
grouped_by_year = df_all.groupby(['year']).mean()



grouped_by_year = grouped_by_year.reset_index()



grouped_by_year.drop('yds',axis=1).plot(x='year')
sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.rate, kind='reg')

plt.xlim(1996,2016)

sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.yds, kind='reg')

plt.xlim(1996,2016)

sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.td, kind='reg')

plt.xlim(1996,2016)

sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.ypa, kind='reg')

plt.xlim(1996,2016)

sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.lg, kind='reg')

plt.xlim(1996,2016)
sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.sack, kind='reg')

plt.xlim(1996,2016)

sns.jointplot(x=grouped_by_year.year,y=grouped_by_year.loss, kind='reg')

plt.xlim(1996,2016)
grouped_by_home_game = df_all.groupby('home_away').mean().drop(['year','yds'],axis=1).copy(deep=True)

grouped_by_home_game['rate'].plot()

plt.ylabel("Rating")

plt.xlabel("Away - Home")

print("Difference between home stats and away stats:")

grouped_by_home_game.ix['home'] - grouped_by_home_game.ix['away']