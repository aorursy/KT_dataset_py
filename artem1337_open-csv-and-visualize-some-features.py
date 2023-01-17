import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/japanese-j1-league/results.csv')

df
# Check NULL values

df.isna().apply(pd.value_counts) 
df.Home.value_counts()
df.Away.value_counts()
df.HG.value_counts()
df.AG.value_counts()
df.Res.value_counts()
sns.barplot(x=df.Res.value_counts().keys(),

            y=df.Res.value_counts().values)
for year in np.unique(df.Season.values):

    plt.title(year)

    sns.barplots(x=df[df.Season == year].Res.value_counts().keys(),

                y=df[df.Season == year].Res.value_counts().values)

    plt.show()
for team in df.Home.value_counts().keys():

    plt.title(team)

    sns.barplot(x=df[df.Home == team].Res.value_counts().keys(),

                y=df[df.Home == team].Res.value_counts().values)

    plt.show()
for team in df.Away.value_counts().keys():

    plt.title(team)

    sns.barplot(x=df[df.Away == team].Res.value_counts().keys(),

                y=df[df.Away == team].Res.value_counts().values)

    plt.show()