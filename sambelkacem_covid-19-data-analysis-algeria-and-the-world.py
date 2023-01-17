import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import zscore
df = pd.read_csv(r"../input/covid19-algeria-and-world-dataset/Data.csv")
df = df.round(3)

df
df.isnull().sum()
df = df.groupby('Entity').apply(lambda x: x.fillna(method='ffill'))

df = df.groupby('Entity').apply(lambda x: x.fillna(method='bfill'))

df = df[df.Cases > 0]
df.describe().round(2)
for column in df.drop(['Entity', 'Continent', 'Date', 'Daily tests', 'Cases', 'Deaths'], axis=1):

    plt.figure()

    df.boxplot([column])
columns = ['Date', 'Daily tests', 'Cases', 'Deaths']

df.drop(columns, axis=1).drop_duplicates().hist(bins=15, figsize=(16, 9), rwidth=0.8)

plt.show()
# Keep the last line (date) for each country and drop unused columns

df_last = df.groupby('Entity').tail(1).drop(['Entity', 'Date'], axis=1)

plt.figure(figsize=(12, 8))

sns.heatmap(df_last.corr(), annot=True, cmap=plt.cm.Reds)

plt.show()
# Target countries and target dates

countries = ['Algeria', 'Bahrain', 'Ethiopia', 'Ghana', 'Kenya', 'Morocco', 'Nigeria', 'Senegal', 'Tunisia']

df_temp = df.loc[df['Date'] > '2020-02-25']



# Plot case and death curves

for output_variable in ['Cases', 'Deaths']:

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, grp in df_temp[df_temp['Entity'].isin(countries)].groupby(['Entity']):

        ax = grp.plot(ax=ax, kind='line', x='Date', y=output_variable, label=key)

    plt.legend(loc='best')

    plt.xticks(rotation=90)

    plt.ylabel(output_variable)

    plt.show()
# Scatter plots readability: remove outliers in all comuns except in the column 'Continent'

column_continent = df_last[['Continent']]

df_last = df_last.drop('Continent', axis=1)

df_last = column_continent.join(df_last[(np.abs(zscore(df_last)) < 3).all(axis=1)])



# Plot feature-output-variable distributions for each column

for column in df_last.columns:

    fig, ax = plt.subplots(ncols=2, figsize=(14, 4))

    df_last.plot.scatter(x=column, y='Cases', ax=ax[0])

    df_last.plot.scatter(x=column, y='Deaths', ax=ax[1])

    if column == 'Continent':

        fig.autofmt_xdate(rotation=90)