import numpy as np # linear algebra

import pandas as pd



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')



df = pd.read_csv('../input/nfl_draft.csv')

df.head(3)
# Remove players who have not retired

rows = df.shape[0]



df = df[df.To < 2016]



print ('{percent:.2%}'.format(percent=float(df.shape[0]) / rows), 'players have retired')
df['YearsPlayed'] = df['To'] - df['Year'] + 1



years_played_counts = pd.DataFrame({'count': df.groupby('YearsPlayed').size()}).reset_index()

years_played_counts['cumul_count'] = years_played_counts['count'].cumsum()

years_played_counts['cumul_pct'] = years_played_counts.cumul_count / years_played_counts['count'].sum()



ax = years_played_counts[['YearsPlayed', 'count']].plot(x='YearsPlayed', kind='bar')

years_played_counts[['YearsPlayed', 'cumul_pct']].plot(x='YearsPlayed', linestyle='-', marker='o', ax=ax, secondary_y=True)
years_played_by_draft_year = pd.DataFrame({'count': df.groupby(['Year', 'YearsPlayed']).size()}).reset_index()

df[['Year', 'YearsPlayed']].boxplot(by='Year')
df[['Rnd', 'YearsPlayed']].boxplot(by='Rnd')
df_rnd1 = df[df.Rnd == 1]

df_rnd1[['Pick', 'YearsPlayed']].boxplot(by='Pick')
df_rnd2 = df[df.Rnd == 2]

df_rnd2[['Pick', 'YearsPlayed']].boxplot(by='Pick')