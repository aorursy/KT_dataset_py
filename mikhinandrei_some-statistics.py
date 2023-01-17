import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np
df = pd.read_csv('../input/results.csv')

df.info()
df.head()
def winner(row):

    if row['home_score'] > row['away_score']: return row['home_team'] 

    elif row['home_score'] < row['away_score']: return row['away_team']

    else: return 'DRAW'

    

df['winner'] = df.apply(lambda row: winner(row), axis=1)

df.head()
def loser(row):

    if row['home_score'] < row['away_score']: return row['home_team'] 

    elif row['home_score'] > row['away_score']: return row['away_team']

    else: return 'DRAW'

    

df['loser'] = df.apply(lambda row: loser(row), axis=1)

df.head()
winners = pd.value_counts(df.winner)

winners = winners.drop('DRAW')

winners.head(20)
fig, ax = plt.subplots(figsize=(25, 150))

sns.set(font_scale=1)

sns.barplot(y = winners.index.tolist(), x = winners.tolist())
goals = pd.Series(index=winners.index, dtype='int32')

for col in goals.index:

    goals[col] = df[df.home_team == col].home_score.sum() + df[df.away_team == col].away_score.sum()

goals = goals.fillna(0).sort_values(ascending=False)

goals.head(20)
fig, ax = plt.subplots(figsize=(25, 150))

sns.set(font_scale=1)

sns.barplot(y = goals.index.tolist(), x = goals.tolist())
stats = pd.DataFrame(columns=winners.index, index=['wins', 'draws', 'loses'], dtype='float64')

for col in goals.index:

    stats[col]['wins'] = len(df[df.winner == col])

    stats[col]['draws'] = len(df[df.home_team == col]) + len(df[df.away_team == col]) - (len(df[df.winner == col]) + len(df[df.loser == col]))

    stats[col]['loses'] = len(df[df.loser == col])

stats.head()
for c in stats.columns.tolist():

    stats[c] /= stats[c].sum()



stats = stats.sort_values(by='wins', axis=1, ascending=False)

stats.head()
df[df.winner == 'Cantabria'].head()
fig, ax = plt.subplots(1,1, figsize=(20, 8))

stats_20 = stats.loc[:, stats.columns[0]:stats.columns[20]]

y_offset = np.zeros(len(stats_20.columns))

index = np.arange(len(stats_20.columns))

colors = ['green', 'yellow', 'red']

plt.xticks(index, stats_20.columns, rotation=30)

for i, row in enumerate(stats_20.index.tolist()):

    plt.bar(index, stats_20.loc[row], bottom=y_offset, color=colors[i], width=0.5)

    y_offset = y_offset + stats_20.loc[row]
stats = stats.sort_values(by='loses', axis=1, ascending=True)

stats.head()
fig, ax = plt.subplots(1,1, figsize=(20, 8))

stats_20 = stats.loc[:, stats.columns[0]:stats.columns[20]]

y_offset = np.zeros(len(stats_20.columns))

index = np.arange(len(stats_20.columns))

colors = ['green', 'yellow', 'red']

plt.xticks(index, stats_20.columns, rotation=30)

for i, row in enumerate(stats_20.index.tolist()):

    plt.bar(index, stats_20.loc[row], bottom=y_offset, color=colors[i], width=0.5)

    y_offset = y_offset + stats_20.loc[row]
df['date'] = [x.split('-')[0] for x in df['date'].tolist()]

df.head()
df['total_score'] = df.home_score + df.away_score

df.head()
goals_per_match = df.groupby('date').total_score.mean()

goals_per_match.head()
_, ax = plt.subplots(1,1, figsize=(12, 8)) 

index = np.arange(len(goals_per_match.index)) 

plt.xticks(index, goals_per_match.index) 

plt.bar(index, goals_per_match.tolist())

for ind, label in enumerate(ax.get_xticklabels()):

    if ind % 10 == 0:  # every 10th label is kept

        label.set_visible(True)

    else:

        label.set_visible(False)