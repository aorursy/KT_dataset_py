import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime, date
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')
df.head()
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns.\n')
print(f'Column names: {df.columns.values}')
df.dtypes
print(df.isna().sum())
print("Highest rating: ", df.rating.max(), " - ", df[df.rating == df.rating.max()].full_name.values[0])
print("Lowest rating: ", df.rating.min(), " - ", df[df.rating == df.rating.min()].full_name.values[0])
average_rating = len(df.rating.value_counts())
print('Average rating value: {}'.format(round(df.rating.mean(), 1)))
fig = plt.figure(figsize = (10, 5))
plt.hist(df.rating, bins=average_rating)
plt.xlabel('Rating')
plt.ylabel('Players')
plt.title('Players and ratings histogram')
plt.show()
labels = [key for key in df.jersey.value_counts(dropna=False).keys()]
values = [value for value in df.jersey.value_counts(dropna=False).values]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10,10))
rects = ax.barh(x, values)

ax.set_xlabel('Players')
ax.set_ylabel('Jersey numbers')
ax.set_yticks(ticks=x)
ax.set_yticklabels(labels)
ax.set_title('Jersey numbers')

plt.show()
labels = [key for key in df.team.value_counts(dropna=False).keys()]
values = [value for value in df.team.value_counts(dropna=False).values]

x = np.arange(len(labels))
fig, ax = plt.subplots()
rects = ax.barh(x, values)

ax.set_xlabel('Number of players in teams')
ax.set_ylabel('Team')
ax.set_yticks(ticks=x)
ax.set_yticklabels(labels)
ax.set_title('Number of players in teams')

plt.show()
free_agents = df[df['team'].isna()]
print(f'Total free agents: {free_agents.shape[0]}')
free_agents
fig = plt.figure(figsize = (10, 5))
sns.countplot('position', data = df, order = df['position'].value_counts().index)
plt.xlabel('Player positions')
plt.ylabel('Count of players')
plt.title('Player positions')
plt.show()
position_rating = df[['position','rating']].groupby('position').mean().sort_values(by='rating', ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=position_rating.rating, y=position_rating.index)
plt.xticks()
plt.xlabel('Position')
plt.ylabel('Average rating')
plt.title('Average ranking of players by position')
plt.show()