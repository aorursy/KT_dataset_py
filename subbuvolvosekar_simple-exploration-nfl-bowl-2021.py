# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt
players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/players.csv")

players.head()
players.height = players.height.str.replace('-','*12+').apply(lambda x: eval(x))

df = players.groupby('height')['nflId'].count().reset_index(name = 'counts')

df = df.sort_values(by = 'counts', ascending = False)

fig = px.bar(df, x='height', y='counts', color = 'height')

fig.update_layout(title_text="Average height: "+ str(round(players.height.mean(),2)))

fig.show()
df = players.groupby('weight')['nflId'].count().reset_index(name = 'counts')

fig = px.bar(df, x='weight', y='counts')

fig.show()
players.birthDate = pd.to_datetime(players.birthDate)

players['birthyear'] = players['birthDate'].dt.year

df = players.groupby('birthyear')['nflId'].count().reset_index(name = 'counts')

fig = px.bar(df, x='birthyear', y='counts', color = 'birthyear')

fig.show()
df = players.groupby('position')['nflId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='position', y='counts', color = 'position')

fig.show()
plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/plays.csv")

plays.head()
df = plays.groupby("quarter")['playId'].count().reset_index(name = 'counts')

fig = px.pie(df, values='counts', names='quarter', title='Quarter Count')

fig.show()
df = plays.groupby("down")['playId'].count().reset_index(name = 'counts')

fig = px.pie(df, values='counts', names='down', title='Down Count')

fig.show()
df = plays.groupby("yardsToGo")['playId'].count().reset_index(name = 'counts')

fig = plt.figure(figsize=(15,10))

ax = fig.add_axes([0,0,1,1])

ax = sns.barplot(x="yardsToGo", y="counts", data=df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.set_title('Yards to go')

plt.show()
df = plays.groupby("playType")['playId'].count().reset_index(name = 'counts')

fig = px.pie(df, values='counts', names='playType', title='playType Count')

fig.show()
fig = plt.figure(figsize=(12,8))

ax = fig.add_axes([0,0,1,1])

cr = plays.groupby("passResult")['playId'].count().reset_index(name = 'counts')

ax = sns.barplot(x="passResult", y="counts", data=cr)
fig = plt.figure(figsize=(12,8))

ax = fig.add_axes([0,0,1,1])

cr = plays.groupby("offensePlayResult")['playId'].count().reset_index(name = 'counts')

cr = cr.sort_values(by= 'counts', ascending = False)

ax = sns.barplot(x="offensePlayResult", y="counts", data=cr[:20])
df = plays.groupby("isDefensivePI")['playId'].count().reset_index(name = 'counts')

fig = px.pie(df, values='counts', names='isDefensivePI', title='isDefensivePI Count')

fig.show()
games = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/games.csv")

games.head()
df = games.groupby('homeTeamAbbr')['gameId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='homeTeamAbbr', y='counts', color = 'homeTeamAbbr')

fig.show()
df = games.groupby('visitorTeamAbbr')['gameId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='visitorTeamAbbr', y='counts', color = 'visitorTeamAbbr')

fig.show()
df = games.groupby('gameDate')['gameId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='gameDate', y='counts', color = 'gameDate')

fig.show()
df = games.groupby('gameTimeEastern')['gameId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='gameTimeEastern', y='counts', color = 'gameTimeEastern')

fig.show()
df = games.groupby('week')['gameId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, x='week', y='counts', color = 'week')

fig.show()