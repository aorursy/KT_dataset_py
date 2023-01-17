import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
sns.set_style('ticks')
data_matches = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_matches.csv')

data_players = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_players.csv')

data_champions = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_champions.csv')
data_matches.info()
data_matches.head()
del data_matches['Unnamed: 0']
data_matches.isna().sum()
data_matches['date'] = pd.to_datetime(data_matches['date'])
data_players.info()
data_players.head()
del data_players['Unnamed: 0']
data_players.isna().sum()
data_players['heraldtime'].isna().sum()
del data_players['heraldtime']
data_champions.info()
data_champions.head()
del data_champions['Unnamed: 0']
data_champions.isna().sum()
data_matches.describe()
winners = set(data_matches['winner'].explode().unique())

winners
teams = set(data_matches['team1'].explode().unique())

teams
teams.difference(winners)
data_players.describe()
data_champions.describe()
data_matches.head()
mvp_data = data_matches['mvp'].value_counts().to_frame().reset_index()

mvp_data.columns = ['name', 'mvp_count']

mvp_data.head()
fig, ax = plt.subplots()

fig.set_size_inches(50.7, 25.27)

sns.barplot(data=mvp_data, x='name', y='mvp_count')
sides = ['Blue', 'Red']

def count_win_on_side(row):

    if (row['winner'] == row['blue']):

        return pd.Series([1, 0], sides)

    else:

        return pd.Series([0, 1], sides)



data_sides = data_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))



colors = ['#51acc6', '#ea96a3']

plt.pie(data_sides, colors=colors, labels=sides)



ax.set_title('Distribution of the winning percentage by side', pad=20)

plt.axis('equal')

plt.show()
pbp_caster_data = data_matches['pbp_caster'].value_counts().to_frame().reset_index()

pbp_caster_data.columns = ['name', 'count']

fig, ax = plt.subplots()

fig.set_size_inches(7, 5)

sns.barplot(data=pbp_caster_data, x='name', y='count')
color_caster_data = data_matches['color_caster'].value_counts().to_frame().reset_index()

color_caster_data.columns = ['name', 'count']

fig, ax = plt.subplots()

fig.set_size_inches(33, 10)

sns.barplot(data=color_caster_data, x='name', y='count')
data_winner = data_matches['winner'].value_counts().to_frame().reset_index()

data_winner.columns = ['name', 'count']

fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect='equal'))



wedges, texts = plt.pie(data_winner['count'], labels=data_winner['name'])



ax.set_title('Distribution of the winning percentage by team', pad=20)

ax.legend(wedges, data_winner['name'],

          title='Teams',

          loc='center left',

          bbox_to_anchor=(1, 0, 0.5, 1))

plt.axis('equal')

plt.show()
data_players.head()
columns_ban = ['ban1', 'ban2', 'ban3', 'ban4', 'ban5']

data_bans = None

for column_name in columns_ban:

    # This is player's data, so we need to divide ban count by members count in a team

    bans = data_players[column_name].value_counts()/5

    bans = bans.astype('int').to_frame().reset_index()

    if (data_bans is None):

        data_bans = bans

    else:

        data_bans = data_bans.merge(bans, on='index')

data_bans['ban_count'] = data_bans['ban1'] + data_bans['ban2'] + data_bans['ban3'] + data_bans['ban4'] + data_bans['ban5']

del data_bans['ban1'], data_bans['ban2'], data_bans['ban3'], data_bans['ban4'], data_bans['ban5']

data_bans.columns = ['champion', 'ban_count']

print(data_bans)
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect='equal'))



wedges, texts = plt.pie(data_bans['ban_count'], labels=data_bans['champion'])



ax.set_title('Distribution of the ban percentage by champion', pad=20)

ax.legend(wedges, data_bans['champion'],

          title='Champions',

          loc='center left',

          bbox_to_anchor=(1, 0, 0.5, 1))

plt.axis('equal')

plt.show()
data_players.groupby('player')['csat15'].mean().sort_values(ascending=False)
data_players.groupby('player')['xpat10'].mean().sort_values(ascending=False)
data_players.groupby('player')['teamkills'].sum().sort_values(ascending=False)
data_players.groupby('player')['ft'].sum().sort_values(ascending=False)
data_players.groupby('player')['totalgold'].mean().sort_values(ascending=False)
data_players.groupby('player')['pentas'].sum().sort_values(ascending=False)
data_players.groupby('player')['fb'].sum().sort_values(ascending=False)
data_players.groupby('player')['minionkills'].mean().sort_values(ascending=False)
data_players.groupby('player')['wards'].mean().sort_values(ascending=False)
data_players.groupby('player')['visionwards'].mean().sort_values(ascending=False)
data_players.groupby('player')['dmgtochamps'].mean().sort_values(ascending=False)
data_players.groupby('player')['fbaron'].sum().sort_values(ascending=False)
data_champions.head()
data_champions.sort_values(by='win_total', ascending=False)
data_champions.sort_values(by='winrate_total', ascending=False)
data_matches.head()
data_phoenix_win_matches = data_matches.loc[((data_matches['team1'] == 'FunPlus Phoenix') | (data_matches['team2'] == 'FunPlus Phoenix') & (data_matches['winner'] == 'FunPlus Phoenix'))]
data_phoenix_win_matches
data_phoenix_win_sides = data_phoenix_win_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))



colors = ['#51acc6', '#ea96a3']

plt.pie(data_phoenix_win_sides, colors=colors, labels=sides)



ax.set_title('Distribution of the FunPlus Phoenix winning percentage by side', pad=20)

plt.axis('equal')

plt.show()
data_phoenix_lose_matches = data_matches.loc[(((data_matches['team1'] == 'FunPlus Phoenix') | (data_matches['team2'] == 'FunPlus Phoenix')) & (data_matches['winner'] != 'FunPlus Phoenix'))]
data_phoenix_lose_sides = data_phoenix_lose_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))



colors = ['#51acc6', '#ea96a3']

plt.pie(data_phoenix_lose_sides, colors=colors, labels=sides)



ax.set_title('Distribution of the FunPlus Phoenix lose percentage by side', pad=20)

plt.axis('equal')

plt.show()
data_phoenix_lose_matches
phoenix_players = ['GimGoon', 'Lwx', 'Crisp', 'Doinb', 'Tian']
data_players.head()
phoenix_player_data = data_players.loc[data_players['player'].isin(phoenix_players)]
phoenix_player_data
data_players.groupby('player')['teamkills'].sum().sort_values(ascending=False)
data_players.groupby('player')['fbaron'].sum().sort_values(ascending=False)
data_players.groupby('player')['dmgtochamps'].mean().sort_values(ascending=False).head(15)
data_players.groupby('player')['visionwards'].mean().sort_values(ascending=False).head(15)
data_players.groupby('player')['wards'].mean().sort_values(ascending=False).head(15)
data_players.groupby('player')['minionkills'].mean().sort_values(ascending=False).head(15)
data_players.groupby('player')['fb'].sum().sort_values(ascending=False).head(15)