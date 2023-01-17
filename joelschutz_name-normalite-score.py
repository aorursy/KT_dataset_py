import pandas as pd
players = pd.read_csv('/kaggle/input/football-players-and-teams-from-brazil/players.csv')

players.drop('Unnamed: 0', axis=1, inplace=True)

players.head(10)
players_name = pd.Series(dtype = 'object')

players_last_name = pd.Series(dtype = 'object')

for name in players.name:

    names = name.split(' ', 1)

    first_name = names[0]

    last_name = names[1] if len(names) > 1 else 'None'

    players_name = players_name.append(pd.Series(first_name))

    players_last_name = players_last_name.append(pd.Series(last_name))
first_name_freq = players_name.value_counts()

first_name_freq
last_name_freq = players_last_name.value_counts()

last_name_freq.pop('None')

last_name_freq
name_norm_score = pd.Series(dtype = 'int64')

for index in players.index:

    names = players.name[index].split(' ', 1)

    first_name = names[0]

    last_name = names[1] if len(names) > 1 else None

    score = first_name_freq[first_name]

    score += last_name_freq[last_name] if last_name else 0

    name_norm_score = name_norm_score.append(pd.Series(score))

name_norm_score.index = players.index

name_norm_score
players['name_score'] = name_norm_score

players
players.sort_values(['name_score', 'name']).head()
players.to_csv('/kaggle/working/players_w_score.csv')