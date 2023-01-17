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



from IPython.display import display

pd.options.display.max_columns = None
meta = pd.read_csv('/kaggle/input/nba-playoff-2020/nba_playoffs_meta.csv')

meta
games = pd.read_csv('/kaggle/input/nba-playoff-2020/nba_playoffs_games.csv')

games
df = pd.merge(games, meta, left_on='home_team', right_on='team_short')

df = df.drop(['team', 'team_short', 'conference'], axis=1)

df['home_team_win_ratio'] = df['game_won'] / df['game_played']

df['home_team_points_per_game'] = df['points_scored'] / df['game_played']

df['home_team_missed_points'] = df['points_missed'] / df['game_played']

df['home_team_home_win_ratio'] = df['game_won_home'] / (df['game_won_home'] + df['game_lose_home'])

df['home_team_away_win_ratio'] = df['game_won_away'] / (df['game_won_away'] + df['game_lose_away'])

df = df.drop(['game_won', 'game_lose', 'game_played', 'points_scored', 'points_missed', 'game_won_home', 'game_lose_home', 'game_won_away', 'game_lose_away'], axis=1)

df.columns = ['series_id', 'home_team', 'away_team', 'home_points', 'away_points', 'round', 'game', 'score_home', 'score_away', 'home_conference_position',

              'home_team_win_ratio', 'home_team_points_per_game', 'home_team_missed_points', 'home_team_home_win_ratio', 'home_team_away_win_ratio']

df = pd.merge(df, meta, left_on='away_team', right_on='team_short')

df = df.drop(['team', 'team_short', 'conference'], axis=1)

df['away_team_win_ratio'] = df['game_won'] / df['game_played']

df['away_team_points_per_game'] = df['points_scored'] / df['game_played']

df['away_team_missed_points'] = df['points_missed'] / df['game_played']

df['away_team_home_win_ratio'] = df['game_won_home'] / (df['game_won_home'] + df['game_lose_home'])

df['away_team_away_win_ratio'] = df['game_won_away'] / (df['game_won_away'] + df['game_lose_away'])

df = df.drop(['game_won', 'game_lose', 'game_played', 'points_scored', 'points_missed', 'game_won_home', 'game_lose_home', 'game_won_away', 'game_lose_away'], axis=1)
X = df.drop(['home_points', 'away_points'], axis=1)

y = df[['home_points', 'away_points']]

X
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score
Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("neg_root_mean_squared_error Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("neg_root_mean_squared_error Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 1}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 0

Xtr['score_away'] = 0

Xtr['home_conference_position'] = X[X['home_team'] == 'BOS']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'BOS']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'BOS']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('BOSTON points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'BOS'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 2}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 0

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'BOS']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'BOS']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'BOS']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('BOSTON points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'BOS'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 3}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 1

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'BOS']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'BOS']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'BOS']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_away_win_ratio'].values[0]

Xtr
print('MIAMI points: ', int(model_home.predict(Xtr)[0]))

print('BOSTON points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'MIA'

Xtr['away_team'] = 'BOS'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 4}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 2

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'BOS']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'BOS']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'BOS']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_away_win_ratio'].values[0]

Xtr
print('MIAMI points: ', int(model_home.predict(Xtr)[0]))

print('BOSTON points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'MIA'

Xtr['away_team'] = 'BOS'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 5}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 1

Xtr['score_away'] = 3

Xtr['home_conference_position'] = X[X['home_team'] == 'BOS']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'BOS']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'BOS']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'BOS']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('BOSTON points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'BOS'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])



X_semi = X.tail(5)

y_semi = y.tail(5)
X = X.head(66)

y = y.head(66)
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 1}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 0

Xtr['score_away'] = 0

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'DEN']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'DEN']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'DEN']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('DENVER points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'DEN'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 2}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 1

Xtr['score_away'] = 0

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'DEN']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'DEN']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'DEN']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('DENVER points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'DEN'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 3}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 1

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'DEN']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'DEN']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'DEN']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr
print('DENVER points: ', int(model_home.predict(Xtr)[0]))

print('LAKERS points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'DEN'

Xtr['away_team'] = 'LAL'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 4}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 1

Xtr['score_away'] = 2

Xtr['home_conference_position'] = X[X['home_team'] == 'DEN']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'DEN']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'DEN']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr
print('DENVER points: ', int(model_home.predict(Xtr)[0]))

print('LAKERS points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'DEN'

Xtr['away_team'] = 'LAL'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 5}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 3

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'DEN']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'DEN']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'DEN']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'DEN']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('DENVER points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'DEN'

X = pd.concat([X, Xtr])
X = pd.concat([X, X_semi])

y = pd.concat([y, y_semi])
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 1}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 0

Xtr['score_away'] = 0

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])
Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 2}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 0

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 3}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 2

Xtr['score_away'] = 0

Xtr['home_conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr
print('MIAMI points: ', int(model_home.predict(Xtr)[0]))

print('LAKERS points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'MIA'

Xtr['away_team'] = 'LAL'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 4}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 2

Xtr['score_away'] = 1

Xtr['home_conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr
print('MIAMI points: ', int(model_home.predict(Xtr)[0]))

print('LAKERS points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'MIA'

Xtr['away_team'] = 'LAL'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 5}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 2

Xtr['score_away'] = 2

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'LAL'

Xtr['away_team'] = 'MIA'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 6}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 3

Xtr['score_away'] = 2

Xtr['home_conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr
print('MIAMI points: ', int(model_home.predict(Xtr)[0]))

print('LAKERS points: ', int(model_away.predict(Xtr)[0]))
ytr = pd.DataFrame({'home_points': int(model_home.predict(Xtr)[0]), 'away_points': int(model_away.predict(Xtr)[0])}, index=[0])

y = pd.concat([y, ytr])

Xtr['home_team'] = 'MIA'

Xtr['away_team'] = 'LAL'

X = pd.concat([X, Xtr])



Xtr = X.drop(['series_id', 'home_team', 'away_team', 'round'], axis=1)

model_home = DecisionTreeRegressor(random_state=666)

target = y['home_points']

model_home.fit(Xtr, target)

scores = cross_val_score(model_home, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Home: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

target = y['away_points']

model_away = DecisionTreeRegressor(random_state=666)

model_away.fit(Xtr, target)

scores = cross_val_score(model_away, Xtr, target, cv=10, scoring='neg_root_mean_squared_error')

print("Score Away: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Xtr = pd.DataFrame([], columns=['game', 'score_home', 'score_away', 'home_conference_position',

       'home_team_win_ratio', 'home_team_points_per_game',

       'home_team_missed_points', 'home_team_home_win_ratio',

       'home_team_away_win_ratio', 'conference_position',

       'away_team_win_ratio', 'away_team_points_per_game',

       'away_team_missed_points', 'away_team_home_win_ratio',

       'away_team_away_win_ratio'])



Xtr = Xtr.append({'game': 7}, ignore_index=True)

Xtr['game'] = Xtr['game'].astype(np.int8)

Xtr['score_home'] = 3

Xtr['score_away'] = 3

Xtr['home_conference_position'] = X[X['home_team'] == 'LAL']['home_conference_position'].values[0]

Xtr['home_team_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_win_ratio'].values[0]

Xtr['home_team_points_per_game'] = X[X['home_team'] == 'LAL']['home_team_points_per_game'].values[0]

Xtr['home_team_missed_points'] = X[X['home_team'] == 'LAL']['home_team_missed_points'].values[0]

Xtr['home_team_home_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_home_win_ratio'].values[0]

Xtr['home_team_away_win_ratio'] = X[X['home_team'] == 'LAL']['home_team_away_win_ratio'].values[0]

Xtr['conference_position'] = X[X['home_team'] == 'MIA']['home_conference_position'].values[0]

Xtr['away_team_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_win_ratio'].values[0]

Xtr['away_team_points_per_game'] = X[X['home_team'] == 'MIA']['home_team_points_per_game'].values[0]

Xtr['away_team_missed_points'] = X[X['home_team'] == 'MIA']['home_team_missed_points'].values[0]

Xtr['away_team_home_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_home_win_ratio'].values[0]

Xtr['away_team_away_win_ratio'] = X[X['home_team'] == 'MIA']['home_team_away_win_ratio'].values[0]

Xtr
print('LAKERS points: ', int(model_home.predict(Xtr)[0]))

print('MIAMI points: ', int(model_away.predict(Xtr)[0]))