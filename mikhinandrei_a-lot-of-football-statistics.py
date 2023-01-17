import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import pandas as pd

import numpy as np



events = pd.read_csv('../input/events.csv')

info = pd.read_csv('../input/ginf.csv')

events.info()

info.info()
goals = events[events.is_goal == 1]

goals.describe()
# The most resultative leagues

info['scored'] = info['fthg'] + info['ftag']

print('Overall scores')

info.groupby('country').scored.sum().sort_values(ascending=False).head()
print('Mean scores')

info.groupby('country').scored.mean().sort_values(ascending=False).head()
# The most resultative teams

home_goals = info.groupby('ht').fthg.sum()

away_goals = info.groupby('at').ftag.sum()

sum_goals = (home_goals + away_goals).sort_values(ascending=False)

sum_goals.head(20)
# Most missing teams

home_missed = info.groupby('ht').ftag.sum()

away_missed = info.groupby('at').fthg.sum()

sum_missed = (home_missed + away_missed).sort_values(ascending=False)

sum_missed.head(20)
(sum_goals - sum_missed).sort_values(ascending=False).head(20)
free_kicks = goals[goals.situation == 4]

best_kickers = free_kicks.groupby('player').player.count().sort_values(ascending=False)

best_kickers.head(20)
goals[goals.situation == 4].groupby('player').player.count().sort_values(ascending=False).head(20)
penalties_goals = goals[goals.location == 14]

penalties_scored = penalties_goals.groupby('player').player.count().sort_values(ascending=False)

penalties_scored.head(20)
non_goals = events[events.is_goal == 0]

penalties_non_goals = non_goals[non_goals.location == 14]

penalties_missed = penalties_non_goals.groupby('player').player.count().sort_values(ascending=False)

penalties_missed.head(20)
penalties_stats = pd.concat([penalties_scored.T, penalties_missed.T], axis=1)

penalties = penalties_stats.fillna(0)

penalties.columns.values[0] = 'goals'

penalties.columns.values[1] = 'missed'

penalties['total'] = penalties['goals'] + penalties['missed']

penalties['success'] = penalties['goals'] / penalties['total']

penalties['unsuccess'] = penalties['missed'] / penalties['total']

penalties.sort_values(by='success', ascending=False).head(10)
penalties_best = penalties[penalties.goals >= 10].sort_values(by='success', ascending=False)

penalties_best.head(20)
penalties_worst = penalties[penalties.goals >= 10].sort_values(by='unsuccess', ascending=False)

penalties_worst.head(20)
fig, ax = plt.subplots(1,1, figsize=(20, 8))

best_20 = penalties_best.loc[penalties_best.index.tolist()[0]:penalties_best.index.tolist()[19], 'success':'unsuccess']

y_offset = np.zeros(len(best_20.index.tolist()))

index = np.arange(len(best_20.index.tolist()))

plt.title('The best penaltists')

plt.xticks(index, best_20.index.tolist(), rotation=30)

plt.bar(index, best_20['success'], color='green')
fig, ax = plt.subplots(1,1, figsize=(20, 8))

worst_20 = penalties_worst.loc[penalties_worst.index.tolist()[0]:penalties_worst.index.tolist()[19], 'success':'unsuccess']

y_offset = np.zeros(len(worst_20.index.tolist()))

index = np.arange(len(worst_20.index.tolist()))

plt.title('The worst penaltists')

plt.xticks(index, worst_20.index.tolist(), rotation=30)

plt.bar(index, worst_20['unsuccess'], color='red')
all_penalties = pd.concat([penalties_goals, penalties_non_goals])

all_penalties.groupby('is_goal').count()
all_penalties.groupby('event_team').event_team.count().sort_values(ascending=False).head(20)
all_penalties.groupby('opponent').event_team.count().sort_values(ascending=False).head(20)
bombardirs = goals.groupby('player').player.count().sort_values(ascending=False)

bombardirs.head(20)
# Right foot

right_foot_goals = goals[goals.bodypart == 1].groupby('player').player.count().sort_values(ascending=False)

right_foot_goals.head(20)
# Left foot

left_foot_goals = goals[goals.bodypart == 2].groupby('player').player.count().sort_values(ascending=False)

left_foot_goals.head(20)
# Head

head_goals = goals[goals.bodypart == 3].groupby('player').player.count().sort_values(ascending=False)

head_goals.head(20)
goals_distr = pd.concat([right_foot_goals, left_foot_goals, head_goals, bombardirs], axis=1).fillna(0)

goals_distr.columns.values[0] = 'rf'

goals_distr.columns.values[1] = 'lf'

goals_distr.columns.values[2] = 'head'

goals_distr.columns.values[3] = 'overall'

goals_distr = goals_distr.sort_values(by='overall', ascending=False)

goals_distr.head(20)
goals_distr['rf'] /= goals_distr['overall']

goals_distr['lf'] /= goals_distr['overall']

goals_distr['head'] /= goals_distr['overall']

goals_distr.head(20)
#Right foot

goals_distr = goals_distr[goals_distr.overall >= 20]

goals_distr = goals_distr.sort_values(by='rf', ascending=False)

goals_distr.head(10)
# Left foot

goals_distr = goals_distr[goals_distr.overall >= 20]

goals_distr = goals_distr.sort_values(by='lf', ascending=False)

goals_distr.head(10)
# Head

goals_distr = goals_distr[goals_distr.overall >= 20]

goals_distr = goals_distr.sort_values(by='head', ascending=False)

goals_distr.head(10)
# By player

attempts = events[events.event_type == 1]

attempts.groupby('player').player.count().sort_values(ascending=False).head(20)
attempts.groupby('event_team').player.count().sort_values(ascending=False).head(20)
shots_per_goal_pl = attempts.groupby('player').player.count() / bombardirs

shots_per_goal_pl.sort_values().head(20)
shots_per_goal_tm = attempts.groupby('event_team').player.count() / sum_goals

shots_per_goal_tm.sort_values().head(20)
fouls = events[events.event_type == 3]



# By team

fouls.groupby('event_team').player.count().sort_values(ascending=False).head(20)
# By player

fouls.groupby('player').player.count().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(1,1, figsize=(40, 20))

sns.set(font_scale=1)

time_distr = fouls.groupby('time').time.count()

time_distr.head()

x = np.arange(len(time_distr))

plt.bar(x, time_distr)
y_cards = events[events.event_type == (4 or 5)]



# By team

y_cards.groupby('event_team').player.count().sort_values(ascending=False).head(20)
# By player

y_cards.groupby('player').player.count().sort_values(ascending=False).head(20)
# Time distribution

fig, ax = plt.subplots(1,1, figsize=(40, 20))

sns.set(font_scale=1)

time_distr = y_cards.groupby('time').time.count()

time_distr.head()

x = np.arange(len(time_distr))

plt.bar(x, time_distr, color='yellow')
r_cards = events[events.event_type == 6]



# By team

r_cards.groupby('event_team').player.count().sort_values(ascending=False).head(20)
# By player

r_cards.groupby('player').player.count().sort_values(ascending=False).head(20)
# Time distribution

fig, ax = plt.subplots(1,1, figsize=(40, 20))

sns.set(font_scale=1)

time_distr = r_cards.groupby('time').time.count()

time_distr.head()

x = np.arange(len(time_distr))

plt.bar(x, time_distr, color='red')
hands = events[events.event_type == 10]

hands.groupby('player').event_type.count().sort_values(ascending=False).head(20)
offs = events[events.event_type == 9]



# By team

offs.groupby('event_team').event_type.count().sort_values(ascending=False).head(20)
# By player

offs.groupby('player').event_type.count().sort_values(ascending=False).head(20)
assists = goals[goals.event_type2 == 12]

assists.groupby('player2').event_type.count().sort_values(ascending=False).head(20)
autogoals = events[events.event_type2 == 15]

autogoals.groupby('player').player.count().sort_values(ascending=False).head(20)
substs = events[events.event_type == 7]

substs.groupby('event_team').event_type.count().sort_values(ascending=False).head(20)
substs.groupby('player_in').player_in.count().sort_values(ascending=False).head(20)
substs.groupby('player_out').player_out.count().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(1,1, figsize=(40, 20))

sns.set(font_scale=1)

time_distr = substs.groupby('time').time.count()

time_distr.head()

x = np.arange(len(time_distr))

plt.bar(x, time_distr, color='green')
substs_prepared = substs[['id_odsp', 'time', 'event_team', 'player_in']]

substs_prepared.columns.values[3] = 'player'

goals_prepared = goals[['id_odsp', 'time', 'event_team', 'player']]

res_substs = pd.merge(substs_prepared, goals_prepared, how='inner', on=['id_odsp', 'player'])

res_substs.head()
substs[substs.id_odsp == 'UBZQ4smg/'].head()
res_substs = res_substs[res_substs.time_x <= res_substs.time_y]

res_substs.head(20)
res_substs.groupby('player').player.count().sort_values(ascending=False).head(20)
res_substs.groupby('event_team_x').player.count().sort_values(ascending=False).head(20)