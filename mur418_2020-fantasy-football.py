import pandas as pd



import matplotlib.pyplot as plt
qbs = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/qb_stats_and_projections.csv')

wrs = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/wr_stats_and_projections.csv')

rbs = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/rb_stats_and_projections.csv')

te = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/te_stats_and_projections.csv')

dst = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/defense_stats_and_projections.csv')

kickers = pd.read_csv('../input/espn-2019-stats-and-2020-nfl-fantasy-projections/kicker_stats_and_projections.csv')
#for now, lets rename defence team name to player name

temp_dst = dst.rename(columns={"TEAM NAME": "PLAYER NAME"})

temp_dst['PLAYER POSITION'] = 'DST'



#rename wrs position to player position .. just like the rest of the tables

wrs = wrs.rename(columns={"POSITION": "PLAYER POSITION"})
full_df = pd.concat([qbs, wrs, rbs, te, temp_dst, kickers])

print(full_df.columns)

names_2019_2020 = full_df[['PLAYER NAME', 'PLAYER POSITION','2019 FPTS', '2020 FPTS']]

names_2019_2020
rbs
names_2019_2020.info()
plt.figure(figsize=(10,7))

plt.scatter(names_2019_2020['2019 FPTS'], names_2019_2020['2020 FPTS'], alpha = 0.7)

plt.title('2020 Fantasy Points vs 2020 Fantasy Points')

plt.show()
players_per_pos = full_df.groupby('PLAYER POSITION').size()

players_per_pos
full_df.loc[full_df['PLAYER POSITION'] == 'DT, RB']
full_df.loc[full_df['PLAYER POSITION'] == '--']

full_df = full_df.drop(190)
players_per_pos = full_df.groupby('PLAYER POSITION').size()

players_per_pos
players_per_pos.values
plt.figure(figsize=(10,7))

plt.bar(players_per_pos.index,players_per_pos.values)

plt.title('Number of players per postition')

plt.xlabel('Position')

plt.ylabel('Number of players')

plt.show()
avg_points_per_pos = full_df.groupby('PLAYER POSITION')['2019 FPTS'].mean()
avg_points_per_pos
plt.figure(figsize=(10,7))

plt.bar(avg_points_per_pos.index,avg_points_per_pos.values)

plt.title('Average Number of points by position')

plt.xlabel('Position')

plt.ylabel('Average number of points')

plt.show()
#15 points per game = 16 * 15 = 240

players_scoring_more_than_15_2020 = names_2019_2020.loc[names_2019_2020['2020 FPTS'] > (15*16)].sort_values('2020 FPTS').groupby('PLAYER POSITION').count()

players_scoring_more_than_15_2020
#no defenses expected to score more than 15 points per game?

dst['2020 FPTS'].head() #nope
plt.figure(figsize=(10,7))

# print(players_scoring_more_than_15_2020.index)

# print(players_scoring_more_than_15_2020['PLAYER NAME'].to_list())







plt.bar(players_scoring_more_than_15_2020.index,players_scoring_more_than_15_2020['PLAYER NAME'].to_list())

plt.title('Number of players expected to score more than 15 points per game in 2020')

plt.xlabel('Position')

plt.ylabel('Number of players')

# plt.show()
#20 points per game = 16 * 15 = 320

players_scoring_more_than_20_2020 = names_2019_2020.loc[names_2019_2020['2020 FPTS'] > 320].sort_values('2020 FPTS').groupby('PLAYER POSITION').count()

players_scoring_more_than_20_2020
plt.figure(figsize=(10,7))



plt.bar(players_scoring_more_than_20_2020.index,players_scoring_more_than_20_2020['PLAYER NAME'].to_list())

plt.title('Number of players expected to score more than 20 points per game in 2020')

plt.xlabel('Position')

plt.ylabel('Number of players expected to score more than 20 points')

# plt.show()
#name of players who are expected to score more than 20 points per game

names_2019_2020.loc[names_2019_2020['2020 FPTS'] > 320].sort_values('2020 FPTS')
#between 10 and 15 

players_scoring_bw_10_15_2020 = names_2019_2020.loc[(names_2019_2020['2020 FPTS'] > 160) & (names_2019_2020['2020 FPTS'] < 15*16)].sort_values('2020 FPTS').groupby('PLAYER POSITION').count()

players_scoring_bw_10_15_2020
plt.figure(figsize=(10,7))



plt.bar(players_scoring_bw_10_15_2020.index,players_scoring_bw_10_15_2020['PLAYER NAME'].to_list())

plt.title('Number of players expected to score between 10 and 15 points per game, grouped by position')

plt.xlabel('Position')

plt.ylabel('Number of players')

plt.show()
sorted_by_proj = names_2019_2020.sort_values('2020 FPTS', ascending = False)

sorted_by_proj
#drop all players who are expected zero points in 2020

sorted_by_proj = sorted_by_proj[sorted_by_proj['2020 FPTS'] != 0]

sorted_by_proj
plt.figure(figsize=(10,7))

plt.scatter(list(range(len(sorted_by_proj))), sorted_by_proj['2020 FPTS'], s =1)

plt.show()