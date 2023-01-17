# Import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Create Dataframe
df = pd.read_csv('../input/train.csv')

#Look data
df.head(5)
fig = plt.figure()
axis = fig.add_subplot(1,1,1)
axis.scatter(df['boosts'],df['winPlacePerc'])
axis.set(title='Win Place Percentile by Boost Used', xlabel='Boosts Used', ylabel='Win Place Percentile')
plt.show()
winners = df[df['winPlacePerc'] >= 0.9]
winners['boosts'].describe()
# Transform data for teams.
df_groups = (df.groupby('groupId', as_index=False).agg({'Id':'count', 'matchId':'mean', 'assists':'sum', 'boosts':'sum',
                                'damageDealt':'sum', 'DBNOs':'sum', 'headshotKills':'sum',
                                'heals':'sum', 'killPlace':'mean', 'killPoints':'max', 'kills':'sum',
                                'killStreaks':'mean', 'longestKill':'mean', 'maxPlace':'mean', 'numGroups':'mean',
                                'revives':'sum', 'rideDistance':'max', 'roadKills':'sum', 'swimDistance':'max',
                                'teamKills':'sum', 'vehicleDestroys':'sum', 'walkDistance':'max',
                                'weaponsAcquired':'sum','winPoints':'max', 'winPlacePerc':'mean'}).rename(columns={'Id':'teamSize'}).reset_index())
# Show changes
df_groups.head(5)
# Get teams of size = 1
alone_players = df_groups[df_groups['teamSize'] == 1]

# Plot win place percentile by heals used
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(alone_players['heals'], alone_players['winPlacePerc'])
ax.set(title='Win Place Percentile by Heals Used', xlabel='Heals Used', ylabel='Win Place Percentile')
plt.show()
# Top 10% alone players (at least 4 heals during match)
alone_winners = alone_players[alone_players['winPlacePerc'] >= 0.9]
alone_winners = alone_winners[alone_winners['heals'] > 3]
# Describe patterns
alone_winners['heals'].describe()
# Walk Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['walkDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Walk Distance', xlabel='Walk Distance (m)', ylabel='Win Place Percentile')
plt.show()
# Swim Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['swimDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Swim Distance', xlabel='Swim Distance (m)', ylabel='Win Place Percentile')
plt.show()
# Ride Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['rideDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Ride Distance', xlabel='Ride Distance (m)', ylabel='Win Place Percentile')
plt.show()
# assists
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['assists'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Total Group Assists', xlabel='Total Group Assists', ylabel='Win Place Percentile')
plt.show()
# revives
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['revives'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Total Group Revives', xlabel='Total Group Revives', ylabel='Win Place Percentile')
plt.show()
print('Top 10%')
print(winners['assists'].describe())
print('\nOther Players')
print(df[df['winPlacePerc'] < 0.9]['assists'].describe())
print('Top 10%')
print(winners['revives'].describe())
print('\nOther Players')
print(df[df['winPlacePerc'] < 0.9]['revives'].describe())