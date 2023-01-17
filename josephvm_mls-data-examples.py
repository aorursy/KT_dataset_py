import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


match_df = pd.read_csv("../input/major-league-soccer-dataset/matches.csv")

event_df = pd.read_csv("../input/major-league-soccer-dataset/events.csv")

table_df = pd.read_csv("../input/major-league-soccer-dataset/all_tables.csv")

player_df = pd.read_csv("../input/major-league-soccer-dataset/all_players.csv")

gk_df = pd.read_csv("../input/major-league-soccer-dataset/all_goalkeepers.csv")
match_df.tail(3)
# to get rid of the starting and bench columns

match2_df = match_df[[item for item in match_df.columns.tolist() if 'starting' not in item and 'bench' not in item]]

match2_df.head()
match2_df.columns.tolist()
# Depending on what you're doing, there's other columns you can safely drop

# for example, unless you're merging this with other soccer data, 'league' can be dropped

# for example, 'shootout' and 'game_status' won't really tell you anything for regular 

# ... season games as there are no shootouts

match3_df = match2_df.drop(['time (utc)', 'venue', 'league', 'part_of_competition', 'shootout', 'game_status'], axis=1)

match3_df.head()
# Using the 'id' column from '2020_matches.csv', you can get the events for a game

# Note: This is the full commentary, not just key events

one_game_events_df = event_df[ event_df['id']==571931 ]

print(len(one_game_events_df))

one_game_events_df.head()
# Specific events can be found by searching for keywords

# The format for specific events, like goals and yellow cards seem to be consistent

one_game_events_df[ one_game_events_df['Event'].str.contains('Goal!')]
# Similarly, all events for one team in a game

one_game_events_df[ one_game_events_df['Event'].str.contains('San Jose Earthquakes')]
# The table for each year has two 1s, two 2s, etc. as the teams are split into two conferences

# Across the years, the Eastern Conference is first and the Western Conference is second

table_2003 = table_df[ table_df['Year']==2003 ].reset_index()
east, west = np.array_split(table_2003, [table_2003[ table_2003['Place'] == 1 ].index[1]])

east
west
# Splitting the whole df into two lists, one for each conference

# Note: Some teams have switched conferences at some point

conferences = np.array_split(table_df, table_df[ table_df['Place'] == 1 ].index[1:].tolist())

easts, wests = conferences[::2], conferences[1::2]

print('east len', len(easts))

print(easts[0])

print('west len', len(wests))

print(wests[-1])
player_df = player_df[ player_df['Year'] == 2020]



# Default sorted by goals ('G')

player_df.head()
# you can search through two columns at once like so

player_df[ (player_df['SOG%'] > 50) & (player_df['SHTS'] > 10)].sort_values('SOG%', ascending=False)
gk_df = gk_df[ gk_df['Year'] == 2020 ]

gk_df.head()
# You can split the penalty kick column into two

gk_df[['PKNS', 'PKF']] = gk_df['PKG/A'].str.split('/', expand=True) # penalties not saved, penalties faced

gk_df.head()
player_df = pd.read_csv("../input/major-league-soccer-dataset/all_players.csv")



g_90 = []

g = []

for year in range(1996, 2020):

    p_df = player_df[ player_df['Year']==year ]

    g_90.append(p_df.iloc[0]['G/90min'])

    g.append(p_df.iloc[0]['G'])
fig, ax1 = plt.subplots(figsize=(8,6))



plt.title('Goals per 90 Mins for the Highest Scorer each Season', fontdict={'fontsize':18})

plt.xlabel('Season', fontdict={'fontsize':16})

ax1.set_ylabel('Goals per 90', fontdict={'fontsize':16})

ax1.plot(range(1996, 2020), g_90, c='r')

ax1.tick_params(axis='y', labelcolor='r', labelsize=14)



ax2 = ax1.twinx()

ax2.plot(range(1996, 2020), g, c='b')

ax2.set_ylabel('Goals', fontdict={'fontsize':16})

ax2.tick_params(axis='y', labelcolor='b', labelsize=14)



fig.tight_layout()

plt.show()