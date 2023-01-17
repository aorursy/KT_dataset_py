import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt
conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()
ids = [i[0] for i in c.execute('SELECT id FROM League').fetchall()]

names = [i[0] for i in c.execute('SELECT name FROM League').fetchall()]

id_league = {i: n for i, n in zip(ids, names)}

id_league
# Country ID

# Country id

ids = [i[0] for i in c.execute('SELECT id FROM Country').fetchall()]

names = [i[0] for i in c.execute('SELECT name FROM Country').fetchall()]

id_country = {i: n for i, n in zip(ids, names)};
c.execute('PRAGMA TABLE_INFO(Player_Stats)').fetchall()
cols = ", ".join(["home_player_Y"+str(i) for i in range(1,12)])

c.execute('SELECT {0:s} FROM Match'.\

         format(cols))

Y_array = c.fetchall()



Y = np.array([a for row in Y_array for a in row]) # flatten

from collections import Counter

print('Player Y value: # of instances in database (home players)')

Counter(Y)
EA_stats = {'player': ', '.join(['overall_rating']), #'attacking_work_rate', 'defensive_work_rate',

#                                   'crossing', 'finishing', 'heading_accuracy', 'short_passing',

#                                   'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

#                                   'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

#                                   'agility', 'reactions', 'balance', 'shot_power', 'jumping',

#                                   'stamina', 'strength', 'long_shots', 'interceptions',

#                                   'positioning', 'vision', 'penalties', 'marking',

#                                   'standing_tackle', 'sliding_tackle']),

            'goalie': ', '.join(['gk_diving', 'gk_handling', 'gk_kicking',

                                 'gk_positioning', 'gk_reflexes'])}





def getTeamScores(match_id, team, EA_stats,

                  printout=False,

                  group='forward_mid_defense_goalie'):

    ''' Return the cumulative average team scores for 

    a given EA Sports FIFA statistic. If particular EA

    stats are not in the database that stat is taken as

    the overall player rating. If any positional stat is

    unavailable (i.e. no goalie information) that stat is

    taken as the average of the others for that team.

    team : str

        'home' or 'away'

    EA_stat : dict

        Names of statistics to cumulate for goalie and players.

        e.g. {'player': 'overall_rating, heading_accuracy',

              'goalie': 'gk_diving, gk_handling'}

    printout : boolean

        Option to print out debug information,

        defaults to False.

    group : str

        How to group scores:

        'forward_mid_defense_goalie': output 4 values

        'all': output 1 value (currently not implemented)

    '''

    

    if team == 'home':

        player_cols = ', '.join(['home_player_'+str(i) for i in range(1,12)])

        player_Y_cols = np.array(['home_player_Y'+str(i) for i in range(1,12)])

    elif team == 'away':

        player_cols = ', '.join(['away_player_'+str(i) for i in range(1,12)])

        player_Y_cols = np.array(['away_player_Y'+str(i) for i in range(1,12)])

        

    # Get the player ids from the Match table

    c.execute('SELECT {0:s} FROM Match WHERE id={1:d}'.\

             format(player_cols, match_id))

    player_api_id = np.array(c.fetchall()[0])

    

    # Return dictionary of NaN if all items in the list are null

    # WARNING: I've hard-coded this dictionary

    if False not in [p==0 or p==None for p in player_api_id]:

#         raise LookupError('No player data found for Match table row_id={}'.\

#                                    format(match_id))

        return {'F': np.array([np.nan]), 'M': np.array([np.nan]),

                'D': np.array([np.nan]), 'G': np.array([np.nan])}

        

    # Remove any empty player entries (if player_api_id == None or nan)

    empty_mask = player_api_id != np.array(None)

    player_api_id = player_api_id[empty_mask]

    player_Y_cols = ', '.join(player_Y_cols[empty_mask])

    

    # Get the player positions from the Match table

    # We only care about the Y position to designate

    # forwards, midfielders, defense, and goalie

    

    c.execute('SELECT {0:s} FROM Match WHERE id={1:d}'.\

             format(player_Y_cols, match_id))

    player_Y = c.fetchall()[0]

    

    def givePosition(Y):

        ''' Input the Y position of the player (as opposed

        to the lateral X position) and return the categorical

        position. '''

        if Y == 1:

            return 'G'

        elif Y == 3:

            return 'D'

        elif Y == 5 or Y == 6 or Y == 7:

            return 'M'

        elif Y == 8 or Y == 9 or Y == 10 or Y == 11:

            return 'F'

        else:

#            sys.exit('Unknown value for Y: {}'.\

#                    format(Y))

            return 'NaN'



    player_pos = np.array([givePosition(Y) for Y in player_Y])

    

    # Get the match date

    

    def toDatetime(datetime):

        ''' Convert string date to datetime object. '''

        return dt.datetime.strptime(datetime, '%Y-%m-%d %H:%M:%S')



    c.execute('SELECT date FROM Match WHERE id={}'.\

             format(match_id))

    match_date = toDatetime(c.fetchall()[0][0])

    

    # Lookup the EA Sports stats for each player

    # The stats are time dependent so we have to

    # find the ones closest to the match date

    

    def getBestDate(player_id, match_date):

        ''' Find most suitable player stats to use based

        on date of match and return the corresponding row

        id from the Player_Stats table. ''' 

        c.execute('SELECT id FROM Player_Stats WHERE player_api_id={}'.\

                 format(player_id))

        ids = np.array([i[0] for i in c.fetchall()])

        c.execute('SELECT date_stat FROM Player_Stats WHERE player_api_id={}'.\

                 format(player_id))

        dates = [toDatetime(d[0]) for d in c.fetchall()]

        dates_delta = np.array([abs(d-match_date) for d in dates])

        return ids[dates_delta==dates_delta.min()][0]

    

    def fill_empty_stats(stats, stat_names):

        ''' Input the incomplete EA player stats and corresponing

        names, return the filled in stats list. Filling with

        overall_rating or averaging otherwise (i.e. for goalies

        where there is no overall_rating stat). '''

        if not np.sum([s==0 or s==None for s in stats]):

            return stats

        stats_dict = {sn: s for sn, s in zip(stat_names, stats)}

        try:

            fill = stats_dict['overall_rating']

        except:

            # Either a goalie or player with no overall rating

            # Filling with average of other stats

            fill = np.mean([s for s in stats if s!=0 and s!=None])

        filled_stats = []

        for s in stats:

            if s==None or s==0:

                filled_stats.append(fill)

            else:

                filled_stats.append(s)

        return filled_stats

    

    positions = ('G', 'D', 'M', 'F')

    average_stats = {}

    for position in positions:

        if printout: print(position)

        if position == 'G':

            stats = EA_stats['goalie']

        else:

            stats = EA_stats['player']

        position_ids = player_api_id[player_pos==position]

        average_stats[position] = np.zeros(len(stats.split(',')))

        for player_id in position_ids:

            if printout: print(player_id)

            best_date_id = getBestDate(player_id, match_date)

            c.execute('SELECT {0:s} FROM Player_Stats WHERE id={1:d}'.\

                     format(stats, best_date_id))

            query = np.array(c.fetchall()[0])

            query = fill_empty_stats(query, stats.split(', '))

            if printout: print(query)

            if sum([q==None or q==0 for q in query]):

                raise LookupError('Found null EA stats entry at stat_id={}'.\

                                  format(best_date_id))

#                 sys.exit('Found null EA stats entry at stat_id={}'.\

#                         format(best_date_id))

            average_stats[position] += query

            if printout: print('')

        average_stats[position] /= len(position_ids) # take average

            

    # Take average of goalie stats

    try:

        average_stats['G'] = np.array([average_stats['G'].mean()])

    except:

        # Missing info: (average_stats['G']) = 0

        pass

    

    # Insert missing stats

    insert_value = np.mean([v[0] for v in average_stats.values() if not np.isnan(v)])

    for k, v in average_stats.items():

        if np.isnan(v[0]):

            average_stats[k] = np.array([insert_value])

    

#     # Return a dictionary of numeric results as strings for storing in SQL table

#     return {key: ' '.join([str(v) for v in value]) for key, value in average_stats.items()}

#     ''' THE LINE ABOVE NEEDS A FIX - UNABLE TO ADD STRINGS LIKE THIS TO SQL TABLE '''        

    return average_stats
# Test of the function above

avg = getTeamScores(999, 'home', EA_stats, printout=True)

avg
# Null test of the function above

avg = getTeamScores(5, 'home', EA_stats, printout=True)

avg
# Get row ids for our Match table

all_ids = c.execute('SELECT id FROM Match').fetchall()

all_ids = [i[0] for i in sorted(all_ids)]



hF, hM, hD, hG = [], [], [], []

aF, aM, aD, aG = [], [], [], []

for i in all_ids:

    h_stats = getTeamScores(i, 'home', EA_stats, printout=False)

    hF.append(h_stats['F'][0])

    hM.append(h_stats['M'][0])

    hD.append(h_stats['D'][0])

    hG.append(h_stats['G'][0])

    a_stats = getTeamScores(i, 'away', EA_stats, printout=False)

    aF.append(a_stats['F'][0])

    aM.append(a_stats['M'][0])

    aD.append(a_stats['D'][0])

    aG.append(a_stats['G'][0])
df = pd.read_sql(sql='SELECT {} FROM Match'.\

                 format('id, country_id, league_id, season, stage, '+\

                        'date, home_team_api_id, away_team_api_id, '+\

                        'home_team_goal, away_team_goal'),

                 con=conn)
features = ['home_F_stats', 'home_M_stats', 'home_D_stats', 'home_G_stats',

            'away_F_stats', 'away_M_stats', 'away_D_stats', 'away_G_stats']



data = [hF, hM, hD, hG, aF, aM, aD, aG]



for f, d in zip(features, data):

    df[f] = d
df.head()
# Dropping NaNs

df = df.dropna()



# Adding a game state column:

# a list of the form [H, D, A]

#

# state = [1, 0, 0], result = 1 => Home team win

# state = [0, 1, 0], result = 2 => Draw

# state = [0, 0, 1], result = 3 => Away team win

H = lambda x: x[0] > x[1]

D = lambda x: x[0] == x[1]

A = lambda x: x[0] < x[1]

state, result = [], []

for goals in df[['home_team_goal', 'away_team_goal']].values:

    r = np.array([H(goals), D(goals), A(goals)])

    state.append(r)

    if (r == [1, 0, 0]).sum() == 3:

        result.append(1)

    elif (r == [0, 1, 0]).sum() == 3:

        result.append(2)

    elif (r == [0, 0, 1]).sum() == 3:

        result.append(3)

df['game_state'] = state

df['game_result'] = result



# Convert to datetimes

df['date'] = pd.to_datetime(df['date'])



# Map leagues names using dictionaries from earlier

df['country'] = df['country_id'].map(id_country)

df['league'] = df['league_id'].map(id_league)



# Average stats for teams (for each game)

f = lambda x: np.mean(x)

df['home_mean_stats'] = list(map(f, df[['home_F_stats', 'home_M_stats',

                                        'home_D_stats', 'home_G_stats']].values))

df['away_mean_stats'] = list(map(f, df[['away_F_stats', 'away_M_stats',

                                        'away_D_stats', 'away_G_stats']].values));
# Here is what we have ...

df.dtypes
sns.pairplot(data=df[features])

plt.suptitle('EA Sports FIFA positional game ratings correlations', fontsize=30, y=1.02);

plt.show();
df.date.hist(bins=100)

plt.title('Frequency of games in all countries')

plot_width = (df.date.max() - df.date.min()).days

bin_width = plot_width/100

print('bin_width = {0:.1f} days'.format(bin_width))

plt.show();
g = sns.FacetGrid(df, col='league', col_wrap=4)

g.map(plt.hist, 'date', bins=100)

for ax in g.axes.flat:  

    plt.setp(ax.get_xticklabels(), rotation=45)

plt.suptitle('Game frequency by league', fontsize=20, y=1.04);
league = np.unique(df.league.values)

N_entries = np.array([len(df[(df.league == L)]) for L in league])

N_entries = N_entries/N_entries.sum()

ax = sns.barplot(league, N_entries)

ax.set_ylabel('Percentage of data - total')

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.title('Amount of game data from each league')

plt.show();
league = np.unique(df.league.values)

N_entries = np.array([len(df[(df.league == L)&(df.season == '2015/2016')]) for L in league])

N_entries = N_entries/N_entries.sum()

ax = sns.barplot(league, N_entries)

ax.set_ylabel('Percentage of data - 2015/2016')

plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.title('Amount of 2015/2016 game data from each league')

plt.show();
fig = plt.figure(figsize=(12, 10))

sns.violinplot(x='home_mean_stats', y='league', data=df)

plt.title('Average EA Sports FIFA player ratings per team since 2009', y=1.02);
fig = plt.figure(figsize=(12, 10))

sns.violinplot(x='home_mean_stats', y='league', data=df[(df.season=='2015/2016')])

plt.title('Average EA Sports FIFA player ratings per team in 2015/2016', y=1.02);