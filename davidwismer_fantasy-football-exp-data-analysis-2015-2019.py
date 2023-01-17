# Data and statistics libraries

import pandas as pd

import numpy as np

import datetime

from datetime import datetime as dt, timedelta, date

from scipy.stats import pointbiserialr

from scipy.stats import f_oneway

from math import floor, ceil



# Visualization libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings; warnings.filterwarnings(action='once')



large = 22; med = 16; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

sns.set_style("white")

%matplotlib inline



import plotly

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px



plotly.offline.init_notebook_mode(connected=True)
# Playoffs in or out?

playoffs = True



# Specify offense table columns and datatypes

offense_col_types = {'gid' : int,

                     'player' : str, 

                     'pa' : float, 

                     'pc' : float, 

                     'py' : float, 

                     'ints' : float, 

                     'tdp' : float, 

                     'ra' : float, 

                     'ry' : float, 

                     'tdr' : float, 

                     'trg' : float, 

                     'rec' : float, 

                     'recy' : float, 

                     'tdrec' : float, 

                     'ret' : float, 

                     'rety' : float, 

                     'tdret' : float, 

                     'fuml' : float, 

                     'conv' : float, 

                     'snp' : float, 

                     'year' : int, 

                     'team' : str,

                     'dcp' : float

                    }



# Isolate column names

offense_cols = offense_col_types.keys()



# Import offense table

offense = pd.read_csv('../input/nfl-game-season-and-salary-data/OFFENSE.csv',

                      delimiter=',',

                      usecols=offense_cols,

                      dtype=offense_col_types

                     )



# Filter table to include only 2012 onward (data before that misses key columns)

offense = offense.loc[offense['year'] >= 2012]



# Add player_gid and player_year columns for merges and groupbys

offense['player_gid'] = offense['player']+ '-' + offense['gid'].map(str)

offense['player_year'] = offense['player']+ '-' + offense['year'].map(str)



# Import player table

player = pd.read_csv('../input/nfl-game-season-and-salary-data/PLAYER.csv',

                      delimiter=',',

                     )



# Import injury table

injury = pd.read_csv('../input/nfl-game-season-and-salary-data/INJURY.csv',

                      delimiter=',',

                     )



# Add player_gid column to injury table for merges

injury['player_gid'] = injury['player']+ '-' + injury['gid'].map(str)





# Import schedule table

schedule = pd.read_csv('../input/nfl-game-season-and-salary-data/SCHEDULE.csv',

                      delimiter=',',

                     )



# Create merge columns (team + season + week)

schedule['home_team_game'] = schedule['h'] + '-' + schedule['seas'].map(str) + '-' + schedule['wk'].map(str)

schedule['visit_team_game'] = schedule['v'] + '-' + schedule['seas'].map(str) + '-' + schedule['wk'].map(str)



# Merge offense table to player table to get positions

player_cols = ['player', 'pos1']

offense = pd.merge(offense, player[player_cols], left_on='player', right_on='player', how='left', suffixes=('', '_drop'))



# Merge injury table to offense table to get practice and game status (pstat, gstat)

injury_cols = ['player_gid', 'pstat', 'gstat']

offense = pd.merge(offense, 

                   injury[injury_cols], 

                   left_on='player_gid', 

                   right_on='player_gid', 

                   how='left', 

                   suffixes=('', '_drop')

                  )



# Fill NA's on injury columns

fill_values = {'pstat': 'No Injury', 

               'gstat': 'No Injury'

              }

offense = offense.fillna(value=fill_values)



# Merge offense table to schedule to bring in week column (wk)

sched_cols = ['gid','wk']

offense = pd.merge(offense, 

                   schedule[sched_cols],

                   left_on='gid',

                   right_on='gid',

                   how='left',

                   suffixes=('', '_drop')

                  )



# Create next_team_game column in order to help each player's next game

offense['next_wk'] = offense['wk'] + 1

offense['next_team_game'] = offense['team'] + '-' + offense['year'].map(str) + '-' + offense['next_wk'].map(str)





# Merge offense table to schedule table to bring over next game ids (gid)

sched_cols = ['home_team_game', 'gid']

offense = pd.merge(offense, 

                   schedule[sched_cols],

                   left_on='next_team_game',

                   right_on='home_team_game',

                   how='left',

                   suffixes=('', '_sched')

                  )



sched_cols = ['visit_team_game', 'gid']

offense = pd.merge(offense, 

                   schedule[sched_cols], 

                   left_on='next_team_game', 

                   right_on='visit_team_game', 

                   how='left', 

                   suffixes=('', '_sched2')

                  )
def add_full_name(data, player_data):

    player_cols = ['player', 'fname','lname']

    data = pd.merge(data, player_data[player_cols], left_on='player', right_on='player', how='left')

    

    data['full_name'] = data['fname'] + ' ' + data['lname']

    data = data.drop(columns=['fname', 

                              'lname'

                             ]

                    )

    

    return data



# Add full names to offense table

offense = add_full_name(offense, player)



# Add fullback table, remove fullbacks from offense table

fullback = pd.read_csv('../input/nfl-game-season-and-salary-data/FULLBACK.csv',

                      delimiter=',',

                     )



offense = pd.merge(offense,

                   fullback,

                   how='left',

                   left_on='full_name',

                   right_on='full_name'

                  )



offense['pos1'] = np.where(offense['fb'] == 'FB', 'FB', offense['pos1'])

offense = offense.drop(['full_name', 'fb'], axis=1)
# Drop unnecessary player positions and drop any player games with zero snaps

position_list = ['RB', 'WR', 'TE', 'QB']

offense = offense.loc[(offense['pos1'].isin(position_list)) &

                      (offense['snp'] > 0)

                     ]



# Replace NaN in gid columns and combine them to form next_gid column

fill_values = {'gid_sched': 0.0, 

               'gid_sched2': 0.0

              }

offense = offense.fillna(value=fill_values)

offense['next_gid'] = offense['gid_sched'].map(int) + offense['gid_sched2'].map(int)



# Drop columns used to create next_gid column

offense = offense.drop(['next_wk', 

                        'next_team_game',

                        'home_team_game',

                        'gid_sched',

                        'visit_team_game',

                        'gid_sched2'

                       ], 

                       axis=1

                      )



# Merge injury table to offense table to get practice and game status (pstat, gstat)

offense['player_next_gid'] = offense['player'] + '-' + offense['next_gid'].map(str)



injury_cols = ['player_gid', 'pstat', 'gstat']



offense = pd.merge(offense, 

                   injury[injury_cols],

                   left_on='player_next_gid',

                   right_on='player_gid',

                   how='left',

                   suffixes=('', '_next_game')

                  ).drop(['player_gid_next_game'],

                         axis=1

                        )



# Fill NA's on injury columns

fill_values = {'pstat_next_game': 'No Injury', 

               'gstat_next_game': 'No Injury'

              }

offense = offense.fillna(value=fill_values)







# Import sacks table to add sacks to offense table

plays = pd.read_csv('../input/nfl-game-season-and-salary-data/PLAY.csv',

                    delimiter=','

                   )

                   

sacks = pd.read_csv('../input/nfl-game-season-and-salary-data/SACK.csv',

                    delimiter=','

                   )





# Merge gid column to sacks table from plays

play_cols = ['pid', 'gid']

sacks = pd.merge(sacks,

                 plays[play_cols],

                 how='left',

                 left_on='pid',

                 right_on='pid'

                )



# Group sacks table by player_gid

sacks['player_gid'] = sacks['qb'] + '-' + sacks['gid'].map(str)

sacks_by_player_gid = (sacks[['player_gid', 'value']]

                       .groupby(['player_gid'])

                       .sum()

                       .reset_index()

                       .rename(columns={'value' : 'sk'})

                      )



# Merge sacks column to offense table

offense = pd.merge(offense,

                   sacks_by_player_gid,

                   how='left',

                   left_on='player_gid',

                   right_on='player_gid'

                  )



# Replace NaN with zero in sacks column

offense = offense.fillna(value={'sk' : 0.0})
# Establish scoring dictionaries

ppr = {'ppr' : 1.0,

       'pass_yd' : 1.0 / 25.0,

       'pass_td' : 6.0,

       'int' : -1.0,

       'sack' : -1.0,

       'rush_yd' : 1.0 / 10.0,

       'rush_td' : 6.0,

       'rec_yd' : 1.0 / 10.0,

       'rec_td' : 6.0,

       'ret_td' : 6.0,

       '2pc' : 2.0,

       'fum_lost' : -2.0

      }



reg = {'ppr' : 0.0,

       'pass_yd' : 1.0 / 25.0,

       'pass_td' : 6.0,

       'int' : -1.0,

       'sack' : -1.0,

       'rush_yd' : 1.0 / 10.0,

       'rush_td' : 6.0,

       'rec_yd' : 1.0 / 10.0,

       'rec_td' : 6.0,

       'ret_td' : 6.0,

       '2pc' : 2.0,

       'fum_lost' : -2.0

      }



# Establish color dictionary for presenting different positions

color_dict = {'WR' : ['blue', 'Blues'], 'RB' : ['green', 'Greens'], 'TE' : ['purple', 'Purples'], 'QB' : ['red', 'Reds']}
def add_scoring_columns(data, score_dict, label):

    # Add pt per activity columns from score_dict

    data['pass_yd_pt'] = score_dict['pass_yd']

    data['pass_td_pt'] = score_dict['pass_td']

    data['int_pt'] = score_dict['int']

    data['sack_pt'] = score_dict['sack']

    data['rush_yd_pt'] = score_dict['rush_yd']

    data['rush_td_pt'] = score_dict['rush_td']

    data['rec_yd_pt'] = score_dict['rec_yd']

    data['rec_td_pt'] = score_dict['rec_td']

    data['rec_pt'] = score_dict['ppr']

    data['fuml_pt'] = score_dict['fum_lost']

    data['conv_pt'] = score_dict['2pc']

    data['ret_td_pt'] = score_dict['ret_td']

    

    # Use pt per activity columns to generate new columns

    data[label + '_pass_pts'] = ((data['py'] * data['pass_yd_pt']) +

                                 (data['tdp'] * data['pass_td_pt']) +

                                 (data['ints'] * data['int_pt']) +

                                 (data['sk'] * data['sack_pt'])

                                )

    

    data[label + '_rush_pts'] = ((data['ry'] * data['rush_yd_pt']) +

                                 (data['tdr'] * data['rush_td_pt'])

                                )

    

    data[label + '_rec_pts'] = ((data['recy'] * data['rec_yd_pt']) +

                               (data['tdrec'] * data['rec_td_pt']) +

                               (data['rec'] * data['rec_pt'])

                              )

    

    data[label + '_fumbles'] = (data['fuml'] * data['fuml_pt'])

    

    data[label + '_2pc'] = (data['conv'] * data['conv_pt'])

                                         

    data[label + '_ret_pts'] = (data['tdret'] * data['ret_td_pt'])

    

    

    # Generate aggregate scoring columns

    data[label + '_opp_no_fumbles'] = (data[label + '_rush_pts'] +

                                       data[label + '_rec_pts'] +

                                       data[label + '_2pc']

                                      )

    

    data[label + '_pts_ex_ret'] = (data[label + '_pass_pts'] +

                                   data[label + '_rush_pts'] +

                                   data[label + '_rec_pts'] +

                                   data[label + '_fumbles'] +

                                   data[label + '_2pc']

                                  )

    

    data[label + '_total_pts'] = data[label + '_pts_ex_ret'] + data[label + '_ret_pts']

    

    # Drop score_dict columns

    data = data.drop(columns=['pass_yd_pt',

                              'pass_td_pt',

                              'int_pt',

                              'sack_pt',

                              'rush_yd_pt',

                              'rush_td_pt',

                              'rec_yd_pt',

                              'rec_td_pt',

                              'rec_pt',

                              'fuml_pt',

                              'conv_pt',

                              'ret_td_pt'],

                     axis=1

                    )

    

    return data
# Add scoring columns to the offense table

offense = add_scoring_columns(offense, ppr, 'ppr')

offense = add_scoring_columns(offense, reg, 'reg')
def snaps_by_depth_chart_boxplot(game_data, position, color_dict):



    # Define and plot snap information

    game_data = game_data[(game_data['pos1'] == position) & 

                          (game_data['year'] >= 2015) &

                          (game_data['dcp'] >= 1)

                         ]

    

    fig = px.box(game_data,

                 x= 'dcp',

                 y='snp',

                 color_discrete_sequence=[color_dict[position][0]]

                )

    

    # Set up title, size, and hover text

    fig.update_layout(

        title_text= 'Snaps by Depth Chart - ' + position,

        title_x=0.5,

        height=500,

        hovermode='x'

    )



    fig.show()



def depth_chart_snap_hist(game_data, position, color_dict):

    

    game_data = game_data[(game_data['pos1'] == position) & 

                          (game_data['year'] >= 2015) &

                          (game_data['dcp'] >= 1)

                         ]

    

    # Select data to be plotted and set up figure and subplots

    fig, axs = plt.subplots(2, 3, figsize=(16,10), dpi= 80)

    fig.subplots_adjust(top=.90, wspace=.3, hspace=.5)

    fig.suptitle('Snaps by Depth Chart Position - ' + position)

    axs = axs.ravel()

    

    #sns.set_color_codes(color_dict[position])

    

    # Loop through subplots and plot 

    i = 1

    for ax in axs:

        if len(game_data[game_data['dcp'] == i]) > 1:

            data = game_data.loc[game_data['dcp'] == i]

            data = data['snp']

            sns.distplot(data, ax=ax, color=color_dict[position][0])    

            ax.set_xlabel('Snaps per Game')

            ax.set_title('Depth Chart - ' + str(i))

            ax.tick_params(labelsize=13)

        else:

            ax.axis('off')

        i += 1    

    

    fig.show
depth_chart_snap_hist(offense, 'WR', color_dict)
snaps_by_depth_chart_boxplot(offense, 'WR', color_dict)
depth_chart_snap_hist(offense, 'RB', color_dict)
snaps_by_depth_chart_boxplot(offense, 'RB', color_dict)
depth_chart_snap_hist(offense, 'TE', color_dict)
snaps_by_depth_chart_boxplot(offense, 'TE', color_dict)
depth_chart_snap_hist(offense, 'QB', color_dict)
snaps_by_depth_chart_boxplot(offense, 'QB', color_dict)
def points_by_pos_and_x_boxplot(data, title, metric, x, positions, years, color_dict):



    # Define and plot snap information

    data = data[(data['pos1'].isin(positions)) & 

                (data['year'].isin(years)) &

                (data['dcp'] >= 1)

               ]

    

    # Specify color code

    pos_dict = {}

    color_list = []

    for position in positions:

        first_row = data.pos1.eq(position).idxmax()

        pos_dict[position] = first_row



    pos_dict = sorted(pos_dict.items(), key=lambda kv: kv[1])

    

    for item in pos_dict:

        color_list.append(color_dict[item[0]][0])

    

    # Set up boxplot

    fig = px.box(data,

                 x=x,

                 y= metric,

                 color='pos1',

                 color_discrete_sequence=color_list

                )

    

    # Prep title

    title += ' - ' + str(years)

    

    # Set up title, size, and hover text

    fig.update_layout(

        title_text=title,

        title_x=0.5,

        height=625,

        hovermode='x'

    )



    fig.show()
points_by_pos_and_x_boxplot(offense, 

                            title='PPR Points by Position and Depth Chart',

                            metric='ppr_total_pts',

                            x='dcp',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2015,2016,2017,2018,2019], 

                            color_dict=color_dict

                           )
points_by_pos_and_x_boxplot(offense, 

                            title='Regular Points by Position and Depth Chart',

                            metric='reg_total_pts',

                            x='dcp',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2015,2016,2017,2018,2019], 

                            color_dict=color_dict

                           )
def prep_for_correlations(data, drop_cols=[], group_cols=[], avg_cols=[], count_cols=[], reset=False):

    

    # Specify sum columns (all others)

    sum_cols = [x for x in list(data.columns) if ((x not in drop_cols) &

                                                     (x not in count_cols) &

                                                     (x not in group_cols) &

                                                     (x not in avg_cols)

                                                    )

               ]

    # Drop columns

    new_data = data.drop(columns=drop_cols)

    

    # Build aggreations dictionary for groupby

    aggregations = {}

    

    for col in count_cols:

        aggregations[col] = 'count'

    for col in avg_cols:

        aggregations[col] = 'mean'

    for col in sum_cols:

        aggregations[col] = 'sum'



    

    # Group table by by group_cols and apply aggregation functions

    if reset:

        new_data = new_data.groupby(group_cols).agg(aggregations).reset_index()

    else:

        new_data = new_data.groupby(group_cols).agg(aggregations)

    

    return new_data
# Set up columns for prep_for_correlations



# Specify columns to drop

drop_cols = ['player_gid',

             'pstat',

             'gstat',

             'wk',

             'next_gid',

             'player_next_gid',

             'pstat_next_game',

             'gstat_next_game'

            ]



# Specify count columns (games played)

count_cols = ['gid']



# Specify grouping columns

group_cols = ['player_year',

              'player',

              'year',

              'team',

              'pos1',  

             ]



# Specify average columns (depth chart position)

avg_cols = ['dcp']



# Get offense tables grouped by player_year

if not playoffs:

    offense = offense[offense['wk'] <= 17 ]



offense_no_playoffs = offense[offense['wk'] <= 17 ]



full_season_stats = prep_for_correlations(data=offense_no_playoffs,

                                          drop_cols=drop_cols,

                                          group_cols=group_cols, 

                                          avg_cols=avg_cols,

                                          count_cols=count_cols

                                         )

full_season_stats_w_playoffs = prep_for_correlations(data=offense,

                                                     drop_cols=drop_cols, 

                                                     group_cols=group_cols, 

                                                     avg_cols=avg_cols,

                                                     count_cols=count_cols

                                                    )



# Filter for 2nd half of season only and run prep_for_correlations

sh_offense = offense[offense['wk'] > 8]

sh_stats = prep_for_correlations(data=sh_offense,

                                 drop_cols=drop_cols,

                                 group_cols=group_cols,

                                 avg_cols=avg_cols,

                                 count_cols=count_cols

                                )



# Create per game tables

per_game_stats = full_season_stats_w_playoffs.copy(deep=True)

per_game_stats.iloc[:,2:] = per_game_stats.iloc[:,2:].div(per_game_stats['gid'], axis=0)



sh_per_game = sh_stats.copy(deep=True)

sh_per_game.iloc[:,2:] = sh_per_game.iloc[:,2:].div(sh_per_game['gid'], axis=0)
# Specify play table columns and datatypes

play_col_types = {'gid' : int,

                  'pid' : int,

                  'off' : str,

                  'type' : str,

                  'yfog' : float,

                  'zone' : int,

                  'pts' : float,

                  'sk' : int,

                  'ints' : int,

                  'fum' : int

                 }



# Isolate column names

play_cols = play_col_types.keys()



# Import play table

plays = pd.read_csv('../input/nfl-game-season-and-salary-data/PLAY.csv',

                   delimiter=',',

                   usecols=play_cols,

                   dtype=play_col_types

                  )



# Specify pass table columns and datatypes

pass_col_types = {'pid' : int,

                  'psr' : str,

                  'trg' : str,

                  'loc' : str,

                  'yds' : float,

                  'comp' : int

                 }



# Isolate column names

pass_cols = pass_col_types.keys()



# Import pass table

passes = pd.read_csv('../input/nfl-game-season-and-salary-data/PASS.csv',

                   delimiter=',',

                   usecols=pass_cols,

                   dtype=pass_col_types

                  )



# Specify rush table columns and datatypes

rush_col_types = {'pid' : int,

                  'bc' : str,

                  'yds' : float,

                  }



# Isolate column names

rush_cols = rush_col_types.keys()



# Import rush table

rushes = pd.read_csv('../input/nfl-game-season-and-salary-data/RUSH.csv',

                   delimiter=',',

                   usecols=rush_cols,

                   dtype=rush_col_types

                  )



# Specify fumble table columns and datatypes

fumble_col_types = {'pid' : int,

                    'fum' : str,

                    'fuml' : str,

                   }



# Isolate column names

fumble_cols = fumble_col_types.keys()



# Import fumble table

fumble = pd.read_csv('../input/nfl-game-season-and-salary-data/FUMBLE.csv',

                      delimiter=',',

                      usecols=fumble_cols,

                      dtype=fumble_col_types

                     )



# Specify penalty table columns and datatypes

penalty_col_types = {'pid' : int,

                     'act' : str,

                    }



# Isolate column names

penalty_cols = penalty_col_types.keys()



# Import fumble table

penalty = pd.read_csv('../input/nfl-game-season-and-salary-data/PENALTY.csv',

                      delimiter=',',

                      usecols=penalty_cols,

                      dtype=penalty_col_types

                     )



# Specify kickoff table columns and datatypes

conv_col_types = {'pid' : int,

                  'type' : str,

                  'bc' : str,

                  'psr' : str,

                  'trg' : str,

                  'conv' : float,

                 }



# Isolate column names

conv_cols = conv_col_types.keys()



# Import kickoff table

conv = pd.read_csv('../input/nfl-game-season-and-salary-data/CONV.csv',

                   delimiter=',',

                   usecols=conv_cols,

                   dtype=conv_col_types

                  )



# Specify kickoff table columns and datatypes

koff_col_types = {'pid' : int,

                  'kr' : str,

                 }



# Isolate column names

koff_cols = koff_col_types.keys()



# Import kickoff table

koff = pd.read_csv('../input/nfl-game-season-and-salary-data/KOFF.csv',

                   delimiter=',',

                   usecols=koff_cols,

                   dtype=koff_col_types

                  )



# Specify punt table columns and datatypes

punt_col_types = {'pid' : int,

                  'pr' : str,

                 }



# Isolate column names

punt_cols = punt_col_types.keys()



# Import kickoff table

punt = pd.read_csv('../input/nfl-game-season-and-salary-data/PUNT.csv',

                   delimiter=',',

                   usecols=punt_cols,

                   dtype=punt_col_types

                  )
# Merge season (year) to plays from schedule table

sched_cols = ['gid', 'seas']

plays = pd.merge(plays, 

                 schedule[sched_cols],

                 left_on='gid',

                 right_on='gid',

                 how='left',

                 suffixes=('', '_sched')

                )



# Merge passes table to plays

plays = pd.merge(plays, 

                 passes,

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_pass')

                ).rename(columns={'yds' : 'py',

                                  'comp' : 'rec'

                                 }

                        )



# Merge rushes table to plays

plays = pd.merge(plays, 

                 rushes,

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_rush')

                ).rename(columns={'yds' : 'ry'})



# Merge fumble table to plays

plays = pd.merge(plays, 

                 fumble,

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_fum')

                ).rename(columns={'fum_fum' : 'fumbler'})



# Merge penalty table to plays

plays = pd.merge(plays, 

                 penalty,

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_pen')

                ).rename(columns={'act': 'penalty'})



# Merge conversion table to plays

conv_cols = ['pid', 'type', 'bc', 'psr', 'trg', 'conv']

plays = pd.merge(plays, 

                 conv[conv_cols],

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_conv')

                )



# Merge kickoff table (returner column) to plays

koff_cols = ['pid', 'kr']

plays = pd.merge(plays, 

                 koff[koff_cols],

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_koff')

                )



# Merge punt table (returner column) to plays

punt_cols = ['pid', 'pr']

plays = pd.merge(plays, 

                 punt[punt_cols],

                 left_on='pid',

                 right_on='pid',

                 how='left',

                 suffixes=('', '_punt')

                )
# Replace NaN for non-pass plays in loc field and drop other NaN rows

mask = (plays['type'] != 'PASS')

plays['loc'][mask] = '-'



plays = plays.dropna(subset=['loc'])



# Fill NA's in py, ry, rec, and conv columns

fill_values = {'py': 0.0, 

               'ry': 0.0,

               'rec' : 0.0,

               'conv' : 0.0

              }

plays = plays.fillna(value=fill_values)



# Add recy column (copy py column)

plays['recy'] = plays['py']



# Add 1 to rec column for successful passing conversions

plays['rec'] = plays.apply(lambda row:

                           row.rec + 1 

                           if ((row.conv == 6) &

                               (row.type_conv == 'PASS')

                              )

                           else row.rec,

                           axis=1

                          )



# Convert fuml from yes/no to 1/0

plays['fuml'] = pd.Series(np.where(plays.fuml.values == 'Y', 1, 0), plays.index)



# Replace null values in penalty column with "-"

plays = plays.fillna(value={'penalty' : '-'})



# Drop plays where penalty was accepted

mask = ((plays['penalty'] != 'A') &

        (plays['penalty'] != 'O')

       )

plays = plays.loc[mask]



# Replace bc_conv, psr_conv, trg_conv, type_conv, bc, psr, trg values with blank ('')

plays = plays.fillna(value={'bc' : '',

                            'psr' : '',

                            'trg' : '',

                            'bc_conv' : '',

                            'psr_conv' : '',

                            'trg_conv' : '',

                            'type_conv' : ''

                           }

                    )



# Combine bc with bc_conv, psr with psr_conv, trg with trg_conv

plays['bc'] = plays['bc'] + plays['bc_conv']

plays['psr'] = plays['psr'] + plays['psr_conv']

plays['trg'] = plays['trg'] + plays['trg_conv']



# Drop skill player '_conv' columns

plays = plays.drop(columns=['bc_conv', 'psr_conv', 'trg_conv'])



# Combine kr, pr, trg, bc to a skill player column (sp)

plays['sp'] = plays['kr'] + plays['pr'] + plays['trg'] + plays['bc']



# Replace kr, pr, trg, bc values with blank ('')

plays = plays.fillna(value={'kr' : '',

                            'pr' : '',

                            'trg' : '',

                            'bc' : ''

                           }

                    )



# Combine kr, pr, trg, bc to a skill player column (sp)

plays['sp'] = plays['kr'] + plays['pr'] + plays['trg'] + plays['bc']



# Recalculate zone column

plays['zone'] = plays['yfog'].apply(lambda x:

                                    '0-20' if x <= 20.0

                                    else ('20-40' if x <= 40.0

                                          else ('40-60' if x <= 60.0

                                                else ('60-80' if x <= 80.0

                                                      else ('80-90' if x <= 90.0

                                                            else '90-100'

                                                           )

                                                     )

                                               )

                                         )

                                   )



# Drop L, M, R, and NL pass plays (small percentage of total list)

mask = ((plays['loc'] != 'L') &

        (plays['loc'] != 'M') &

        (plays['loc'] != 'NL') &

        (plays['loc'] != 'R')

       )

        

plays = plays.loc[mask]



# Drop second letter in loc column, leaving only S (short) or D (deep)

plays['loc'] = plays['loc'].apply(lambda x: x[0])



# Add play type and zone column (type_zone)

plays['type_zone'] = plays['type'] + '_' + plays['loc'] + '_' + plays['zone']

plays['type_zone'] = plays['type_zone'].str.replace('_-_','_')
# Add tdp, tdr, tdrec, tdret, conv fields 

plays['tdp'] = plays.apply(lambda row:

                           1 

                           if ((row.pts >= 6) &

                               (row.type == 'PASS')

                              )

                           else 0,

                           axis=1

                          )



plays['tdrec'] = plays.apply(lambda row:

                           1 

                           if ((row.pts >= 6) &

                               (row.type == 'PASS')

                              )

                           else 0,

                           axis=1

                          )



plays['tdr'] = plays.apply(lambda row:

                           1 

                           if ((row.pts >= 6) &

                               (row.type == 'RUSH')

                              )

                           else 0,

                           axis=1

                          )



plays['tdret'] = plays.apply(lambda row:

                           1 

                           if ((row.pts >= 6) &

                               ((row.type == 'KOFF') |

                                (row.type == 'ONSD') |

                                (row.type == 'PUNT')

                               )

                              )

                           else 0,

                           axis=1

                          )                   
# Add scoring columns to plays table

plays = add_scoring_columns(plays, ppr, 'ppr')

plays = add_scoring_columns(plays, reg, 'reg')



# Create second half only plays table

sh_plays = plays[plays['gid'].isin(sh_offense['gid'])]



# Create table without playoffs

plays_no_playoffs = plays[plays['gid'].isin(offense_no_playoffs['gid'])]
def play_type_zone_avg_points(play_data, pts_metric, ppr=True):

    # Filter for CONV, PASS, and RUSH plays 2015 or later

    data = play_data[(play_data['seas'] >= 2015) &

                     ((play_data['type'] == 'CONV') |

                      (play_data['type'] == 'PASS') |

                      (play_data['type'] == 'RUSH')

                     )

                    ]

    

    # Pivot to show points/opp by type_zone and season

    data_pivot = data.pivot_table(values=pts_metric, 

                                  index='type_zone',

                                  columns='seas',

                                  aggfunc='mean',

                                  dropna=True,

                                  fill_value=0,

                                  margins=True

                                 )

    

    # Remove rows that are all zero

    data_pivot = data_pivot.loc[(data_pivot!=0).any(axis=1)]

    

    # PPR title string

    ppr_string = 'PPR Scoring' if ppr else 'Regular Scoring'

    

    fig = plt.figure(figsize=(16,10))

    sns.heatmap(data_pivot, fmt='g', cmap='Blues', annot=True, annot_kws={"size": 15})

    plt.yticks(rotation=0)

    plt.xlabel('Season', labelpad=20)

    plt.ylabel('Play Type + Field Location ')

    plt.title('Average Points per Opportunity (Ex. Fumbles) - ' + ppr_string, pad=30, size=20)

    

    data_pivot_weighted  = data_pivot / data_pivot.loc['All']

    

    return data_pivot, data_pivot_weighted
ppr_type_zone_pts, ppr_type_zone_pts_weighted = play_type_zone_avg_points(plays, 'ppr_opp_no_fumbles', True)
ppr_type_zone_pts_weighted
reg_type_zone_pts, reg_type_zone_pts_weighted = play_type_zone_avg_points(plays, 'reg_opp_no_fumbles', False)
reg_type_zone_pts_weighted
# Create PPR scoring merge column 

ppr_opp_weights = ppr_type_zone_pts_weighted.reset_index().rename_axis(None, axis=1)



ppr_opp_weights = pd.melt(ppr_opp_weights, 

                             id_vars=['type_zone'],

                             value_vars=[2015, 2016, 2017, 2018, 2019],

                             var_name='seas',

                             value_name='weight'

                            )



ppr_opp_weights['type_zone_year'] = ppr_opp_weights['type_zone'] + '_' + ppr_opp_weights['seas'].astype(str)



# Create regular scoring merge column

reg_opp_weights = reg_type_zone_pts_weighted.reset_index().rename_axis(None, axis=1)



reg_opp_weights = pd.melt(reg_opp_weights, 

                             id_vars=['type_zone'],

                             value_vars=[2015, 2016, 2017, 2018, 2019],

                             var_name='seas',

                             value_name='weight'

                            )



reg_opp_weights['type_zone_year'] = reg_opp_weights['type_zone'] + '_' + reg_opp_weights['seas'].astype(str)
# ADD OPPS AND WEIGHTED OPPS (PPR AND REGULAR) TO FULL SEASON, PER GAME, AND OFFENSE TABLES 



# Get list of type_zone combos for weighted opps

type_zone_list = list(ppr_type_zone_pts.index)[:-1]



# Filter plays table to relevant type_zones and 2015 onward

type_zone_plays = plays[plays['type_zone'].isin(type_zone_list)]

type_zone_plays = type_zone_plays[type_zone_plays['seas'] >= 2015]

# Second half only

sh_type_zone = sh_plays[sh_plays['type_zone'].isin(type_zone_list)]

sh_type_zone = sh_type_zone[sh_type_zone['seas'] >= 2015]

# No playoffs

type_zone_plays_no_playoffs = plays_no_playoffs[plays_no_playoffs['type_zone'].isin(type_zone_list)]

type_zone_plays_no_playoffs = type_zone_plays_no_playoffs[type_zone_plays_no_playoffs['seas'] >= 2015]



# Group by player, season, type_zone to show total opps

keep_cols = ['sp', 'type_zone', 'seas', 'pid', 'off']

player_opps_grouped_w_playoffs = type_zone_plays[keep_cols].groupby(['sp','seas','off','type_zone']).count().reset_index().rename(columns={'pid' : 'opps'})

player_opps_grouped = type_zone_plays_no_playoffs[keep_cols].groupby(['sp','seas','off','type_zone']).count().reset_index().rename(columns={'pid' : 'opps'})

# Second half only

sh_player_opps = sh_type_zone[keep_cols].groupby(['sp','seas','off','type_zone']).count().reset_index().rename(columns={'pid' : 'opps'})

# Individual games

keep_cols = ['sp', 'type_zone','seas', 'gid', 'pid']

opps_by_game = type_zone_plays[keep_cols].groupby(['sp', 'seas', 'gid', 'type_zone']).count().reset_index().rename(columns={'pid' : 'opps'})



# Add type_zone_year as lookup columnn for merge

player_opps_grouped_w_playoffs['type_zone_year'] = player_opps_grouped_w_playoffs['type_zone'] + '_' + player_opps_grouped_w_playoffs['seas'].astype(str)

player_opps_grouped['type_zone_year'] = player_opps_grouped['type_zone'] + '_' + player_opps_grouped['seas'].astype(str)

# Second half only

sh_player_opps['type_zone_year'] = sh_player_opps['type_zone'] + '_' + sh_player_opps['seas'].astype(str) 

# Individual games

opps_by_game['type_zone_year'] = opps_by_game['type_zone'] + '_' + opps_by_game['seas'].astype(str)



# Merge weighting column from PPR opp weights table

weight_cols = ['type_zone_year', 'weight']

player_opps_grouped_w_playoffs = pd.merge(player_opps_grouped_w_playoffs, 

                               ppr_opp_weights[weight_cols],

                               left_on='type_zone_year',

                               right_on='type_zone_year',

                               how='left',

                               suffixes=('', '_ppr')

                              ).rename(columns={'weight' : 'weight_ppr'})

player_opps_grouped = pd.merge(player_opps_grouped, 

                               ppr_opp_weights[weight_cols],

                               left_on='type_zone_year',

                               right_on='type_zone_year',

                               how='left',

                               suffixes=('', '_ppr')

                              ).rename(columns={'weight' : 'weight_ppr'})

# Second half only

sh_player_opps = pd.merge(sh_player_opps, 

                          ppr_opp_weights[weight_cols],

                          left_on='type_zone_year',

                          right_on='type_zone_year',

                          how='left',

                          suffixes=('', '_ppr')

                         ).rename(columns={'weight' : 'weight_ppr'})

# Individual games

opps_by_game = pd.merge(opps_by_game, 

                        ppr_opp_weights[weight_cols],

                        left_on='type_zone_year',

                        right_on='type_zone_year',

                        how='left',

                        suffixes=('', '_ppr')

                       ).rename(columns={'weight' : 'weight_ppr'})





# Merge weighting column from REGULAR scoring opp weights table

player_opps_grouped_w_playoffs = pd.merge(player_opps_grouped_w_playoffs, 

                               reg_opp_weights[weight_cols],

                               left_on='type_zone_year',

                               right_on='type_zone_year',

                               how='left',

                               suffixes=('', '_reg')

                              ).rename(columns={'weight' : 'weight_reg'})

player_opps_grouped = pd.merge(player_opps_grouped, 

                               reg_opp_weights[weight_cols],

                               left_on='type_zone_year',

                               right_on='type_zone_year',

                               how='left',

                               suffixes=('', '_reg')

                              ).rename(columns={'weight' : 'weight_reg'})

# Second half only

sh_player_opps = pd.merge(sh_player_opps, 

                          reg_opp_weights[weight_cols],

                          left_on='type_zone_year',

                          right_on='type_zone_year',

                          how='left',

                          suffixes=('', '_reg')

                         ).rename(columns={'weight' : 'weight_reg'})

# Individual games

opps_by_game = pd.merge(opps_by_game, 

                        reg_opp_weights[weight_cols],

                        left_on='type_zone_year',

                        right_on='type_zone_year',

                        how='left',

                        suffixes=('', '_reg')

                       ).rename(columns={'weight' : 'weight_reg'})





# Drop type_zone_year (only used for merges)

player_opps_grouped_w_playoffs = player_opps_grouped_w_playoffs.drop(columns=['type_zone_year'])

player_opps_grouped = player_opps_grouped.drop(columns=['type_zone_year'])

# Second half only

sh_player_opps = sh_player_opps.drop(columns=['type_zone_year'])

# Individual games

opps_by_game = opps_by_game.drop(columns=['type_zone_year'])





# Add weighted opps columns (ppr and reg)

player_opps_grouped_w_playoffs['ppr_weighted_opps'] = player_opps_grouped_w_playoffs['opps'] * player_opps_grouped_w_playoffs['weight_ppr']

player_opps_grouped_w_playoffs['reg_weighted_opps'] = player_opps_grouped_w_playoffs['opps'] * player_opps_grouped_w_playoffs['weight_reg']

player_opps_grouped['ppr_weighted_opps'] = player_opps_grouped['opps'] * player_opps_grouped['weight_ppr']

player_opps_grouped['reg_weighted_opps'] = player_opps_grouped['opps'] * player_opps_grouped['weight_reg']

# Second half only

sh_player_opps['ppr_weighted_opps'] = sh_player_opps['opps'] * sh_player_opps['weight_ppr']

sh_player_opps['reg_weighted_opps'] = sh_player_opps['opps'] * sh_player_opps['weight_reg']

# Individual games

opps_by_game['ppr_weighted_opps'] = opps_by_game['opps'] * opps_by_game['weight_ppr']

opps_by_game['reg_weighted_opps'] = opps_by_game['opps'] * opps_by_game['weight_reg']





# Add merge columns (player_year / player_gid)

player_opps_grouped_w_playoffs['player_year'] = player_opps_grouped_w_playoffs['sp'] + '-' + player_opps_grouped_w_playoffs['seas'].astype(str) 

player_opps_grouped_w_playoffs['player_year_team'] = player_opps_grouped_w_playoffs['player_year'] + '-' + player_opps_grouped_w_playoffs['off'] 

player_opps_grouped['player_year'] = player_opps_grouped['sp'] + '-' + player_opps_grouped['seas'].astype(str) 

player_opps_grouped['player_year_team'] = player_opps_grouped['player_year'] + '-' + player_opps_grouped['off']

# Second half only

sh_player_opps['player_year'] = sh_player_opps['sp'] + '-' + sh_player_opps['seas'].astype(str) 

sh_player_opps['player_year_team'] = sh_player_opps['player_year'] + '-' + sh_player_opps['off']

# Individual games

opps_by_game['player_gid'] = opps_by_game['sp'] + '-' + opps_by_game['gid'].astype(str) 





# Create weighted opps table grouped to merge to full season, per game, and offense tables

opps_by_player_w_playoffs = player_opps_grouped_w_playoffs[['player_year_team', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']].groupby('player_year_team').sum().reset_index()

opps_by_player = player_opps_grouped[['player_year_team', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']].groupby('player_year_team').sum().reset_index()

# Second half only

sh_opps_by_player = sh_player_opps[['player_year_team', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']].groupby('player_year_team').sum().reset_index()

# Individual games

opps_by_player_gid = opps_by_game[['player_gid', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']].groupby('player_gid').sum().reset_index()

    



# Merge weighted opps columns to full season stats table

full_season_stats = full_season_stats.reset_index()

full_season_stats['player_year_team'] = full_season_stats['player_year'] + '-' + full_season_stats['team']

opps_cols = ['player_year_team', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']

full_season_stats = pd.merge(full_season_stats,

                             opps_by_player[opps_cols],

                             left_on='player_year_team',

                             right_on='player_year_team',

                             how='left',

                             suffixes=('', '_opps')

                            )



full_season_stats_w_playoffs = full_season_stats_w_playoffs.reset_index()

full_season_stats_w_playoffs['player_year_team'] = full_season_stats_w_playoffs['player_year'] + '-' + full_season_stats_w_playoffs['team']

full_season_stats_w_playoffs = pd.merge(full_season_stats_w_playoffs,

                             opps_by_player_w_playoffs[opps_cols],

                             left_on='player_year_team',

                             right_on='player_year_team',

                             how='left',

                             suffixes=('', '_opps')

                            )

# Merge weighted opps columns to per game stats table

per_game_stats = per_game_stats.reset_index()

per_game_stats['player_year_team'] = per_game_stats['player_year'] + '-' + per_game_stats['team']

opps_cols = ['player_year_team', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']

per_game_stats = pd.merge(per_game_stats,

                          opps_by_player_w_playoffs[opps_cols],

                          left_on='player_year_team',

                          right_on='player_year_team',

                          how='left',

                          suffixes=('', '_opps')

                         )

# Second half only

sh_per_game = sh_per_game.reset_index()

sh_per_game['player_year_team'] = sh_per_game['player_year'] + '-' + sh_per_game['team']

sh_per_game = pd.merge(sh_per_game,

                       sh_opps_by_player[opps_cols],

                       left_on='player_year_team',

                       right_on='player_year_team',

                       how='left',

                       suffixes=('', '_opps')

                      )

# Individual games

offense = offense.reset_index()

opps_cols = ['player_gid', 'opps', 'ppr_weighted_opps', 'reg_weighted_opps']

offense = pd.merge(offense,

                   opps_by_player_gid[opps_cols],

                   left_on='player_gid',

                   right_on='player_gid',

                   how='left',

                   suffixes=('', '_opps')

                  )





# Divide total opps and weighted opps columns by games played (gid) in per game table

per_game_stats['opps'] = per_game_stats['opps'] / per_game_stats['gid'] 

per_game_stats['ppr_weighted_opps'] = per_game_stats['ppr_weighted_opps'] / per_game_stats['gid']

per_game_stats['reg_weighted_opps'] = per_game_stats['reg_weighted_opps'] / per_game_stats['gid']

# Second half only

sh_per_game['opps'] = sh_per_game['opps'] / sh_per_game['gid'] 

sh_per_game['ppr_weighted_opps'] = sh_per_game['ppr_weighted_opps'] / sh_per_game['gid']

sh_per_game['reg_weighted_opps'] = sh_per_game['reg_weighted_opps'] / sh_per_game['gid'] 
def stats_correlogram(data, metrics, num_y, per_game, ppr, prior_year, second_half, year_min=2015, year_max=2019, positions=[], min_games=0):

    

    # Filter the data for specified metrics, years, positions, minimum games

    data = data[(data['year'] >= year_min) & 

                (data['year'] <= year_max) &

                (data['gid'] >= min_games)

               ]

    

    if prior_year:

        data = data[data['gid_py'] >= min_games]

    

    if len(positions) > 0:

        data = data[data['pos1'].isin(positions)]

    

    data = data[metrics]

   

    # Prep title components

    if per_game:

        cgram_string = 'Per Game'

    elif second_half:

        cgram_string = 'Second Half (Per Game)'

    else:

        cgram_string = 'Full Season'

    

    cgram_string += (' - PY stats vs CY pts'  if prior_year else ' - CY stats vs CY pts')

    cgram_string += (' - PPR Scoring\n' if ppr else ' - Reg Scoring\n')

    

    

    if len(positions) > 0:

        position_string = ', '.join([str(x) for x in positions])

    else:

        position_string = 'All Positions'



    if year_min == year_max:

        year_string = str(year_min)

    else:

        year_string = str(year_min) + '-' + str(year_max)

    

    title = (cgram_string +

             year_string +

             ' - ' + position_string +

             ' - Min Games: ' + str(min_games)

            )



    # Create correlation column

    corr_table = data.corr()

    

    if ppr:

        corr_table = corr_table[['ppr_total_pts']]

    else:

        corr_table = corr_table[['reg_total_pts']]

        

             



    # Plot

    colors = 'RdYlGn' if ppr else 'RdYlBu'

    

    plt.figure(figsize=(12,7), dpi= 80)

    sns.heatmap(data.corr().head(num_y), 

                xticklabels=data.corr().columns, 

                yticklabels=list(data.corr().columns)[0:num_y], 

                cmap=colors,

                vmin=-1.0,

                vmax=1.0,

                center=0, 

                annot=True)



    # Decorations

    plt.title(title, fontsize=22, pad=12)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=12)

    plt.yticks(rotation=0)

    plt.show()

    

    return corr_table
def corr_table_gradient(full_corr, per_game_corr, ppr, sh_corr=pd.DataFrame()):

    # Create list of correlation columns

    corr_list = [full_corr, per_game_corr]

    if not sh_corr.empty:

        corr_list.append(sh_corr)

    

    # Create list of column names

    corr_cols = ['full_season', 'per_game']

    if not sh_corr.empty:

        corr_cols.append('2nd_half_per_game')

    

    # Create table and rename columns

    corr_comparison = pd.concat(corr_list, axis=1)

    corr_comparison.columns = corr_cols

    

    # Filter for relevant rows

    corr_comparison = corr_comparison.iloc[0:11]

    

    # Determine color map

    cm = 'RdYlGn' if ppr else 'RdYlBu'

    

    # Set high and low float parameters

    low = (1.0 - abs(corr_comparison.values.min())) / 2.0

    high = (1.0 - abs(corr_comparison.values.max())) / 2.0

    

    # Stylize table (gradient and column width)

    corr_comparison = corr_comparison.style.background_gradient(cmap=cm,

                                                                low=low,

                                                                high=high

                                                               ).set_properties(**{'width': '125px'})

    return corr_comparison

    

        
# Establish ppr metrics for correlograms

ppr_metrics = ['dcp',

               'ra',

               'ry',

               'tdr',

               'trg',

               'rec',

               'recy',

               'tdrec',

               'snp',

               'opps',

               'ppr_weighted_opps',

               'ppr_rush_pts',

               'ppr_rec_pts',

               'ppr_opp_no_fumbles',

               'ppr_pts_ex_ret',

               'ppr_total_pts'

              ]



# Establish regular scoring metrics for correlograms

reg_metrics = ['dcp',

               'ra',

               'ry',

               'tdr',

               'trg',

               'rec',

               'recy',

               'tdrec',

               'snp',

               'opps',

               'reg_weighted_opps',

               'reg_rush_pts',

               'reg_rec_pts',

               'reg_opp_no_fumbles',

               'reg_pts_ex_ret',

               'reg_total_pts'

              ]



# Establish regular scoring metrics for correlograms

qb_metrics = ['dcp',

              'pa',

              'pc',

              'py',

              'tdp',

              'ints',

              'sk',

              'ra',

              'ry',

              'tdr',

              'snp',

              'ppr_pass_pts',

              'ppr_rush_pts',

              'ppr_opp_no_fumbles',

              'ppr_pts_ex_ret',

              'ppr_total_pts'

             ]
full_corr = stats_correlogram(full_season_stats,

                              metrics=qb_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['QB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=qb_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['QB'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['WR'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['WR'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['WR'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['WR'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['TE'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['TE'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB','WR','TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB','WR','TE'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True)

corr_table
full_corr = stats_correlogram(full_season_stats,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=False,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB','WR','TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_stats,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=False,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB','WR','TE'],

                                  min_games=4

                                 )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False)

corr_table
def py_stats_vs_cy_pts(data, scoring_labels, cy_pts_data=pd.DataFrame()):

    # Define columns to keep from current year data (points, player data)

    cy_cols = ['player_year',

               'player',

               'year',

               'team',

               'pos1',

               'gid',

              ]

    

    for label in scoring_labels:

        cy_cols += [label + '_pass_pts',

                    label + '_rush_pts',

                    label+ '_rec_pts',

                    label + '_fumbles',

                    label + '_2pc',

                    label + '_ret_pts',

                    label + '_opp_no_fumbles',

                    label + '_pts_ex_ret',

                    label + '_total_pts'

                   ]

    

    # Copy cy_cols to new table

    if cy_pts_data.empty:

        py_vs_cy = data[cy_cols]

    else:

        py_vs_cy = cy_pts_data[cy_cols]

    

    # Add prior_year and player_prior_year columns

    if cy_pts_data.empty:

        py_vs_cy['prior_year'] = py_vs_cy['year'] - 1

        py_vs_cy['player_prior_year'] = py_vs_cy['player'] + '-' + py_vs_cy['prior_year'].map(str)

    else:

        data_copy = data.copy(deep=True)

        data_copy['prior_year'] = data_copy['year'] - 1

        data_copy['player_prior_year'] = data_copy['player'] + '-' + data_copy['prior_year'].map(str)

    

    # Define prior year columns (metrics, player data)

    py_cols = ['player_year',

               'team',

               'gid',

               'pa',

               'dcp',

               'pc', 

               'py',

               'ints',

               'sk',

               'tdp',

               'ra', 

               'ry',

               'tdr',

               'trg',

               'rec',

               'recy',

               'tdrec',

               'ret',

               'rety',

               'tdret',

               'fuml',

               'conv',

               'snp',

               'opps',

               'ppr_weighted_opps',

               'reg_weighted_opps'

              ]

    

    if cy_pts_data.empty:

        pass

    else:

        py_cols.append('player_prior_year')

    

    # Merge prior year metrics to current year points

    if cy_pts_data.empty:

        py_vs_cy = pd.merge(py_vs_cy,

                            data[py_cols],

                            left_on='player_prior_year',

                            right_on='player_year',

                            how='left',

                            suffixes=('', '_py')

                           )

    else:

        py_vs_cy = pd.merge(data_copy[py_cols],

                            py_vs_cy,

                            left_on='player_prior_year',

                            right_on='player_year',

                            how='left',

                            suffixes=('_py', '')

                           )

    

    # Drop unnecessary columns and all rows with NaN values

    py_vs_cy = py_vs_cy.drop(columns=['player_year_py', 'player_prior_year'])

    py_vs_cy = py_vs_cy.dropna(axis=0)

    

    # Reorder the columns

    py_vs_cy_cols = list(dict.fromkeys(cy_cols[0:5] + py_cols + cy_cols[6:]))

    

    if cy_pts_data.empty: 

        pass

    else:

        py_vs_cy_cols.remove('player_prior_year')

        



    py_vs_cy_cols.insert(6, 'gid_py')

    py_vs_cy_cols.insert(4, 'team_py')

    

    py_vs_cy = py_vs_cy[py_vs_cy_cols]

    

    if cy_pts_data.empty: 

        pass

    else:

        py_vs_cy['year'] = py_vs_cy['year'] + 1

    

    return py_vs_cy
# Create table with prior year stats and current year scoring columns

full_season_py_vs_cy = py_stats_vs_cy_pts(full_season_stats, ['ppr', 'reg'])

per_game_py_vs_cy = py_stats_vs_cy_pts(per_game_stats, ['ppr', 'reg'])

# Second half only

per_game_sh_py_vs_cy = py_stats_vs_cy_pts(sh_per_game, ['ppr', 'reg'], per_game_stats)

per_game_sh_py_vs_cy = per_game_sh_py_vs_cy.astype({'year': int})
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=qb_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['QB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=qb_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['QB'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=qb_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=True,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['QB'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=ppr_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=True,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['RB'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=reg_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=False,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['RB'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['WR'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['WR'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=ppr_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=True,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['WR'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['WR'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['WR'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=reg_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=False,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['WR'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['TE'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=ppr_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=True,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['TE'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['TE'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=reg_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=False,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['TE'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=ppr_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=True,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB','WR','TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=ppr_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=True,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB','WR','TE'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=ppr_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=True,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['RB','WR','TE'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=True, sh_corr=sh_corr)

corr_table
full_corr = stats_correlogram(full_season_py_vs_cy,

                              metrics=reg_metrics,

                              num_y=11,

                              per_game=False,

                              ppr=False,

                              prior_year=True,

                              second_half=False,

                              year_min=2015,

                              year_max=2019,

                              positions=['RB','WR','TE'],

                              min_games=4

                             )
per_game_corr = stats_correlogram(per_game_py_vs_cy,

                                  metrics=reg_metrics,

                                  num_y=11,

                                  per_game=True,

                                  ppr=False,

                                  prior_year=True,

                                  second_half=False,

                                  year_min=2015,

                                  year_max=2019,

                                  positions=['RB','WR','TE'],

                                  min_games=4

                                 )
sh_corr = stats_correlogram(per_game_sh_py_vs_cy,

                            metrics=reg_metrics,

                            num_y=11,

                            per_game=False,

                            ppr=False,

                            prior_year=True,

                            second_half=True,

                            year_min=2015,

                            year_max=2019,

                            positions=['RB','WR','TE'],

                            min_games=4

                           )
corr_table = corr_table_gradient(full_corr, per_game_corr, ppr=False, sh_corr=sh_corr)

corr_table
# Filter offense table for relevant correlation columns

past_games = offense[['player',

                      'gid',

                      'pa',

                      'pc',

                      'py',

                      'tdp',

                      'sk',

                      'ints',

                      'ra',

                      'ry',

                      'tdr',

                      'trg',

                      'rec',

                      'recy',

                      'tdrec',

                      'ret',

                      'rety',

                      'tdret',

                      'fuml',

                      'conv',

                      'snp',

                      'dcp',

                      'opps',

                      'player_gid',

                      'player_year',                    

                      'ppr_weighted_opps',

                      'reg_weighted_opps'

                     ]

                    ]



# Convert past games metrics to rolling averages

past_games = past_games.sort_values(by=['gid'])

past_games.set_index(['player', 'player_gid'], inplace=True)



# Past 10 games

past_10 = past_games.groupby(level=0, 

                            group_keys=False

                           ).rolling(window=10,

                                     min_periods=1

                                    ).mean()



# Past 5 games

past_5 = past_games.groupby(level=0, 

                            group_keys=False

                           ).rolling(window=5,

                                     min_periods=1

                                    ).mean()



# Past 3 games

past_3 = past_games.groupby(level=0, 

                            group_keys=False

                           ).rolling(window=3,

                                     min_periods=1

                                    ).mean()
# Merge offense table scoring columns to past games tables (3, 5, 10), drop NaN rows

offense_cols = ['player_gid',

                'player',

                'year',

                'team',

                'pos1',

                'ppr_pass_pts',

                'ppr_rush_pts',

                'ppr_rec_pts',

                'ppr_opp_no_fumbles',

                'ppr_pts_ex_ret',

                'ppr_total_pts',

                'reg_rush_pts',

                'reg_rec_pts',

                'reg_opp_no_fumbles',

                'reg_pts_ex_ret',

                'reg_total_pts'

               ]



past_10 = pd.merge(past_10,

                   offense[offense_cols],

                   left_on='player_gid',

                   right_on='player_gid',

                   how='left',

                   suffixes=('', 'past')

                  ).dropna(axis=0)



past_5 = pd.merge(past_5,

                   offense[offense_cols],

                   left_on='player_gid',

                   right_on='player_gid',

                   how='left',

                   suffixes=('', 'past')

                  ).dropna(axis=0)



past_3 = pd.merge(past_3,

                   offense[offense_cols],

                   left_on='player_gid',

                   right_on='player_gid',

                   how='left',

                   suffixes=('', 'past')

                  ).dropna(axis=0)
def past_x_games_corr(data, metrics, ppr, year_min=2015, year_max=2019, positions=[], min_games=0):

    

    # Filter the data for specified metrics, years, positions, minimum games

    data = data[(data['year'] >= year_min) & 

                (data['year'] <= year_max) &

                (data['gid'] >= min_games)

               ]

    

    if len(positions) > 0:

        data = data[data['pos1'].isin(positions)]

    

    data = data[metrics]



    # Create correlation column

    corr_table = data.corr()

    

    if ppr:

        corr_table = corr_table[['ppr_total_pts']]

    else:

        corr_table = corr_table[['reg_total_pts']]      

  

    return corr_table
def corr_table_gradient(corr_1, corr_2, ppr, column_names, corr_3=pd.DataFrame()):

    

    # Create list of correlation columns

    corr_list = [corr_1, corr_2]

    if not sh_corr.empty:

        corr_list.append(corr_3)

 

    # Create table and rename columns

    corr_comparison = pd.concat(corr_list, axis=1)

    corr_comparison.columns = column_names

    

    # Filter for relevant rows

    corr_comparison = corr_comparison.iloc[0:11]

    

    # Determine color map

    cm = 'RdYlGn' if ppr else 'RdYlBu'

    

    # Set high and low float parameters

    low = (1.0 - abs(corr_comparison.values.min())) / 2.0

    high = (1.0 - abs(corr_comparison.values.max())) / 2.0

    

    # Stylize table (gradient and column width)

    corr_comparison = corr_comparison.style.background_gradient(cmap=cm,

                                                                low=low,

                                                                high=high,

                                                                axis=None

                                                               ).set_properties(**{'width': '125px'})

    return corr_comparison
def corr_3_5_10(positions, qb, ppr, year_min, year_max):

    

    if qb:

        metrics = qb_metrics 

    elif ppr:

        metrics = ppr_metrics 

    else:

        metrics = reg_metrics

    

    past_10_corr = past_x_games_corr(past_10, 

                                     metrics,

                                     ppr=ppr,

                                     year_min=year_min,

                                     year_max=year_max,

                                     positions=positions

                                    )



    past_5_corr = past_x_games_corr(past_5, 

                                     metrics,

                                     ppr=ppr,

                                     year_min=year_min,

                                     year_max=year_max,

                                     positions=positions

                                    )



    past_3_corr = past_x_games_corr(past_3, 

                                     metrics,

                                     ppr=ppr,

                                     year_min=year_min,

                                     year_max=year_max,

                                     positions=positions

                                    )



    corr_comp = corr_table_gradient(past_3_corr,

                                    past_5_corr, 

                                    ppr, 

                                    ['past_3','past 5','past 10'],

                                    past_10_corr

                                   )

    

    return corr_comp
qb_ppr = corr_3_5_10(positions=['QB'], 

                     qb=True,

                     ppr=True, 

                     year_min=2015,

                     year_max=2019

                    )



qb_ppr
rb_ppr = corr_3_5_10(positions=['RB'], 

                     qb=False,

                     ppr=True, 

                     year_min=2015,

                     year_max=2019

                    )



rb_ppr
rb_reg = corr_3_5_10(positions=['RB'], 

                     qb=False,

                     ppr=False, 

                     year_min=2015,

                     year_max=2019

                    )



rb_reg
wr_ppr = corr_3_5_10(positions=['WR'], 

                     qb=False,

                     ppr=True, 

                     year_min=2015,

                     year_max=2019

                    )



wr_ppr
wr_reg = corr_3_5_10(positions=['WR'], 

                     qb=False,

                     ppr=False, 

                     year_min=2015,

                     year_max=2019

                    )



wr_reg
te_ppr = corr_3_5_10(positions=['TE'], 

                     qb=False,

                     ppr=True, 

                     year_min=2015,

                     year_max=2019

                    )



te_ppr
te_reg = corr_3_5_10(positions=['TE'], 

                     qb=False,

                     ppr=False, 

                     year_min=2015,

                     year_max=2019

                    )



te_reg
flex_ppr = corr_3_5_10(positions=['RB', 'WR', 'TE'], 

                       qb=False,

                       ppr=True,

                       year_min=2015,

                       year_max=2019

                      )



flex_ppr
flex_reg = corr_3_5_10(positions=['RB', 'WR', 'TE'], 

                       qb=False,

                       ppr=False, 

                       year_min=2015,

                       year_max=2019

                      )



flex_reg
# Specify o_line_cap table columns and datatypes

o_line_cap_col_types = {'seas' : int,

                        'roster_starters' : str, 

                        'team' : str, 

                        'players' : int, 

                        'cap_dollars' : int, 

                        'avg_cap' : int,

                        'perc_of_cap' : float

                       }



# Isolate column names

o_line_cap_cols = o_line_cap_col_types.keys()



# Import o_line_cap table

o_line_cap = pd.read_csv('../input/nfl-game-season-and-salary-data/CAP.csv',

                         delimiter=',',

                         usecols=o_line_cap_cols,

                         dtype=o_line_cap_col_types

                        )





# Specify snap table columns and datatypes

snap_col_types = {'gid' : int,

                  'tname' : str, 

                  'player' : str, 

                  'posd' : str, 

                  'poss' : str, 

                  'snp' : int,

                 }



# Isolate column names

snap_cols = snap_col_types.keys()



# Import snap table

snap = pd.read_csv('../input/nfl-game-season-and-salary-data/SNAP.csv',

                   delimiter=',',

                   usecols=snap_cols,

                   dtype=snap_col_types

                  )





# Specify game table columns and datatypes

games_col_types = {'gid' : int,

                   'seas' : int, 

                   'wk' : int, 

                   'day' : str, 

                   'v' : str, 

                   'h' : str,

                   'temp' : float,

                   'humd' : float,

                   'wspd' : float,

                   'cond' : str,

                   'surf' : str,

                   'ou' : float,

                   'sprv' : float,

                   'ptsv' : float,

                   'ptsh' : float

                 }



# Isolate column names

games_cols = games_col_types.keys()



# Import snap table

games = pd.read_csv('../input/nfl-game-season-and-salary-data/GAME.csv',

                    delimiter=',',

                    usecols=games_cols,

                    dtype=games_col_types

                   )



# Filter for OL only

o_line_pos = ['OL', 'C', 'LG', 'RG', 'LT', 'RT', 'NT']

o_line_table = snap[snap['posd'].isin(o_line_pos)]



# Bring in draft position and start year from player table

o_line_table = pd.merge(o_line_table,

                        player[['player', 'start', 'dpos']],

                        how='left',

                        left_on='player',

                        right_on='player'

                       )



# Replace NaN and zero in dpos with 300

o_line_table = o_line_table.replace(0, 300)

values = {'dpos': 300}

o_line_table = o_line_table.fillna(value=values)





# Bring in year and week from game table

o_line_table = pd.merge(o_line_table,

                        games[['gid', 'seas', 'wk']],

                        how='left',

                        left_on='gid',

                        right_on='gid'

                       )



# Create yrs_in_nfl column

o_line_table['yrs_in_nfl'] = o_line_table['seas'] - o_line_table['start']



# Create team_gid and team_year columns for merges (o_line and offense tables)

o_line_table['team_gid'] = o_line_table['tname'] + '-' + o_line_table['gid'].map(str)

o_line_table['team_year'] = o_line_table['tname'] + '-' + o_line_table['seas'].map(str)

offense['team_gid'] = offense['team'] + '-' + offense['gid'].map(str)

offense['team_year'] = offense['team'] + '-' + offense['year'].map(str)

offense_no_playoffs['team_gid'] = offense_no_playoffs['team'] + '-' + offense_no_playoffs['gid'].map(str)

offense_no_playoffs['team_year'] = offense_no_playoffs['team'] + '-' + offense_no_playoffs['year'].map(str)



# Calculate weighted avg colums for dpos and yrs_in_nfl (year and game)

o_line_table['dpos_wgt_game'] = (o_line_table.snp 

                                 / o_line_table.groupby('team_gid').snp.transform('sum') 

                                 * o_line_table.dpos

                                )



o_line_table['dpos_wgt_seas'] = (o_line_table.snp 

                                 / o_line_table.groupby('team_year').snp.transform('sum') 

                                 * o_line_table.dpos

                                )



o_line_table['yrs_nfl_wgt_game'] = (o_line_table.snp 

                                    / o_line_table.groupby('team_gid').snp.transform('sum') 

                                    * o_line_table.yrs_in_nfl

                                   )



o_line_table['yrs_nfl_wgt_seas'] = (o_line_table.snp 

                                    / o_line_table.groupby('team_year').snp.transform('sum') 

                                    * o_line_table.yrs_in_nfl

                                   )
# Create o_line_by_game table



# Specify columns to drop

drop_cols = ['gid',

             'player',

             'posd',

             'poss',

             'snp',

             'start',

             'dpos',

             'wk',

             'yrs_in_nfl',

             'wk',

             'dpos_wgt_seas',

             'yrs_nfl_wgt_seas',

             'team_year'

            ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['team_gid',

              'seas',

              'tname'

             ]

              

# Specify average columns

avg_cols = []



# Run prep_for_correlations function

o_line_by_game = prep_for_correlations(data=o_line_table, 

                                       drop_cols=drop_cols, 

                                       group_cols=group_cols,

                                       avg_cols=avg_cols,

                                       reset=True

                                      )



# Create o_line_by_seas table



# Specify columns to drop

drop_cols = ['gid',

             'player',

             'posd',

             'poss',

             'snp',

             'start',

             'dpos',

             'wk',

             'yrs_in_nfl',

             'wk',

             'dpos_wgt_game',

             'yrs_nfl_wgt_game',

             'team_gid'

            ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['team_year',

              'seas',

              'tname'

             ]

              

# Specify average columns

avg_cols = []



# Run prep_for_correlations function

o_line_by_seas = prep_for_correlations(data=o_line_table, 

                                       drop_cols=drop_cols, 

                                       group_cols=group_cols,

                                       avg_cols=avg_cols,

                                       reset=True

                                      )





# Group offense table to determine fantasy points by team_gid

team_gid_pts = offense[['team_gid',

                        'ppr_pass_pts',

                        'ppr_rush_pts',

                        'ppr_rec_pts',

                        'ppr_total_pts',

                        'reg_rec_pts',

                        'reg_total_pts'

                       ]

                      ].groupby(['team_gid']).sum()



# Merge scoring columns to o_line_table_by_game

o_line_by_game = pd.merge(o_line_by_game,

                                team_gid_pts,

                                how='left',

                                left_on='team_gid',

                                right_on='team_gid',

                               )





# Group offense table to determine fantasy points by team_gid

team_year_pts = offense_no_playoffs[['team_year',

                         'ppr_pass_pts',

                         'ppr_rush_pts',

                         'ppr_rec_pts',

                         'ppr_total_pts',

                         'reg_rec_pts',

                         'reg_total_pts'

                        ]

                       ].groupby(['team_year']).sum()



# Merge scoring columns to o_line_table

o_line_by_seas = pd.merge(o_line_by_seas,

                                team_year_pts,

                                how='left',

                                left_on='team_year',

                                right_on='team_year',

                               )



# Rename weighted columns

o_line_by_game = o_line_by_game.rename(columns={'dpos_wgt_game': 'dpos_wgt', 'yrs_nfl_wgt_game': 'yrs_nfl_wgt'})

o_line_by_seas = o_line_by_seas.rename(columns={'dpos_wgt_seas': 'dpos_wgt', 'yrs_nfl_wgt_seas': 'yrs_nfl_wgt'})



# Merge cap information to season table

o_line_cap['team_year'] = o_line_cap['team'] + '-' + o_line_cap['seas'].map(str)

o_line_by_seas = pd.merge(o_line_by_seas,

                          o_line_cap[['team_year', 'cap_dollars', 'avg_cap', 'perc_of_cap']],

                          how='left',

                          left_on='team_year',

                          right_on='team_year'

                       )
def o_line_stats_correlogram(data, metrics, season, num_y, year_min=2015, year_max=2019):

    

    # Filter the data for specified metrics, years, positions, minimum games

    data = data[(data['seas'] >= year_min) & 

                (data['seas'] <= year_max)

               ]

    

    data = data[metrics]

   

    # Prep title components

    if season:

        cgram_string = 'O-line Correlation with Fantasy Points - By Season'

    else:

        cgram_string = 'O-line Correlation with Fantasy Points - By Game'



    if year_min == year_max:

        year_string = str(year_min)

    else:

        year_string = str(year_min) + '-' + str(year_max)

    

    title = (cgram_string + ' - ' +

             year_string

            )



    # Create correlation column

    corr_table = data.corr()



    # Plot

    colors = 'RdYlGn'

    

    plt.figure(figsize=(12,5), dpi= 80)

    sns.heatmap(data.corr().head(num_y), 

                xticklabels=data.corr().columns, 

                yticklabels=list(data.corr().columns)[0:num_y], 

                cmap=colors,

                vmin=-1.0,

                vmax=1.0,

                center=0, 

                annot=True)



    # Decorations

    plt.title(title, fontsize=22, pad=12)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=12)

    plt.yticks(rotation=0)

    plt.show()
# Establish metrics for correlogram (game)

o_line_metrics = ['dpos_wgt',

                  'yrs_nfl_wgt',

                  'ppr_pass_pts',

                  'ppr_rush_pts',

                  'ppr_rec_pts',

                  'ppr_total_pts',

                  'reg_rec_pts',

                  'reg_total_pts'

                 ]



# Create o_line_by_game correlogram

o_line_stats_correlogram(o_line_by_game, 

                         o_line_metrics, 

                         num_y=2,

                         season=False, 

                         year_min=2015, 

                         year_max=2019)



# Establish metrics for correlogram (season)

o_line_metrics = ['dpos_wgt',

                  'yrs_nfl_wgt',

                  'cap_dollars',

                  'avg_cap',

                  'perc_of_cap',

                  'ppr_pass_pts',

                  'ppr_rush_pts',

                  'ppr_rec_pts',

                  'ppr_total_pts',

                  'reg_rec_pts',

                  'reg_total_pts'

                 ]



# Create o_line_by_seas correlogram

o_line_stats_correlogram(o_line_by_seas, 

                         o_line_metrics, 

                         num_y=5,

                         season=True, 

                         year_min=2015, 

                         year_max=2019)
# Establish dictionary to consolidate weather conditions

weather_dict = {'Cold' : 'Snow',

                'Light Snow' : 'Snow',

                'Flurries' : 'Snow',

                'Clear' : 'Fair/Clear',

                'Mostly Sunny' : 'Fair/Clear',

                'Partly Cloudy' : 'Fair/Clear',

                'Sunny' : 'Fair/Clear',

                'Fair' : 'Fair/Clear',

                'Chance Rain' : 'Light Rain',

                'Light Showers' : 'Light Rain',

                'Windy' : 'Light Rain',

                'Cloiudy' : 'Overcast',

                'Cloudy' : 'Overcast',

                'Foggy' : 'Overcast',

                'Hazy' : 'Overcast',

                'Mostly Cloudy' : 'Overcast',

                'Partly Sunny' : 'Overcast',

                'Raining' : 'Rain',

                'Showers' : 'Rain',

                'Thunderstorms' : 'Rain',

                'Closed Roof' : 'Roof',

                'Covered Roof' : 'Roof'

               }



# Establish dictionary to consolidate weather conditions

surface_dict = {'AstroTurf' : 'Astro Turf',

                'FieldTurf' : 'Sythetic Turf',

                'AstroPlay' : 'Astro Turf',

                'NeXTurf' : 'Sythetic Turf',

                'MomentumTurf' : 'Sythetic Turf',

                'SportGrass' : 'Sythetic Turf',

                'DD GrassMaster' : 'GrassTurf Hybrid',

                'Sportex' : 'Sythetic Turf',

                'A Turf Titan' : 'Sythetic Turf',

                'UBU Speed Series S5M' : 'Sythetic Turf',

                'Artificial Turf' : 'Sythetic Turf',

                'Matrix RealGrass' : 'Sythetic Turf',

                'Hellas Matrix Turf' : 'Sythetic Turf',

                'Shaw Sports Momentum Pro' : 'Sythetic Turf',

               }



# Replace values in cond and surf columns using dictionaries

games = games.replace({'cond' : weather_dict,

                       'surf' : surface_dict

                      }

                     )



# Merge cond and surf columns to offense table

games_cols = ['gid', 

              'cond', 

              'surf', 

              'v', 

              'h', 

              'temp',

              'humd',

              'wspd',

              'ou',

              'ptsv',

              'ptsh']



offense = pd.merge(offense,

                   games[games_cols],

                   how='left',

                   left_on='gid',

                   right_on='gid'

                  )

points_by_pos_and_x_boxplot(offense, 

                            title='PPR Points by Position and Weather Conditions',

                            metric='ppr_total_pts',

                            x='cond',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
points_by_pos_and_x_boxplot(offense, 

                            title='Regular Points by Position and Weather Conditions',

                            metric='reg_total_pts',

                            x='cond',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
points_by_pos_and_x_boxplot(offense, 

                            title='PPR Points by Position and Playing Surface',

                            metric='ppr_total_pts',

                            x='surf',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
points_by_pos_and_x_boxplot(offense, 

                            title='Regular Points by Position and Playing Surface',

                            metric='reg_total_pts',

                            x='surf',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Correlation of over/under and implied home/visitor scores  to fantasy points 

offense['home_visitor'] = np.where(offense['team'] == offense['h'], 'home', 'visitor')

offense['proj_team_pts'] = np.where(offense['home_visitor'] == 'home', offense['ptsh'], offense['ptsv'])



# Set up columns for prep_for_correlations



# Specify columns to sum

sum_cols = ['ppr_pass_pts', 

            'ppr_rush_pts',

            'ppr_rec_pts', 

            'ppr_pts_ex_ret', 

            'ppr_total_pts',

            'reg_pass_pts', 

            'reg_rush_pts',

            'reg_rec_pts', 

            'reg_pts_ex_ret', 

            'reg_total_pts'

           ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['team_gid',

              'year',

              'team'

             ]



# Specify average columns (depth chart position)

avg_cols = ['ou', 'proj_team_pts']



# Specify columns to drop

drop_cols = [x for x in list(offense.columns) if ((x not in sum_cols) &

                                                  (x not in count_cols) &

                                                  (x not in group_cols) &

                                                  (x not in avg_cols)

                                                 )

            ]



# Run prep_for_correlations on projected points table

offense_proj_pts = prep_for_correlations(data=offense, 

                                         drop_cols=drop_cols, 

                                         group_cols=group_cols,

                                         avg_cols=avg_cols,

                                         reset=True

                                        )



# Specify metrics for correlograms

ppr_metrics = ['ou', 

               'proj_team_pts',

               'ppr_pass_pts', 

               'ppr_rush_pts',

               'ppr_rec_pts', 

               'ppr_pts_ex_ret', 

               'ppr_total_pts'

              ]



reg_metrics = ['ou', 

               'proj_team_pts',

               'reg_pass_pts', 

               'reg_rush_pts',

               'reg_rec_pts', 

               'reg_pts_ex_ret', 

               'reg_total_pts'

              ]
def game_conditions_correlogram(data, title, metrics, num_y, ppr, fig_height, year_min=2015, year_max=2019):

    

    # Filter the data for specified metrics, years, positions, minimum games

    data = data[(data['year'] >= year_min) & 

                (data['year'] <= year_max)

               ]

    

    data = data[metrics]

   

    # Prep title components

    cgram_string = title



    if year_min == year_max:

        year_string = str(year_min)

    else:

        year_string = str(year_min) + '-' + str(year_max)

    

    title = (cgram_string + ' - ' +

             year_string

            )



    # Create correlation column

    corr_table = data.corr()



    # Plot

    colors = 'RdYlGn' if ppr else 'RdYlBu'

    

    plt.figure(figsize=(12,fig_height), dpi= 80)

    sns.heatmap(data.corr().head(num_y), 

                xticklabels=data.corr().columns, 

                yticklabels=list(data.corr().columns)[0:num_y], 

                cmap=colors,

                vmin=-1.0,

                vmax=1.0,

                center=0, 

                annot=True)



    # Decorations

    plt.title(title, fontsize=22, pad=12)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=12)

    plt.yticks(rotation=0)

    plt.show()
game_conditions_correlogram(offense_proj_pts, 

                            title='Projected Game Points vs PPR Fantasy Points',

                            metrics=ppr_metrics,  

                            fig_height=5,

                            num_y=2,

                            ppr=True,

                            year_min=2015, 

                            year_max=2019)
game_conditions_correlogram(offense_proj_pts, 

                            title='Projected Game Points vs Regular Fantasy Points',

                            metrics=reg_metrics,  

                            fig_height=5,

                            num_y=2,

                            ppr=False,

                            year_min=2015, 

                            year_max=2019)
points_by_pos_and_x_boxplot(offense, 

                            title='PPR Points by Position and Home/Away',

                            metric='ppr_total_pts',

                            x='home_visitor',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
points_by_pos_and_x_boxplot(offense, 

                            title='Regular Points by Position and Home/Away',

                            metric='reg_total_pts',

                            x='home_visitor',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Correct blanks in humidity and windspeed for Dome games

offense['humd'] = np.where(offense['cond'] == 'Dome', 0, offense['humd'])

offense['wspd'] = np.where(offense['cond'] == 'Dome', 0, offense['wspd'])



# Filter for games where humd, wspd, and temp are all populated

offense_conditions = offense.dropna()



# Set up columns for prep_for_correlations



# Specify columns to sum

sum_cols = ['ppr_pass_pts', 

            'ppr_rush_pts',

            'ppr_rec_pts', 

            'ppr_pts_ex_ret', 

            'ppr_total_pts',

            'reg_pass_pts', 

            'reg_rush_pts',

            'reg_rec_pts', 

            'reg_pts_ex_ret', 

            'reg_total_pts'

           ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['gid',

              'year',

             ]



# Specify average columns (depth chart position)

avg_cols = ['temp',

            'humd',

            'wspd'

           ]



# Specify columns to drop

drop_cols = [x for x in list(offense_conditions.columns) if ((x not in sum_cols) &

                                                             (x not in count_cols) &

                                                             (x not in group_cols) &

                                                             (x not in avg_cols)

                                                            )

            ]



# Run prep_for_correlations on conditions table

offense_cond_grouped = prep_for_correlations(data=offense_conditions, 

                                             drop_cols=drop_cols, 

                                             group_cols=group_cols,

                                             avg_cols=avg_cols,

                                             reset=True

                                            )



# Specifiy metrics for correlograms

ppr_metrics = ['temp', 

               'humd',

               'wspd',

               'ppr_pass_pts', 

               'ppr_rush_pts',

               'ppr_rec_pts', 

               'ppr_pts_ex_ret', 

               'ppr_total_pts'

              ]



reg_metrics = ['temp', 

               'humd',

               'wspd',

               'reg_pass_pts', 

               'reg_rush_pts',

               'reg_rec_pts', 

               'reg_pts_ex_ret', 

               'reg_total_pts'

              ]
game_conditions_correlogram(offense_cond_grouped, 

                            title='PPR Fantasy Points and Weather Correlation',

                            metrics=reg_metrics,  

                            fig_height=5,

                            num_y=3,

                            ppr=True,

                            year_min=2015, 

                            year_max=2019)
game_conditions_correlogram(offense_cond_grouped, 

                            title='Regular Fantasy Points and Weather Correlation',

                            metrics=reg_metrics,  

                            fig_height=5,

                            num_y=3,

                            ppr=False,

                            year_min=2015, 

                            year_max=2019)
# Bin temp column

bins = [-20, 34, 49, 69, 84, 120]

labels = ['A - Under 35',

          'B - 35-49',

          'C - 50-69',

          'D - 70-85',

          'E - Over 85'

         ]



offense_conditions['temp_bins'] = pd.cut(offense_conditions['temp'], bins=bins, labels=labels)

offense_conditions = offense_conditions.sort_values(by=['temp_bins']).reset_index(drop=True)

offense_conditions = offense_conditions.replace({'temp_bins': {'A - Under 35' : 'Under 35',

                                                               'B - 35-49' : '35-49',

                                                               'C - 50-69' : '50-69',

                                                               'D - 70-85' : '70-85',

                                                               'E - Over 85' : 'Over 85'

                                                              }

                                                }

                                               )



# Run boxplot

points_by_pos_and_x_boxplot(offense_conditions, 

                            title='PPR Points by Position and Game Temp',

                            metric='ppr_total_pts',

                            x='temp_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Bin humd column

bins = [-1, 19, 39, 59, 79, 100]

labels = ['A - Under 20',

          'B - 20-39',

          'C - 40-59',

          'D - 60-79',

          'E - Over 80'

         ]



offense_conditions['humd_bins'] = pd.cut(offense_conditions['humd'], bins=bins, labels=labels)

offense_conditions = offense_conditions.sort_values(by=['humd_bins']).reset_index(drop=True)

offense_conditions = offense_conditions.replace({'humd_bins': {'A - Under 20' : 'Under 20',

                                                               'B - 20-39' : '20-99',

                                                               'C - 40-59' : '40-59',

                                                               'D - 60-79' : '60-79',

                                                               'E - Over 80' : 'Over 80'

                                                              }

                                                }

                                               )



# Run boxplot

points_by_pos_and_x_boxplot(offense_conditions, 

                            title='PPR Points by Position and Humidity',

                            metric='ppr_total_pts',

                            x='humd_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Bin wspd column

bins = [-1, 4, 9, 14, 19, 100]

labels = ['Under 5',

          '5-9',

          '10-14',

          '15-19',

          'Over 20'

         ]



offense_conditions['wspd_bins'] = pd.cut(offense_conditions['wspd'], bins=bins, labels=labels)

offense_conditions = offense_conditions.sort_values(by=['wspd_bins']).reset_index(drop=True)

offense_conditions = offense_conditions.replace({'wspd_bins': {'A - Under 5' : 'Under 5',

                                                               'B - 5-9' : '5-9',

                                                               'C - 10-14' : '10-14',

                                                               'D - 15-19' : '15-19',

                                                               'E - Over 20' : 'Over 20'

                                                              }

                                                }

                                               )



# Run boxplot

points_by_pos_and_x_boxplot(offense_conditions, 

                            title='PPR Points by Position and Wind Speed',

                            metric='ppr_total_pts',

                            x='wspd_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Merge start year column from player table and filter for rookies (1st season)

player_cols = ['player', 

               'height',

               'weight',

               'forty',

               'bench',

               'vertical',

               'broad',

               'shuttle',

               'cone',

               'arm',

               'hand',

               'dpos',

               'start'

              ]



full_season_rookies = pd.merge(full_season_stats,

                               player[player_cols],

                               how='left',

                               left_on='player',

                               right_on='player'

                              )



full_season_rookies = full_season_rookies[full_season_rookies['year'] == full_season_rookies['start']] 



# Replace 0 with 300 in dpos column

full_season_rookies = full_season_rookies.replace({'dpos' : 0}, 300)



# Replace 0 with NaN in various columns (for correlations)

cols = ['forty',

        'bench',

        'vertical',

        'broad',

        'shuttle',

        'cone',

        'arm',

        'hand'

       ]



full_season_rookies[cols] = full_season_rookies[cols].replace({0 : np.nan})





# Establish metrics for correlograms

ppr_metrics = ['height',

               'weight',

               'forty',

               'bench',

               'vertical',

               'broad',

               'shuttle',

               'cone',

               'arm',

               'hand',

               'dpos',

               'ppr_pass_pts',

               'ppr_rush_pts',

               'ppr_rec_pts',

               'ppr_total_pts'

              ]



reg_metrics = ['height',

               'weight',

               'forty',

               'bench',

               'vertical',

               'broad',

               'shuttle',

               'cone',

               'arm',

               'hand',

               'dpos',

               'reg_pass_pts',

               'reg_rush_pts',

               'reg_rec_pts',

               'reg_total_pts'

              ]
def rookie_correlogram(data, title, metrics, num_y, fig_height, ppr, positions, year_min=2015, year_max=2019):

    

    # Filter the data for specified metrics, years, positions, minimum games

    data = data[(data['year'] >= year_min) & 

                (data['year'] <= year_max)

               ]

    

    if len(positions) > 0:

        data = data[data['pos1'].isin(positions)]

    

    data = data[metrics]

   

    # Prep title components

    cgram_string = title



    if year_min == year_max:

        year_string = str(year_min)

    else:

        year_string = str(year_min) + '-' + str(year_max)

    

    if len(positions) > 0:

        position_string = ', '.join([str(x) for x in positions])

    else:

        position_string = 'All Positions'

    

    title = (cgram_string + ' - ' +

             position_string + ' - ' +

             year_string

            )



    # Create correlation column

    #dropped = data.shape[0] - data.dropna().shape[0]

    corr_table = data.corr()



    # Plot

    colors = 'RdYlGn' if ppr else 'RdYlBu'

    

    plt.figure(figsize=(12,fig_height), dpi= 80)

    sns.heatmap(corr_table.head(num_y), 

                xticklabels=corr_table.columns, 

                yticklabels=list(corr_table.columns)[0:num_y], 

                cmap=colors,

                vmin=-1.0,

                vmax=1.0,

                center=0, 

                annot=True)



    # Decorations

    plt.title(title, fontsize=22, pad=12)

    plt.xticks(fontsize=12)

    plt.yticks(fontsize=12)

    plt.yticks(rotation=0)

    plt.show()
rookie_correlogram(full_season_rookies,

                   metrics=ppr_metrics,

                   title='PPR Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=True, 

                   positions=['WR'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=reg_metrics,

                   title='Regular Scoring Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=False, 

                   positions=['WR'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=ppr_metrics,

                   title='PPR Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=True, 

                   positions=['RB'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=reg_metrics,

                   title='Regular Scoring Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=False, 

                   positions=['RB'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=ppr_metrics,

                   title='PPR Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=True, 

                   positions=['TE'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=reg_metrics,

                   title='Regular Scoring Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=False, 

                   positions=['TE'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=ppr_metrics,

                   title='PPR Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=True, 

                   positions=['QB'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(full_season_rookies,

                   metrics=reg_metrics,

                   title='Regular Scoring Rookie Correlogram',

                   num_y=11, 

                   fig_height=8, 

                   ppr=False, 

                   positions=['QB'], 

                   year_min=2015, 

                   year_max=2019)
# Set up bins for draft position

bins = [0, 32, 64, 96, 128, 224, 500]

labels = ['A - 1st Rd',

          'B - 2nd Rd',

          'C - 3rd Rd',

          'D - 4th Rd',

          'E - 5th-7th', 

          'F - Undrafted'

         ]



full_season_rookies['draft_bins'] = pd.cut(full_season_rookies['dpos'], bins=bins, labels=labels)

full_season_rookies = full_season_rookies.sort_values(by=['draft_bins']).reset_index(drop=True)

full_season_rookies = full_season_rookies.replace({'draft_bins': {'A - 1st Rd' : '1st Rd',

                                                                  'B - 2nd Rd' : '2nd Rd',

                                                                  'C - 3rd Rd' : '3rd Rd',

                                                                  'D - 4th Rd' : '4th Rd',

                                                                  'E - 5th-7th' : '5th-7th', 

                                                                  'F - Undrafted' : 'Undrafted'

                                                                 }

                                                  }

                                                 )

                                                      

# Run rookie draft position boxplot

points_by_pos_and_x_boxplot(full_season_rookies, 

                            title='Rookie PPR Points by Position and Draft Position',

                            metric='ppr_total_pts',

                            x='draft_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2015, 2016, 2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Set up bins for 40 yd dash

bins = [4.00, 4.40, 4.50, 4.60, 4.80, 5.00, 10.00]

labels = ['A - Under 4.4',

          'B - 4.4-4.49',

          'C - 4.5-4.59',

          'D - 4.6-4.79',

          'E - 4.8-4.99', 

          'F - 5.0+'

         ]



full_season_rookies['40_bins'] = pd.cut(full_season_rookies['forty'], bins=bins, labels=labels)

full_season_rookies = full_season_rookies.sort_values(by=['40_bins']).reset_index(drop=True)

full_season_rookies = full_season_rookies.replace({'40_bins': {'A - Under 4.4' : 'Under 4.4',

                                                               'B - 4.4-4.49' : '4.4-4.49',

                                                               'C - 4.5-4.59' : '4.5-4.59',

                                                               'D - 4.6-4.79' : '4.6-4.79',

                                                               'E - 4.8-4.99' : '4.8-4.99',

                                                               'F - 5.0+' : '5.0+'

                                                              }

                                                  }

                                                 )

                                                      

# Run rookie 40 yd dash boxplot

points_by_pos_and_x_boxplot(full_season_rookies, 

                            title='Rookie PPR Points by Position and 40 Yard Dash',

                            metric='ppr_total_pts',

                            x='40_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2015, 2016, 2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Set up bins for broad jump

bins = [5, 100, 110, 120, 130, 200]

labels = ['A - Under 100',

          'B - 100-109',

          'C - 110-119',

          'D - 120-129',

          'E - 130+'

         ]



full_season_rookies['broad_jump_bins'] = pd.cut(full_season_rookies['broad'], bins=bins, labels=labels)

full_season_rookies = full_season_rookies.sort_values(by=['broad_jump_bins']).reset_index(drop=True)

full_season_rookies = full_season_rookies.replace({'broad_jump_bins': {'A - Under 100' : 'Under 100',

                                                                       'B - 100-109' : '100-109',

                                                                       'C - 110-119' : '110-119',

                                                                       'D - 120-129' : '120-129',

                                                                       'E - 130+' : '130+'

                                                                      }

                                                  }

                                                 )

                                                      

# Run rookie broad jump boxplot

points_by_pos_and_x_boxplot(full_season_rookies, 

                            title='Rookie PPR Points by Position and Broad Jump',

                            metric='ppr_total_pts',

                            x='broad_jump_bins',

                            positions=['QB', 'RB', 'WR', 'TE'], 

                            years=[2015, 2016, 2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Set up bins for hand size

bins = [1, 9.5, 10.0, 20.0]

labels = ['A - Under 9.5',

          'B - 9.5-9.9',

          'C - 10.0+',

         ]



full_season_rookies['hand_size_bins'] = pd.cut(full_season_rookies['hand'], bins=bins, labels=labels)

full_season_rookies = full_season_rookies.sort_values(by=['hand_size_bins']).reset_index(drop=True)

full_season_rookies = full_season_rookies.replace({'hand_size_bins': {'A - Under 9.5' : 'Under 9.5',

                                                                      'B - 9.5-9.9' : '9.5-9.9',

                                                                      'C - 10.0+' : '10.0+',

                                                                     }

                                                  }

                                                 )

                                                      

# Run rookie QB hand size boxplot

points_by_pos_and_x_boxplot(full_season_rookies, 

                            title='Rookie QB PPR Points by Hand Size',

                            metric='ppr_total_pts',

                            x='hand_size_bins',

                            positions=['QB'], 

                            years=[2015, 2016, 2017, 2018, 2019], 

                            color_dict=color_dict

                           )
# Prepare prior year veteran points table

py_veteran_points = per_game_stats[['player_year', 

                                    'player', 

                                    'year', 

                                    'team', 

                                    'pos1', 

                                    'gid', 

                                    'ppr_total_pts', 

                                    'reg_total_pts'

                                   ]

                                  ]



py_veteran_points['next_year'] = py_veteran_points['year'] + 1

py_veteran_points['team_next_year_pos1'] = (py_veteran_points['team'] + '-' + 

                                            py_veteran_points['next_year'].map(str) + '-' +

                                            py_veteran_points['pos1']

                                           )



# Specify columns to drop

drop_cols = ['gid',

             'player',

             'player_year',

             'team',

             'next_year',

             'year',

             'pos1'

            ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['team_next_year_pos1']

              

# Specify average columns

avg_cols = []



# Run prep_for_correlations function

py_veteran_points_grouped = prep_for_correlations(data=py_veteran_points, 

                                                  drop_cols=drop_cols, 

                                                  group_cols=group_cols,

                                                  avg_cols=avg_cols,

                                                  reset=True

                                                 )



# Rename points columns

py_veteran_points_grouped = py_veteran_points_grouped.rename(columns={'ppr_total_pts' : 'vet_py_ppr_pts_per_game',

                                                                      'reg_total_pts' : 'vet_py_reg_pts_per_game'

                                                                     }

                                                            )



# Add team_year to full season rookie table

full_season_rookies['team_year_pos1'] = (full_season_rookies['team'] + '-' + 

                                         full_season_rookies['year'].map(str) + '-' + 

                                         full_season_rookies['pos1']

                                        )
# Prepare team points table

team_pts_by_position = offense[['pos1',

                                'year',

                                'team',

                                'ppr_total_pts',

                                'reg_total_pts'

                               ]

                              ]

                               

team_pts_by_position['next_year'] = team_pts_by_position['year'] + 1



team_pts_by_position['team_next_year_pos1'] = (team_pts_by_position['team'] + '-' + 

                                               team_pts_by_position['next_year'].map(str) + '-' +

                                               team_pts_by_position['pos1']

                                              )



# Specify columns to drop

drop_cols = ['pos1',

             'year',

             'team',

             'next_year'

            ]



# Specify count columns (games played)

count_cols = []



# Specify grouping columns

group_cols = ['team_next_year_pos1']

              

# Specify average columns

avg_cols = []



# Run prep_for_correlations function

team_pts_by_position_grouped = prep_for_correlations(data=team_pts_by_position, 

                                                     drop_cols=drop_cols, 

                                                     group_cols=group_cols,

                                                     avg_cols=avg_cols,

                                                     reset=True

                                                    )



# Rename points columns

team_pts_by_position_grouped = team_pts_by_position_grouped.rename(columns={'ppr_total_pts' : 'py_ppr_pts_by_position',

                                                                            'reg_total_pts' : 'py_reg_pts_by_position'

                                                                           }

                                                                  )
# Merge tables for correlation of rookie performance vs vet performance

rookie_cols = ['player_year',

               'team_year_pos1',

               'year',

               'pos1',

               'ppr_total_pts',

               'reg_total_pts'

              ]



vet_cols = ['team_next_year_pos1',

            'vet_py_ppr_pts_per_game',

            'vet_py_reg_pts_per_game'

           ]



rookie_performance_vs_py = pd.merge(full_season_rookies[rookie_cols],

                                    py_veteran_points_grouped[vet_cols],

                                    how='left',

                                    left_on='team_year_pos1',

                                    right_on='team_next_year_pos1'

                                   )



# Merge tables for correlation of rookie performance vs team points by position

team_pts_cols = ['team_next_year_pos1',

                 'py_ppr_pts_by_position',

                 'py_reg_pts_by_position'

                ]



rookie_performance_vs_py = pd.merge(rookie_performance_vs_py,

                                    team_pts_by_position_grouped[team_pts_cols],

                                    how='left',

                                    left_on='team_year_pos1',

                                    right_on='team_next_year_pos1'

                                   )

ppr_metrics = ['vet_py_ppr_pts_per_game',

               'py_ppr_pts_by_position',

               'ppr_total_pts'

              ]



reg_metrics = ['vet_py_reg_pts_per_game',

               'py_reg_pts_by_position',

               'reg_total_pts'

              ]



rookie_correlogram(rookie_performance_vs_py,

                   metrics=ppr_metrics,

                   title='PPR Rookie Pts vs PY Metrics',

                   num_y=11, 

                   fig_height=4, 

                   ppr=True, 

                   positions=['RB', 'WR', 'TE', 'QB'], 

                   year_min=2015, 

                   year_max=2019)
rookie_correlogram(rookie_performance_vs_py,

                   metrics=reg_metrics,

                   title='Regular Rookie Pts vs PY Metrics',

                   num_y=11, 

                   fig_height=4, 

                   ppr=False, 

                   positions=['RB', 'WR', 'TE', 'QB'], 

                   year_min=2015, 

                   year_max=2019)