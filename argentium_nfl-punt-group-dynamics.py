import os
import re
import pandas as pd
import numpy as np
import seaborn as sns

import scipy
import math
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings("ignore")
df_play_info = pd.read_csv('../input/play_information.csv')
df_injury = pd.read_csv('../input/video_review.csv')
df_punt_role = pd.read_csv('../input/play_player_role_data.csv')
team_positions = {'Return': 
                  ['VR', 'VRo', 'VRi', 
                   'VL', 'VLo', 'VLi',
                   'PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6',
                   'PDM',
                   'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6',
                   'PLR', 'PLR1', 'PLR2', 'PLR3',
                   'PLM', 'PLM1',
                   'PLL', 'PLL1', 'PLL2', 'PLL3', 'PLLi',
                   'PR', 'PFB'
                   ],
     'Coverage': ['GR', 'GRo', 'GRi',
                  'GL', 'GLo', 'GLi',
                  'PRG', 'PRT', 'PRW',
                  'PPR', 'PPRo', 'PPRi', 
                  'PPL', 'PPLo', 'PPLi',
                  'P', 'PC', 'PLS',
                  'PLW', 'PLT', 'PLG'
                  ]}

role_categories = {'G': ['GR', 'GRo', 'GRi',
                        'GL', 'GLo', 'GLi'],
                      'Coverage_Center': ['PRG', 'PLG', 'PRT', 'PLT', 'PRW', 'PLW'],
                  'PP': ['PPR', 'PPRo', 'PPRi',
                         'PPL', 'PPLo', 'PPLi'],
                  'P': ['P'],
                  'PC': ['PC'],
                  'PLS': ['PLS'],
                    'V': ['VR', 'VRo', 'VRi',
                        'VL', 'VLo', 'VLi'],
                  'PD': ['PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6',
                          'PDM',
                         'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6'],
                  'PL': ['PLR', 'PLR1', 'PLR2', 'PLR3',
                         'PLM', 'PLM1',
                         'PLL', 'PLL1', 'PLL2', 'PLL3', 'PLLi'],
                  'PR': ['PR'],
                  'PFB': ['PFB']
                 }

sides = {'Right': ['GR', 'GRo', 'GRi', 'PRG', 'PRT', 'PRW',
                  'PPR', 'PPRo', 'PPRi',
                   'PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6',
                   'PLR', 'PLR1', 'PLR2', 'PLR3',
                  'VR', 'VRo', 'VRi'],
         'Left': ['GL', 'GLo', 'GLi', 'PLG', 'PLT', 'PLW',
                 'PPL', 'PPLo', 'PPLi',
                  'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6',
                  'PLL', 'PLL1', 'PLL2', 'PLL3', 'PLLi',
                 'VL', 'VLo', 'VLi'],
         'Center': ['P', 'PC', 'PLS', 'PDM', 'PLM', 'PLM1', 'PR', 'PFB']
                 }

# Add the corresponding side of their role
def set_category(role, dictionary):
    for catgory in dictionary.keys():
        if str(role) in dictionary[catgory]:
            return str(catgory)
    return None

df_punt_role['Team'] = df_punt_role.apply(lambda row: 
                                          set_category(row['Role'], team_positions), axis=1)
df_punt_role['Side'] = df_punt_role.apply(lambda row: 
                                          set_category(row['Role'], sides), axis=1)
df_punt_role['Role_Category'] = df_punt_role.apply(lambda row: 
                                                   set_category(row['Role'], role_categories),
                                                   axis=1)
df_punt_role = df_punt_role.drop(columns=['Season_Year'])
def get_goal(activity):
    if (activity == 'Blocking') or (activity == 'Tackled'):
        return 'Offensive'
    else:
        return 'Defensive'

# Add the corresponding side of their role
def set_phase(row):
    goal = get_goal(row['Player_Activity_Derived'])
    if row['Team'] == 'Coverage':
        if goal == 'Offensive':
            return 1
        else:
            return 2
    else: # Return Team
        if goal == 'Offensive':
            return 2
        else:
            return 1

# Convert to int data type
df_injury['Primary_Partner_GSISID'] = df_injury.apply(lambda row: 
                                                                  row['Primary_Partner_GSISID'] 
                                                                  if (row['Primary_Partner_GSISID'] != 'Unclear')
                                                                 else 0,
                                                                 axis=1)
df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].fillna(0)
df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].astype(int)

# Merge with df_punt_role
df_injury = df_injury.merge(df_punt_role,
                                right_on=['GameKey', 'PlayID', 'GSISID'],
                                 left_on=['GameKey', 'PlayID', 'GSISID'],
                           how='left')
df_injury = df_injury.merge(df_punt_role,
                                right_on=['GameKey', 'PlayID', 'GSISID'],
                                 left_on=['GameKey', 'PlayID', 'Primary_Partner_GSISID'],
                            suffixes=('', '_Partner'),
                           how='left')
df_injury['Phase'] = df_injury.apply(lambda row: 
                                                set_phase(row), 
                                                axis=1)
df_yardline = df_play_info['YardLine'].str.split(" ", n = 1, expand = True)
df_play_info['yard_team'] = df_yardline[0]
df_play_info['yard_number'] = df_yardline[1].astype(float)

# Process Team Sides
df_home_visit = df_play_info['Home_Team_Visit_Team'].str.split("-", n = 1, expand = True)
df_play_info['home'] = df_home_visit[0]
df_play_info['visit'] = df_home_visit[1]

# Convert to coordinate system, origin at goal line
def convert_yardage(row):
    actual_yards = row['yard_number'] + 10
    if row['yard_team'] == row['home']:
        return actual_yards
    else:
        return 120 - actual_yards

# Convert to goal line distance
def convert_goal_distance(row):
    if row.loc[('Poss_Team')] == row.loc[('home')]:
        return row.loc[('Scrimmage_Line')]
    else:
        return 120 - row.loc[('Scrimmage_Line')]

df_play_info['Home_Poss'] = df_play_info.apply(lambda row: row.loc[('Poss_Team')] == row.loc[('home')], axis=1)
df_play_info['Scrimmage_Line'] = df_play_info.apply(lambda row: convert_yardage(row), axis=1)
df_play_info['Goal_Line'] = df_play_info.apply(lambda row: convert_goal_distance(row), axis=1)

# Convert to meters
df_play_info['Scrimmage_Line'] = 0.9144 * df_play_info['Scrimmage_Line']
df_play_info['Goal_Line'] = 0.9144 * df_play_info['Goal_Line']
# Graph
def graph_distribution(column):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
    mean = column.mean()
    median = column.median()
    mode = column.mode().get_values()[0]

    sns.boxplot(column, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    ax_box.axvline(mode, color='b', linestyle='-')

    sns.distplot(column, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    ax_hist.axvline(mode, color='b', linestyle='-')

    plt.legend({'Mean':mean,'Median':median,'Mode':mode})

    ax_box.set(xlabel='')
    plt.show()
import gc
import tqdm
import feather

dtypes = {'Season_Year': 'int16',
         'GameKey': 'int64',
         'PlayID': 'int16',
         'GSISID': 'float32',
         'Time': 'str',
         'x': 'float32',
         'y': 'float32',
         'dis': 'float32',
         'o': 'float32',
         'dir': 'float32',
         'Event': 'str'}
col_names = list(dtypes.keys())

ngs_files = ['NGS-2016-pre.csv',
             'NGS-2016-reg-wk1-6.csv',
             'NGS-2016-reg-wk7-12.csv',
             'NGS-2016-reg-wk13-17.csv',
             'NGS-2016-post.csv',
             'NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv',
             'NGS-2017-reg-wk13-17.csv',
             'NGS-2017-post.csv']

# Load each ngs file and append it to a list. 
# We will turn this into a DataFrame in the next step

df_list = []

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv('../input/'+i, usecols=col_names,dtype=dtypes)
    
    df_list.append(df)

# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()

# # Convert Time to datetime
ngs['Time'] = pd.to_datetime(ngs['Time'], format='%Y-%m-%d %H:%M:%S')

# There are 2536 out of 66,492,490 cases where GSISID is NAN. Let's drop those to convert the data type
ngs = ngs[~ngs['GSISID'].isna()]

# Convert GSISID to integer
ngs['GSISID'] = ngs['GSISID'].astype('int32')

# ngs.set_index(['GameKey', 'PlayID', 'GSISID'], inplace=True)
# Get Injury Moves
# Ensure same data types
columns = ['GameKey', 'PlayID', 'GSISID']
for col in columns:
    df_injury[col] = df_injury[col].astype(ngs[col].dtype)
    df_punt_role[col] = df_punt_role[col].astype(ngs[col].dtype)

df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].astype(ngs['GSISID'].dtype)

# Get Only Games with Injuries
df_injury_moves = ngs.merge(df_injury[['GameKey', 'PlayID']],
                                  left_on=['GameKey', 'PlayID'],
                                  right_on=['GameKey', 'PlayID'],
                                 suffixes=('', '_Injury'))

# # Get All Moves of PR
# df_pr = df_punt_role[df_punt_role['Role']=='PR']
# df_pr_moves = ngs.merge(df_pr[['GameKey', 'PlayID', 'GSISID']],
#                                   left_on=['GameKey', 'PlayID', 'GSISID'],
#                                   right_on=['GameKey', 'PlayID', 'GSISID'])

# Delete the dataframe list to release memory
del ngs
gc.collect()
# I added the role for easier categorization
df_injury_moves = df_injury_moves.merge(df_punt_role,
                                  left_on=['GameKey', 'PlayID', 'GSISID'],
                                  right_on=['GameKey', 'PlayID', 'GSISID'],
                                 suffixes=('', '_Injury'))

# Create Gameplay ID
df_injury_moves['Gameplay'] = df_injury_moves.apply(lambda row: 
                                            str(row['GameKey']) + '_' + 
                                            str(row['PlayID']),
                                           axis=1)
df_injury_moves['x'] = 0.9144 * df_injury_moves['x']
df_injury_moves['y'] = 0.9144 * df_injury_moves['y']
df_injury_moves['dis'] = 0.9144 * df_injury_moves['dis']
# Get only events
df_events = df_injury_moves.dropna(subset=['Event'])
df_events_indexed = df_events.set_index(['GameKey', 'PlayID', 'Event'])

# Get event
df_injury_moves[['Event']] = df_injury_moves[['Event']].fillna('')
df_injury_moves_gameplay = df_injury_moves.groupby(['GameKey', 'PlayID']).agg({'Time': ['min', 'max'],
                                                                               'dis': ['min', 'median', 'max', 'sum'],
                                                                               'dir': ['min', 'median', 'max'],
                                                                               'o': ['min', 'median', 'max']})

# Get start time by subtracting the time at play start in the specified GameKey, PlayID
def get_start_time(row):
    start = df_events_indexed.loc[(row['GameKey'], row['PlayID'], 'ball_snap')]['Time'][0]
    end = row['Time']
    return (end - start).total_seconds()

df_injury_moves['PlayStartTime'] = df_injury_moves.apply(lambda row: 
                                                         get_start_time(row),
                                                         axis=1)
df_injury_moves = df_injury_moves[df_injury_moves['PlayStartTime'] >= 0]
df_injury_moves['Sequence'] = df_injury_moves.apply(lambda row: 
                                                    str(int(row['PlayStartTime']*10)), 
                                                    axis=1)
df_injury_moves['Second'] = df_injury_moves.apply(lambda row: 
                                                    str(int(row['PlayStartTime'])), 
                                                    axis=1)
def get_speed(row):
    meters_per_sec = row['dis'] / 0.1
    kph = 3.6 * meters_per_sec
    return kph

df_injury_moves = df_injury_moves.sort_values(by=['GameKey', 'PlayID', 'GSISID', 'PlayStartTime'])
df_injury_moves['kph'] = df_injury_moves.apply(lambda row: get_speed(row), axis=1)
# Get the PR Moves
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves_indexed = df_pr_moves.set_index(['GameKey', 'PlayID', 'Sequence'])

# Compute each distance from the PR
def get_axial_distance(row, axis):
    try:
        coordinates = df_pr_moves_indexed.loc[(row['GameKey'], row['PlayID'], row['Sequence'])]
        return abs(coordinates[axis] - row[axis])
    except:
        return None

def get_distance(row):
    try:
#         coordinates = df_pr_moves.loc[(row['GameKey'], row['PlayID'], row['Sequence'])]
#         return math.sqrt(pow(coordinates['x'] - row['x'], 2) + pow(coordinates['y'] - row['y'], 2))
        return math.sqrt(pow(row['PR_X'], 2) + pow(row['PR_Y'], 2))
    except:
        return None

df_injury_moves['PR_X'] = df_injury_moves.apply(lambda row: 
                                                get_axial_distance(row, 'x'), 
                                                axis=1)
df_injury_moves['PR_Y'] = df_injury_moves.apply(lambda row: 
                                                get_axial_distance(row, 'y'), 
                                                axis=1)
df_injury_moves['PR_Distance'] = df_injury_moves.apply(lambda row: 
                                                       get_distance(row), 
                                                       axis=1)
def get_distance(row):
    return math.sqrt(pow(row['x_Player'] - row['x_Partner'], 2) + 
                     pow(row['y_Player'] - row['y_Partner'], 2))


df_injury_moves_player = df_injury_moves.merge(df_injury[['GameKey', 'PlayID', 'GSISID']],
                                                left_on=['GameKey', 'PlayID', 'GSISID'],
                                                 right_on=['GameKey', 'PlayID', 'GSISID'],
                                                 how='right')
df_injury_moves_partner = df_injury_moves.merge(df_injury[['GameKey', 'PlayID', 
                                                          'Primary_Partner_GSISID']],
                                                left_on=['GameKey', 'PlayID', 'GSISID'],
                                                 right_on=['GameKey', 'PlayID', 'Primary_Partner_GSISID'],
                                                 how='right')

# Put it side-by-side in a row
df_involved_pairs = df_injury_moves_player.merge(df_injury_moves_partner,
                                                left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                suffixes=('_Player', '_Partner'))

# Compute distance of pairs
df_involved_pairs['PairDistance'] = df_involved_pairs.apply(lambda row:
                                                            get_distance(row),
                                                            axis=1)

# Collision point at minimum pair distance
df_min_distances = df_involved_pairs.groupby(["GameKey", "PlayID"])['PairDistance'].idxmin()
df_collision_point = df_involved_pairs.loc[df_min_distances]
events = ['punt', 'punt_received', 'tackle']
events_list = []
for event in events:
    df_event = df_injury_moves[df_injury_moves['Event']==event]
    events_list.append(df_event)

df_events = pd.concat(events_list)

# ANOVA
mod = ols('PlayStartTime ~ Event',
            data=df_events).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)

# Graph
ax = sns.boxplot(x="PlayStartTime", y="Event", data=df_events)
ax.set_title('Timeline')
df_injury_moves_coverage = df_injury_moves[(df_injury_moves['Team']=='Coverage') & 
                                          (df_injury_moves['Role']!='P')]
df_injury_moves_coverage = df_injury_moves_coverage[df_injury_moves_coverage['PlayStartTime'] < 15]

df_injury_moves_coverage = df_injury_moves_coverage.dropna()

# Pearson for linear correlation
output = scipy.stats.pearsonr(df_injury_moves_coverage['PlayStartTime'], 
                    df_injury_moves_coverage['PR_Distance'])
print(output)

# Spearman for non-linear correlation
output = scipy.stats.spearmanr(df_injury_moves_coverage['PlayStartTime'], 
                    df_injury_moves_coverage['PR_Distance'])
print(output)

# Grah
sns.jointplot(x="PlayStartTime", y="PR_Distance", kind='hex', data=df_injury_moves_coverage)
def get_time_diff(row, start_event, end_event):
    try:
        start = df_events_indexed.loc[(row['GameKey'], row['PlayID'], start_event)]['Time'][0]
        end = df_events_indexed.loc[(row['GameKey'], row['PlayID'], end_event)]['Time'][0]
        return (end - start).total_seconds()
    except:
        return None

df_injury['waiting_time'] = df_injury.apply(lambda row: 
                                                   get_time_diff(row, 'punt', 'punt_received'),
                                                  axis=1)
graph_distribution(df_injury['waiting_time'].dropna())
df_injury['waiting_time'].describe()

df_injury_moves_received = df_injury_moves[(df_injury_moves['Event']=='punt_received')]
df_injury_moves_received_nostars = df_injury_moves_received[(df_injury_moves_received['Role']!='PR') &
                                                           (df_injury_moves_received['Role']!='P')]

# Graph
ax = sns.boxplot(x='PR_Distance', 
                 y='Team',
                 data=df_injury_moves_received_nostars)
ax.set_title("Team Distance from PR\nwhen Punt is Received")

# ANOVA
mod = ols('PR_Distance ~ Team',
            data=df_injury_moves_received_nostars).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_received_nostars_coverage = df_injury_moves_received_nostars[df_injury_moves_received_nostars['Team']=='Coverage']
graph_distribution(df_injury_moves_received_nostars_coverage['PR_Distance'].dropna())
df_injury_moves_received_nostars_coverage['PR_Distance'].describe()
graph_distribution(df_injury_moves_received_nostars_coverage['kph'].dropna())
df_injury_moves_received_nostars_coverage['kph'].describe()
def get_time_diff(row, start_event, end_event):
    try:
        start = df_events_indexed.loc[(row['GameKey'], row['PlayID'], start_event)]['Time'][0]
        end = df_events_indexed.loc[(row['GameKey'], row['PlayID'], end_event)]['Time'][0]
        return (end - start).total_seconds()
    except:
        return None

df_injury['reaction_time'] = df_injury.apply(lambda row: 
                                                   get_time_diff(row, 'punt_received', 'tackle'),
                                                  axis=1)
graph_distribution(df_injury['reaction_time'].dropna())
df_injury['reaction_time'].describe()
df_moves_pr = df_injury_moves[df_injury_moves['Role']=='PR']
df_moves_pr_tackle = df_moves_pr[df_moves_pr['Event']=='tackle']

graph_distribution(df_moves_pr_tackle['kph'].dropna())
df_moves_pr_tackle['kph'].describe()
df_injury_moves_received = df_injury_moves[(df_injury_moves['Event']=='tackle')]
df_injury_moves_received_nostars = df_injury_moves_received[(df_injury_moves_received['Role']!='PR') &
                                                           (df_injury_moves_received['Role']!='P')]

# Graph
ax = sns.boxplot(x='kph', 
                 y='Team',
                 data=df_injury_moves_received_nostars)
ax.set_title("Team Speed from PR\nwhen a Tackle Occured")

# ANOVA
mod = ols('kph ~ Team',
            data=df_injury_moves_received_nostars).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_received = df_injury_moves[(df_injury_moves['Event']=='tackle')]
df_injury_moves_received_nostars = df_injury_moves_received[(df_injury_moves_received['Role']!='PR') &
                                                           (df_injury_moves_received['Role']!='P')]

# Graph
ax = sns.boxplot(x='PR_Distance', 
                 y='Team',
                 data=df_injury_moves_received_nostars)
ax.set_title("Team Distance from PR\nwhen a tackle occured")

# ANOVA
mod = ols('PR_Distance ~ Team',
            data=df_injury_moves_received_nostars).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_opponent = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='No']
df_injury_phase2_opponent_tackled = df_injury_phase2_opponent[(df_injury_phase2_opponent['Player_Activity_Derived']=='Tackled') |
                                                             (df_injury_phase2_opponent['Player_Activity_Derived']=='Tackled')]

# Get only the block collisions
df_collision_tackled = df_collision_point.merge(df_injury_phase2_opponent_tackled[['GameKey',
                                                   'PlayID']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                   how='left')

df_collision_tackled_indexed = df_collision_tackled.set_index(['GameKey', 'PlayID', 'PlayStartTime'])

# Get all players during the block collision
df_collision_tackled_moves = df_injury_moves.merge(df_collision_tackled,
                                        left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                        right_on=['GameKey', 'PlayID', 'PlayStartTime'])
df_collision_tackled_moves = df_collision_tackled_moves[(df_collision_tackled_moves['Role']!='PR') &
                                                       (df_collision_tackled_moves['Role']!='P')]
graph_distribution(df_collision_tackled_moves['PR_Distance'])
df_collision_tackled_moves['PR_Distance'].describe()
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_tackle = df_injury_phase2[(df_injury_phase2['Player_Activity_Derived']=='Tackling') |
                                         (df_injury_phase2['Player_Activity_Derived']=='Tackled')]
df_collision_point_info = df_collision_point.merge(df_injury_phase2_tackle[['GameKey', 'PlayID',
                                                                     'Player_Activity_Derived']],
                                                  left_on=['GameKey', 'PlayID'],
                                                  right_on=['GameKey', 'PlayID'],
                                                  how='right')

# Get the PR Moves
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves = df_pr_moves.set_index(['GameKey', 'PlayID', 'Event'])

df_collision_point_indexed = df_collision_point.set_index(['GameKey', 'PlayID'])
df_collision_point_indexed.head()

# Compute each distance from the PR
# TODO: punt_received, fumble, fair catch
def get_collision_distance_from_ball_landing(row):
    try:
        # fair_catch, punt_received,fumble
        ball_location = df_pr_moves.loc[(row['GameKey'], row['PlayID'], 'punt_received')]
        collision_location = df_collision_point_indexed.loc[(row['GameKey'], row['PlayID'])]
        return abs(ball_location['x'][0] - collision_location['x_Player'])
    except:
        return None

df_collision_point_info['Ball_X'] = df_collision_point_info.apply(lambda row: get_collision_distance_from_ball_landing(row), 
                                                       axis=1)

# Graph
graph_distribution(df_collision_point_info['Ball_X'].dropna())
df_collision_point_info['Ball_X'].describe()
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_opponent = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='No']
df_injury_phase2_opponent_tackle = df_injury_phase2_opponent[(df_injury_phase2_opponent['Player_Activity_Derived']=='Tackled') |
                                            (df_injury_phase2_opponent['Player_Activity_Derived']=='Tackling')]

df_moves = df_injury_moves.merge(df_injury_phase2_opponent_tackle[['GameKey',
                                                   'PlayID']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                        how='right')
df_moves_collision = df_moves.merge(df_collision_point[['GameKey',
                                                   'PlayID', 'PlayStartTime']],
                                        left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                        right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                        how='inner')
df_moves_collision.describe()
# df_injury_moves_received = df_injury_moves[(df_injury_moves['Event']=='tackle')]
# df_injury_moves_received_nostars = df_injury_moves_received[(df_injury_moves_received['Role']!='PR') &
#                                                            (df_injury_moves_received['Role']!='P')]

# # Graph
# df_injury_moves_received_nostars.describe()
df_injury_moves_normal = df_injury_moves[(df_injury_moves['PlayStartTime'] < 15) &
                                        (df_injury_moves['kph'] < 50)]

# Ensure that there is only one instance of player per GameKey PlayID
df_injury_speed_max_roles = df_injury_moves_normal.groupby(['GameKey', 'PlayID', 'GSISID', 'Role_Category'])['kph'].max().reset_index()

df_medians = df_injury_speed_max_roles.groupby(['Role_Category'])['kph'].median().reset_index().sort_values('kph')

# Graph
ax = sns.boxplot(x='kph', y='Role_Category', 
            order = df_medians['Role_Category'].values,
            data = df_injury_speed_max_roles)
ax.set_title('Maximum Speeds (kph)')

# Table
df_injury_speed_roles_g = df_injury_speed_max_roles.groupby(['Role_Category']).agg({'kph':
                                                                            ['min', 'max',
                                                                            'median', 'mean']})
df_injury_speed_roles_g.head(40)
# ANOVA
mod = ols('kph ~ Role_Category',
            data=df_injury_speed_max_roles).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_collision_point_gunner = df_collision_point[df_collision_point['Role_Category_Player']=='G']
df_collision_point_gunner[['GameKey', 'PlayID', 'kph_Player', 'kph_Partner']].head(10)
df_collision_point_gunner = df_collision_point[df_collision_point['Role_Category_Partner']=='G']
df_collision_point_gunner[['GameKey', 'PlayID', 'kph_Player', 'kph_Partner']].head(10)
# Get time of min-max points
df_moves_agg = df_injury_moves.groupby(['GameKey','PlayID', 'GSISID']).agg({'kph': ['idxmin', 'idxmax']})
df_moves_agg['kph_Max'] = df_injury_moves.loc[df_moves_agg[('kph', 'idxmax')].values]['PlayStartTime'].values
df_moves_agg['kph_Min'] = df_injury_moves.loc[df_moves_agg[('kph', 'idxmin')].values]['PlayStartTime'].values

df_moves_agg = df_moves_agg.reset_index()

# Get Collision Time
df_moves_agg = df_moves_agg.merge(df_collision_point[['GameKey',
                                                   'PlayID',
                                                   'PlayStartTime']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                        how='right')

# Get Time difference of turning point to collision time
df_moves_agg['kph_Diff'] = df_moves_agg.apply(lambda row:
                                            row['PlayStartTime'] - row[('kph_Max', '')],
                                            axis=1)

# Get activity of involved injury
df_moves_agg = df_moves_agg.merge(df_injury[['GameKey','PlayID', 'GSISID', 'Role_Category',
                                             'Player_Activity_Derived', 
                                             'Primary_Impact_Type',
                                            'Primary_Partner_Activity_Derived']],
                                  left_on=['GameKey', 'PlayID'],
                                  right_on=['GameKey', 'PlayID'],
                                  how='right')

df_grouped = df_moves_agg.groupby(['Role_Category'])['kph_Diff'].median().reset_index()
df_grouped = df_grouped.sort_values('kph_Diff')

# Remove outliers for the graph
df_moves_agg = df_moves_agg[(df_moves_agg['kph_Diff'] > -10) &
                           (df_moves_agg['kph_Diff'] < 30)]
# Graph
ax = sns.boxplot(x='kph_Diff', y='Role_Category', 
                 order = df_grouped['Role_Category'],
            data=df_moves_agg)
ax.set_title('Time Since Peak Speed (seconds)')
df_grouped.head()
# ANOVA
mod = ols('kph_Diff ~ Role_Category',
            data=df_moves_agg).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
# Get only the position at the start of the play
df_injury_moves_normal_start = df_injury_moves_normal[df_injury_moves_normal['PlayStartTime']==0]

df_injury_group = df_injury_moves_normal_start.groupby(['GameKey', 'PlayID', 'GSISID', 'Role_Category'])['PR_Distance'].max().reset_index()
df_injury_group = df_injury_group.sort_values('PR_Distance')

df_medians = df_injury_group.groupby(['Role_Category'])['PR_Distance'].median().reset_index().sort_values('PR_Distance')

# Graph
ax = sns.boxplot(x='PR_Distance', 
            y='Role_Category', 
            order = df_medians['Role_Category'].values,
            data = df_injury_group)
ax.set_title('PR Distance in formation (meters)')
# ANOVA
mod = ols('PR_Distance ~ Role_Category',
            data=df_injury_group).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
width = 53.3

df_injury_moves_normal_start['BorderDistance'] = df_injury_moves_normal_start.apply(lambda row:
                                                                        min(width - row['y'], row['y']),
                                                                       axis=1)

df_grouped = df_injury_moves_normal_start.groupby(['Role_Category'])['BorderDistance'].median().reset_index()
df_grouped = df_grouped.sort_values('BorderDistance')

# Graph
ax = sns.boxplot(x="BorderDistance", y="Role_Category", 
            order=df_grouped['Role_Category'],
            data=df_injury_moves_normal_start)
ax.set_title('Borderline Distance per Role Category')

# ANOVA
mod = ols('BorderDistance ~ Role_Category',
            data=df_injury_moves_normal_start).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_return = df_injury_moves[(df_injury_moves['Team']=='Return') & 
                                          (df_injury_moves['Role']!='PR')]
df_injury_moves_return = df_injury_moves_return[df_injury_moves_return['PlayStartTime'] < 15]

sns.jointplot(x="PlayStartTime", y="PR_Distance", kind='hex', data=df_injury_moves_return)
df_injury_moves_return = df_injury_moves_return.dropna()

# Pearson for linear correlation
output = scipy.stats.pearsonr(df_injury_moves_return['PlayStartTime'], 
                    df_injury_moves_return['PR_Distance'])
print(output)

# Spearman for non-linear correlation
output = scipy.stats.spearmanr(df_injury_moves_return['PlayStartTime'], 
                    df_injury_moves_return['PR_Distance'])
print(output)
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_block = df_injury_phase2[(df_injury_phase2['Player_Activity_Derived']=='Blocking') |
                                         (df_injury_phase2['Player_Activity_Derived']=='Blocked')]
df_collision_point_info = df_collision_point.merge(df_injury_phase2_block[['GameKey', 'PlayID',
                                                                     'Player_Activity_Derived']],
                                                  left_on=['GameKey', 'PlayID'],
                                                  right_on=['GameKey', 'PlayID'],
                                                  how='right')

# Graph
graph_distribution(df_collision_point_info['PR_Distance_Player'].dropna())
df_collision_point_info['PR_Distance_Player'].describe()
# Get the PR Moves
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves = df_pr_moves.set_index(['GameKey', 'PlayID', 'Event'])

df_collision_point_indexed = df_collision_point.set_index(['GameKey', 'PlayID'])
df_collision_point_indexed.head()

# Compute each distance from the PR
# TODO: punt_received, fumble, fair catch
def get_collision_distance_from_ball_landing(row):
    try:
        # fair_catch, punt_received,fumble
        ball_location = df_pr_moves.loc[(row['GameKey'], row['PlayID'], 'punt_received')]
        collision_location = df_collision_point_indexed.loc[(row['GameKey'], row['PlayID'])]
        return abs(ball_location['x'][0] - collision_location['x_Player'])
    except:
        return None

df_collision_point_info['Ball_X'] = df_collision_point_info.apply(lambda row: get_collision_distance_from_ball_landing(row), 
                                                       axis=1)

# Graph
graph_distribution(df_collision_point_info['Ball_X'].dropna())
df_collision_point_info['Ball_X'].describe()
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='PR') & (df_injury_moves['Role']!='P')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]

df_injury_moves_nostars = df_injury_moves_nostars.dropna(subset=['PR_Distance'])
df_injury_moves_nostars['Second'] = df_injury_moves_nostars['Second'].astype(int)

ax = sns.boxplot(x="Second", y="PR_Distance", 
            hue="Team",
            data=df_injury_moves_nostars)
ax.set_title('PR Distance Distribution\n(meters vs seconds)')
# ANOVA
for second in range(15):
    df_injury_moves_second = df_injury_moves_nostars[df_injury_moves_nostars['Second'] == second]
    print("Second " + str(second) + ':')
    mod = ols('PR_Distance ~ Team',
                data=df_injury_moves_second).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    print()
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='PR') & (df_injury_moves['Role']!='P')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]
df_injury_moves_nostars_gameplay = df_injury_moves_nostars[(df_injury_moves_nostars['GameKey']==5) &
                                                          (df_injury_moves_nostars['PlayID']==3129)]
df_injury_moves_nostars_gameplay['Second'] = df_injury_moves_nostars_gameplay['Second'].astype(int)

ax = sns.boxplot(x="Second", y="PR_X", 
            hue="Team",
            data=df_injury_moves_nostars_gameplay)
ax.set_title("PR's X Distance Distribution\n(meters vs seconds)")
# Filter
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_opponent = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='No']

df_moves = df_injury_moves.merge(df_injury_phase2_opponent[['GameKey',
                                                   'PlayID']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                        how='right')

# Get time of min-max points
df_moves_agg = df_moves.groupby(['GameKey','PlayID', 'GSISID']).agg({'x': ['idxmin', 'idxmax']})
df_moves_agg['X_Max'] = df_moves.loc[df_moves_agg[('x', 'idxmax')].values]['PlayStartTime'].values
df_moves_agg['X_Min'] = df_moves.loc[df_moves_agg[('x', 'idxmin')].values]['PlayStartTime'].values

df_moves_agg = df_moves_agg.reset_index()

# Get Collision Time
df_moves_agg = df_moves_agg.merge(df_collision_point[['GameKey',
                                                   'PlayID',
                                                   'PlayStartTime']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                        how='right')


# Get Time difference of turning point to collision time
df_moves_agg['X_Diff'] = df_moves_agg.apply(lambda row:
#                                             get_facing(row),
                                            min(abs(row['PlayStartTime'] - row[('X_Max', '')]),
                                                abs(row['PlayStartTime'] - row[('X_Min', '')])),
                                            axis=1)

# Get activity of involved injury
df_moves_agg = df_moves_agg.merge(df_injury[['GameKey','PlayID',
                                             'Player_Activity_Derived']],
                                  left_on=['GameKey', 'PlayID'],
                                  right_on=['GameKey', 'PlayID'],
                                  how='right')

# Graph
ax = sns.boxplot(x='X_Diff', y='Player_Activity_Derived', 
            data=df_moves_agg)
ax.set_title('Deceleration Time Before Collision (seconds)')
df_moves_agg.groupby(['Player_Activity_Derived'])['X_Diff'].median().reset_index().head()
# ANOVA
mod = ols('X_Diff ~ Player_Activity_Derived',
            data=df_moves_agg).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
aov_table.head()
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_opponent = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='No']
df_injury_phase2_opponent_block = df_injury_phase2_opponent[(df_injury_phase2_opponent['Player_Activity_Derived']=='Blocked') |
                                            (df_injury_phase2_opponent['Player_Activity_Derived']=='Blocking')]

# Get only the block collisions
df_collision_block = df_collision_point.merge(df_injury_phase2_opponent_block[['GameKey',
                                                   'PlayID']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                   how='left')

df_collision_block_indexed = df_collision_block.set_index(['GameKey', 'PlayID', 'PlayStartTime'])

# Get all players during the block collision
df_collision_block_moves = df_injury_moves.merge(df_collision_block,
                                        left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                        right_on=['GameKey', 'PlayID', 'PlayStartTime'])

# Compute each distance from the PR
def get_player_axial_distance(row, axis):
    try:
        coordinates = df_collision_block_indexed.loc[(row['GameKey'], row['PlayID'], row['PlayStartTime'])]
        index = axis+'_Player'
        return abs(coordinates[index] - row[axis])
    except:
        return None

def get_player_distance(row):
    try:
        return math.sqrt(pow(row['Player_X'], 2) + pow(row['Player_Y'], 2))
    except:
        return None

df_collision_block_moves['Player_X'] = df_collision_block_moves.apply(lambda row: 
                                                                     get_player_axial_distance(row, 'x'),
                                                                     axis=1)
df_collision_block_moves['Player_Y'] = df_collision_block_moves.apply(lambda row: 
                                                                     get_player_axial_distance(row, 'y'),
                                                                     axis=1)
df_collision_block_moves['Player_Distance']= df_collision_block_moves.apply(lambda row: 
                                                                     get_player_distance(row),
                                                                     axis=1)

# Graph
graph_distribution(df_collision_block_moves['Player_Distance'])
df_collision_block_moves['Player_Distance'].describe()
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_opponent = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='No']
df_injury_phase2_opponent_block = df_injury_phase2_opponent[(df_injury_phase2_opponent['Player_Activity_Derived']=='Blocked') |
                                            (df_injury_phase2_opponent['Player_Activity_Derived']=='Blocking')]

# Get only the moves for block injuries
df_moves = df_injury_moves.merge(df_injury_phase2_opponent_block[['GameKey',
                                                   'PlayID']],
                                        left_on=['GameKey', 'PlayID'],
                                        right_on=['GameKey', 'PlayID'],
                                        how='right')

# Get only the moves during the collision time
df_moves_collision = df_moves.merge(df_collision_point[['GameKey',
                                                   'PlayID', 'PlayStartTime']],
                                        left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                        right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                   how='inner')
df_moves_collision.describe()
# Ensure there is only one instance per person
df_injury_max_speeds = df_injury_moves_nostars.groupby(['GameKey', 'PlayID', 'GSISID', 'Team'])['kph'].max().reset_index()

# Graph
ax = sns.boxplot(x='kph', y='Team', data = df_injury_max_speeds)
ax.set_title('Maximum Team Speed (kph)')
ax.set(xlabel='kph', ylabel='Team')

df_injury_max_speeds_g = df_injury_max_speeds.groupby(['Team']).agg({'kph':
                                                                            ['min', 'max',
                                                                            'median', 'mean']})
df_injury_max_speeds_g.head(40)
mod = ols('kph ~ Team',
            data=df_injury_max_speeds).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
# Ensure there is only one instance per person
df_injury_max_distances = df_injury_moves_nostars.groupby(['GameKey', 'PlayID', 'GSISID', 'Team'])['PR_Distance'].max().reset_index()

# Graph
ax = sns.boxplot(x='PR_Distance', y='Team', data = df_injury_max_distances)
ax.set_title('Maximum Team-PR Distance (meters)')
ax.set(xlabel='PR Distance', ylabel='Team')

df_injury_max_distances_g = df_injury_max_distances.groupby(['Team']).agg({'PR_Distance':
                                                                            ['min', 'max',
                                                                            'median', 'mean']})
df_injury_max_distances_g.head(40)
mod = ols('PR_Distance ~ Team',
            data=df_injury_max_distances).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='PR') & (df_injury_moves['Role']!='P')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 30]

# Ensure there is only one instance per person
df_injury_min_distances = df_injury_moves_nostars.groupby(['GameKey', 'PlayID', 'GSISID', 'Team'])['PR_Distance'].min().reset_index()

# Graph
ax = sns.boxplot(x='PR_Distance', y='Team', data = df_injury_min_distances)
ax.set_title('Minimum Team-PR Distances (meters)')
ax.set(xlabel='PR Distance', ylabel='Team')

df_injury_min_distances_g = df_injury_min_distances.groupby(['Team']).agg({'PR_Distance':
                                                                            ['min', 'max',
                                                                            'median', 'mean']})
df_injury_min_distances_g.head(40)
mod = ols('PR_Distance ~ Team',
            data=df_injury_min_distances).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='P') & 
                                          (df_injury_moves['Role']!='PR')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]
df_injury_moves_nostars = df_injury_moves_nostars.dropna()

# Pearson for linear correlation
output = scipy.stats.pearsonr(df_injury_moves_nostars['PlayStartTime'], 
                    df_injury_moves_nostars['PR_Distance'])
print(output)

# Spearman for non-linear correlation
output = scipy.stats.spearmanr(df_injury_moves_nostars['PlayStartTime'], 
                    df_injury_moves_nostars['PR_Distance'])
print(output)

sns.jointplot(x="PlayStartTime", y="PR_Distance", kind='hex', data=df_injury_moves_nostars)
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='PR') & (df_injury_moves['Role']!='P')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]

df_injury_moves_nostars = df_injury_moves_nostars.dropna(subset=['PR_Distance'])
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PR_Distance'] < 50]
df_injury_moves_nostars['Second'] = df_injury_moves_nostars['Second'].astype(int)

# Graph
ax = sns.boxplot(x="Second", y="PR_Distance", 
            hue="Team",
            data=df_injury_moves_nostars)
ax.set_title('PR Distance Distribution\n(meters vs seconds)')

# ANOVA
for second in range(15):
    df_injury_moves_second = df_injury_moves_nostars[df_injury_moves_nostars['Second'] == second]
    print("Second " + str(second) + ':')
    mod = ols('PR_Distance ~ Team',
                data=df_injury_moves_second).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    print()
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='P') & 
                                          (df_injury_moves['Role']!='PR')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]

# Pearson for linear correlation
output = scipy.stats.pearsonr(df_injury_moves_nostars['PlayStartTime'], 
                    df_injury_moves_nostars['kph'])
print(output)

# Spearman for non-linear correlation
output = scipy.stats.spearmanr(df_injury_moves_nostars['PlayStartTime'], 
                    df_injury_moves_nostars['kph'])
print(output)

sns.jointplot(x="PlayStartTime", y="kph", kind='hex', data=df_injury_moves_nostars)
df_injury_moves_nostars = df_injury_moves[(df_injury_moves['Role']!='PR') & (df_injury_moves['Role']!='P')]
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['PlayStartTime'] < 15]

df_injury_moves_nostars = df_injury_moves_nostars.dropna(subset=['kph'])
df_injury_moves_nostars = df_injury_moves_nostars[df_injury_moves_nostars['kph'] < 50]
df_injury_moves_nostars['Second'] = df_injury_moves_nostars['Second'].astype(int)

ax = sns.boxplot(x="Second", y="kph", 
            hue="Team",
            data=df_injury_moves_nostars)
ax.set_title('Speed Distribution\n(kph vs seconds)')
df_injury_moves = df_injury_moves.sort_values(by=['GameKey', 'PlayID', 'GSISID', 'PlayStartTime'])

# Delta: Change in movements
df_injury_moves['dx'] = df_injury_moves['x'] - df_injury_moves.groupby(['GameKey', 'PlayID', 'GSISID'])['x'].shift(1)
df_injury_moves['dy'] = df_injury_moves['y'] - df_injury_moves.groupby(['GameKey', 'PlayID', 'GSISID'])['y'].shift(1)
df_injury_moves['dt'] = df_injury_moves['PlayStartTime'] - df_injury_moves.groupby(['GameKey', 'PlayID', 'GSISID'])['PlayStartTime'].shift(1)

# Velocity: Convert meters per second to kph
df_injury_moves['vx'] = 3.6 * (df_injury_moves['dx']/0.1)
df_injury_moves['vy'] = 3.6 * (df_injury_moves['dy']/0.1)

# Velocity: Convert to absolute value speed
df_injury_moves['vx'] = df_injury_moves.apply(lambda row: 
                                              abs(row['vx']),
                                              axis = 1)
df_injury_moves['vy'] = df_injury_moves.apply(lambda row: 
                                              abs(row['vy']),
                                              axis = 1)

# Put the velocities into one column
df_velocities = pd.melt(df_injury_moves, 
                        id_vars=['GameKey', 'PlayID', 'GSISID', 'PlayStartTime'], 
                        value_vars=['vx', 'vy'])

# Get the maximum speed of player per gameplay
df_v_max = df_velocities.groupby(['GameKey', 'PlayID', 'GSISID', 'variable'])['value'].max().reset_index()

# Clean data that is faster than maximum human speed
df_v_max_limit = df_v_max[df_v_max['value'] < 50]

# Graph
ax = sns.boxplot(y="variable", x="value", data=df_v_max_limit)
ax.set_title('Axial Velocity (kph)')

# ANOVA
mod = ols('value ~ variable',
            data=df_v_max).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
df_injury_moves_details = df_injury_moves.merge(df_injury[['GameKey', 'PlayID', 'Player_Activity_Derived']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 suffixes=('', '_Player'))
df_collision_point_injury = df_collision_point.merge(df_injury_moves_details[['GameKey', 'PlayID', 
                                                                              'Player_Activity_Derived',
                                                               'kph', 'PR_Distance', 'PlayStartTime',
                                                                              'Team']],
                                                left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                 right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                    how='left')
# Graph
ax = sns.boxplot(x='kph', 
                 y='Player_Activity_Derived',
                 hue='Team',
                 data=df_collision_point_injury)
ax.set_title("Players Speed\nwhen Collision Occured")

# ANOVA
mod = ols('kph ~ Player_Activity_Derived',
            data=df_collision_point_injury).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print('Activity Variance:')
print(aov_table)
print()

# ANOVA
print('Coverage vs Return:')
activities=['Tackling', 'Blocked', 'Blocking', 'Tackled']
for activity in activities:
    df_activity = df_collision_point_injury[df_collision_point_injury['Player_Activity_Derived']==activity]
    mod = ols('kph ~ Team',
                data=df_activity).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(activity + ':')
    print(aov_table)
    print()
df_injury_moves_details = df_injury_moves.merge(df_injury[['GameKey', 'PlayID', 'Player_Activity_Derived']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 suffixes=('', '_Player'))
df_collision_point_injury = df_collision_point.merge(df_injury_moves_details[['GameKey', 'PlayID', 
                                                                              'Player_Activity_Derived',
                                                               'kph', 'PR_Distance', 'PlayStartTime',
                                                                              'Team']],
                                                left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                 right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                    how='left')
# Graph
ax = sns.boxplot(x='PR_Distance', 
                 y='Player_Activity_Derived',
                 hue='Team',
                 data=df_collision_point_injury)
ax.set_title("Players PR Distance\nwhen Collision Occured")

# ANOVA
mod = ols('PR_Distance ~ Player_Activity_Derived',
            data=df_collision_point_injury).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print('Activity Variance:')
print(aov_table)
print()

# ANOVA
print('Coverage vs Return:')
activities=['Tackling', 'Blocked', 'Blocking', 'Tackled']
for activity in activities:
    df_activity = df_collision_point_injury[df_collision_point_injury['Player_Activity_Derived']==activity]
    mod = ols('PR_Distance ~ Team',
                data=df_activity).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(activity + ':')
    print(aov_table)
    print()
# Get the PR Moves
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves = df_pr_moves.set_index(['GameKey', 'PlayID', 'Event'])

df_collision_point_indexed = df_collision_point.set_index(['GameKey', 'PlayID'])
df_collision_point_indexed.head()

# Compute each distance from the PR
# TODO: punt_received, fumble, fair catch
def get_collision_distance_from_ball_landing(row):
    try:
        # fair_catch, punt_received,fumble
        ball_location = df_pr_moves.loc[(row['GameKey'], row['PlayID'], 'punt_received')]
        collision_location = df_collision_point_indexed.loc[(row['GameKey'], row['PlayID'])]
        return abs(ball_location['x'][0] - collision_location['x_Player'])
    except:
        return None

df_collision_point['Ball_X'] = df_collision_point.apply(lambda row: get_collision_distance_from_ball_landing(row), 
                                                       axis=1)
df_collision_point['Ball_X_yards'] = 1.09361 * df_collision_point['Ball_X']

# Graph
graph_distribution(df_collision_point['Ball_X_yards'].dropna())
df_collision_point['Ball_X_yards'].describe()
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves['Event'].value_counts()