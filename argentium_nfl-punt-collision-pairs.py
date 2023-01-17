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
df_play_info = pd.read_csv('../input/play_information.csv')
df_punt_role = pd.read_csv('../input/play_player_role_data.csv')
df_injury = pd.read_csv('../input/video_review.csv')
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

# Add the corresponding side of their role
def set_team(role):
    for team in team_positions.keys():
        if str(role) in team_positions[team]:
            return str(team)
    return None

def set_role_category(role):
    for category in role_categories.keys():
        if str(role) in role_categories[category]:
            return str(category)
    return None

df_punt_role['Team'] = df_punt_role.apply(lambda row: set_team(row['Role']), axis=1)
df_punt_role['Role_Category'] = df_punt_role.apply(lambda row: set_role_category(row['Role']), 
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
columns = ['GameKey', 'PlayID']
for col in columns:
    df_injury[col] = df_injury[col].astype(ngs[col].dtype)
    df_punt_role[col] = df_punt_role[col].astype(ngs[col].dtype)

df_injury['GSISID'] = df_injury['GSISID'].astype(ngs['GSISID'].dtype)
df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].astype(ngs['GSISID'].dtype)

# Get Only Games with Injuries
df_injury_moves = ngs.merge(df_injury[['GameKey', 'PlayID']],
                            left_on=['GameKey', 'PlayID'],
                            right_on=['GameKey', 'PlayID'])

# Create Gameplay ID
df_injury_moves['Gameplay'] = df_injury_moves.apply(lambda row: 
                                            str(row['GameKey']) + '_' + 
                                            str(row['PlayID']),
                                           axis=1)

# Delete the dataframe list to release memory
del ngs
gc.collect()
# I added the role for easier categorization
df_injury_moves = df_injury_moves.merge(df_punt_role,
                                  left_on=['GameKey', 'PlayID', 'GSISID'],
                                  right_on=['GameKey', 'PlayID', 'GSISID'],
                                 suffixes=('', '_Injury'))
df_injury_moves['x'] = 0.9144 * df_injury_moves['x']
df_injury_moves['y'] = 0.9144 * df_injury_moves['y']
df_injury_moves['dis'] = 0.9144 * df_injury_moves['dis']
# Get only events
df_events = df_injury_moves.dropna(subset=['Event'])
df_events_indexed = df_events.set_index(['GameKey', 'PlayID', 'Event'])

# Get Aggregates
df_injury_moves_gameplay = df_injury_moves.groupby(['GameKey', 'PlayID']).agg({'Time': ['min', 'max'],
                                                                               'dis': ['min', 'median', 'max', 'sum'],
                                                                               'dir': ['min', 'median', 'max'],
                                                                               'o': ['min', 'median', 'max']})

# Get start time by subtracting the time at play start in the specified GameKey, PlayID
def get_start_time(row):
    start = None
    try:
        start = df_events_indexed.loc[(row['GameKey'], row['PlayID'], 'ball_snap')]['Time'][0]
    finally:
        if start==None:
            return None
        else:
            end = row['Time']
            return (end - start).total_seconds()

df_injury_moves['PlayStartTime'] = df_injury_moves.apply(lambda row: 
                                                         get_start_time(row),
                                                         axis=1)
# Remove data beyond the start of the punt play
df_injury_moves = df_injury_moves[df_injury_moves['PlayStartTime'] >= 0]
def get_speed(row):
    meters_per_sec = row['dis'] / 0.1
    kph = 3.6 * meters_per_sec
    return kph

df_injury_moves = df_injury_moves.sort_values(by=['GameKey', 'PlayID', 'GSISID', 'PlayStartTime'])
df_injury_moves['kph'] = df_injury_moves.apply(lambda row: get_speed(row), axis=1)
# Get the PR Moves
df_pr_moves = df_injury_moves[df_injury_moves['Role']=='PR']
df_pr_moves = df_pr_moves.set_index(['GameKey', 'PlayID', 'PlayStartTime'])

# Compute each distance from the PR
def get_distance(row):
    try:
        coordinates = df_pr_moves.loc[(row['GameKey'], row['PlayID'], row['PlayStartTime'])]
        return math.sqrt(pow(coordinates['x'] - row['x'], 2) + pow(coordinates['y'] - row['y'], 2))
    except:
        return None

df_injury_moves['PR_Distance'] = df_injury_moves.apply(lambda row: get_distance(row), 
                                                       axis=1)
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

# Put it all in a list
df_injury_moves_player['Involvement'] = 'Player'
df_injury_moves_partner['Involvement'] = 'Partner'
df_injury_moves_involved = pd.concat([df_injury_moves_player, df_injury_moves_partner])
# Put it side-by-side in a row
df_involved_pairs = df_injury_moves_player.merge(df_injury_moves_partner,
                                                left_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                right_on=['GameKey', 'PlayID', 'PlayStartTime'],
                                                suffixes=('_Player', '_Partner'))

# Compute Pair Distance
def get_distance(row):
    return math.sqrt(pow(row['x_Player'] - row['x_Partner'], 2) + 
                     pow(row['y_Player'] - row['y_Partner'], 2))


df_involved_pairs['PairDistance'] = df_involved_pairs.apply(lambda row:
                                                            get_distance(row),
                                                            axis=1)
# Create Gameplay ID
df_involved_pairs['Gameplay'] = df_involved_pairs.apply(lambda row: 
                                            str(row['GameKey']) + '_' + 
                                            str(row['PlayID']),
                                           axis=1)
df_min_distances = df_involved_pairs.groupby(["GameKey", "PlayID"])['PairDistance'].idxmin()
df_collision_point = df_involved_pairs.loc[df_min_distances]
df_injury_phase1 = df_injury[df_injury['Phase']==1]
df_injury_phase1_blocking = df_injury_phase1[df_injury_phase1['Player_Activity_Derived']=='Blocking']
df_pair = df_injury_moves_involved.merge(df_injury_phase1_blocking,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", height=5) 
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", height=5) 
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
df_injury_phase2 = df_injury[df_injury['Phase']==2]
df_injury_phase2_tackled = df_injury_phase2[df_injury_phase2['Player_Activity_Derived']=='Tackled']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_tackled,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", height=5) 
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", height=5) 
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
df_injury_phase2_tackling = df_injury_phase2[df_injury_phase2['Player_Activity_Derived']=='Tackling']
df_injury_phase2_tackling_opponent = df_injury_phase2_tackling[df_injury_phase2_tackling['Friendly_Fire']=='No']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_tackling_opponent,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
df_pair = df_collision_point.merge(df_injury_phase2_tackling_opponent[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_gunner = df_injury[(df_injury['Role_Category']=='G') |
                                 (df_injury['Role_Category_Partner']=='G')]

df_injury_moves_paired = df_injury_gunner.merge(df_injury_moves_involved,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='left')

df_injury_moves_paired = df_injury_moves_paired[df_injury_moves_paired['PlayStartTime']<15]
g = sns.FacetGrid(df_injury_moves_paired, hue='Involvement', col="Gameplay", col_wrap=4, height=5)
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
df_injury_moves_paired_pivot = df_injury_gunner.merge(df_involved_pairs,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='left')

g = sns.FacetGrid(df_injury_moves_paired, hue='Involvement', col="Gameplay", col_wrap=4, height=5)
g = g.map(plt.plot, "PlayStartTime", "kph", marker=".")
df_injury_gunner = df_injury[(df_injury['Role_Category']=='G')]

df_injury_moves_paired = df_injury_gunner.merge(df_injury_moves_involved,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='left')
df_injury_moves_paired = df_injury_moves_paired[df_injury_moves_paired['PlayStartTime']<15]
g = sns.FacetGrid(df_injury_moves_paired, hue='Involvement', col="Gameplay", col_wrap=3, height=5)
g = g.map(plt.plot, "PlayStartTime", "PR_Distance", marker=".")
g = sns.FacetGrid(df_injury_moves_paired, hue='Involvement', col="Gameplay", col_wrap=3, height=5)
g = g.map(plt.plot, "PlayStartTime", "x", marker=".")
df_injury_gunner_player = df_injury[(df_injury['Role_Category']=='G')]
df_min_distances_gunner = df_collision_point.merge(df_injury_gunner_player[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_min_distances_gunner[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_gunner = df_injury[(df_injury['Role_Category_Partner']=='G')]

df_injury_moves_paired = df_injury_gunner.merge(df_injury_moves_involved,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='left')
df_injury_moves_paired = df_injury_moves_paired[df_injury_moves_paired['PlayStartTime']<15]

g = sns.FacetGrid(df_injury_moves_paired, hue='Involvement', col="Gameplay", col_wrap=4, height=5)
g = g.map(plt.plot, "PlayStartTime", "PR_Distance", marker=".")
df_injury_gunner_partner = df_injury[(df_injury['Role_Category_Partner']=='G')]
df_min_distances_gunner = df_collision_point.merge(df_injury_gunner_partner[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_min_distances_gunner[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_phase2_blocked = df_injury_phase2[(df_injury_phase2['Player_Activity_Derived']=='Blocked')]
df_injury_phase2_blocked_opponent = df_injury_phase2_blocked[df_injury_phase2_blocked['Friendly_Fire']=='No']

df_pair = df_injury_moves_involved.merge(df_injury_phase2_blocked_opponent,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5)
g = g.map(plt.plot, "PlayStartTime", "PR_Distance", marker=".")
df_pair_distance = df_involved_pairs.merge(df_injury_phase2_blocked_opponent,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair_distance = df_pair_distance[df_pair_distance['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair_distance, col='Gameplay', col_wrap=5, height=5) 
g = g.map(plt.scatter, 'PlayStartTime', 'PairDistance', marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.plot, "PlayStartTime", "kph", marker=".")
df_min_distances_pair = df_collision_point.merge(df_injury_phase2_blocked_opponent[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_min_distances_pair[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_phase2_blocking = df_injury_phase2[df_injury_phase2['Player_Activity_Derived']=='Blocking']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_blocking,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5)
g = g.map(plt.plot, "PlayStartTime", "PR_Distance", marker=".")
df_pair_distance = df_involved_pairs.merge(df_injury_phase2_blocking, 
                                  left_on=['GameKey', 'PlayID'],
                                 right_on=['GameKey', 'PlayID'],
                                 how='right')
df_pair_distance = df_pair_distance[df_pair_distance['PlayStartTime'] < 15]

# Graph
g = sns.FacetGrid(df_pair_distance, col='Gameplay', col_wrap=5, height=5) 
g = g.map(plt.plot, 'PlayStartTime', 'PairDistance', marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=5, height=5) 
g = g.map(plt.plot, "PlayStartTime", "kph", marker=".")
df_min_distances_pair = df_collision_point.merge(df_injury_phase2_blocking[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_min_distances_pair[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_phase2_friend = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='Yes']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_friend, 
                                  left_on=['GameKey', 'PlayID'],
                                 right_on=['GameKey', 'PlayID'],
                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=3, height=5)
g = g.map(plt.scatter, "PlayStartTime", "x", marker=".")
df_injury_phase2_tackling_friend = df_injury_phase2[df_injury_phase2['Friendly_Fire']=='Yes']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_tackling_friend,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=3, height=5)
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=3, height=5) 
g = g.map(plt.plot, "PlayStartTime", "kph", marker=".")
df_collision_time = df_collision_point.merge(df_injury_phase2_tackling_friend[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_collision_time[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)
df_injury_phase2_blocked = df_injury_phase2[df_injury_phase2['Player_Activity_Derived']=='Blocked']
df_injury_phase2_blocked_friend = df_injury_phase2_blocked[df_injury_phase2_blocked['Friendly_Fire']=='Yes']
df_pair = df_injury_moves_involved.merge(df_injury_phase2_blocked_friend,
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair = df_pair[df_pair['PlayStartTime']<15]

g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=3, height=5)
g = g.map(plt.scatter, "PlayStartTime", "y", marker=".")
df_pair_distance = df_involved_pairs.merge(df_injury_phase2_blocked_friend, 
                                  left_on=['GameKey', 'PlayID'],
                                 right_on=['GameKey', 'PlayID'],
                                 how='right')
df_pair_distance = df_pair_distance[df_pair_distance['PlayStartTime'] < 15]

# Graph
g = sns.FacetGrid(df_pair_distance, col='Gameplay', col_wrap=5, height=5) 
g = g.map(plt.plot, 'PlayStartTime', 'PairDistance', marker=".")
# Graph
g = sns.FacetGrid(df_pair, hue='Involvement', col="Gameplay", col_wrap=3, height=5) 
g = g.map(plt.plot, "PlayStartTime", "kph", marker=".")
df_pair = df_collision_point.merge(df_injury_phase2_blocked_friend[['GameKey', 'PlayID']],
                                                left_on=['GameKey', 'PlayID'],
                                                 right_on=['GameKey', 'PlayID'],
                                                 how='right')
df_pair[['GameKey', 'PlayID', 'PlayStartTime', 'kph_Player', 'kph_Partner']].head(40)