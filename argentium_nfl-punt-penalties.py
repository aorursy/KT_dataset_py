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
df_injury = pd.read_csv('../input/video_review.csv')
df_punt_role = pd.read_csv('../input/play_player_role_data.csv')
team_positions = {'Return': 
                  ['VR', 'VRo', 'VRi', 
                   'PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6',
                   'PLR', 'PLR1', 'PLR2', 'PLR3',
                   'PR', 'PFB', 'PDM', 'VL', 'VLo', 'VLi',
                   'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6',
                   'PLL', 'PLL1', 'PLL2', 'PLL3', 'PLLi'],
     'Coverage': ['GR', 'GRo', 'GRi',
                'PRG', 'PRT', 'PRW',
                'PPR', 'PPRo', 'PPRi', 'P', 'PC', 'PLS',
                  'GL', 'GLo', 'GLi',
               'PLW', 'PLT', 'PLG',
               'PPL', 'PPLo', 'PPLi']}

# Add the corresponding side of their role
def set_team(role):
    for team in team_positions.keys():
        if str(role) in team_positions[team]:
            return str(team)
    return None

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

df_punt_role['Team'] = df_punt_role.apply(lambda row: set_team(row['Role']), 
                                                axis=1)

# Clean
df_injury['Primary_Partner_GSISID'] = df_injury.apply(lambda row: 
                                                                  row['Primary_Partner_GSISID'] 
                                                                  if (row['Primary_Partner_GSISID'] != 'Unclear')
                                                                 else 0,
                                                                 axis=1)
df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].fillna(0)
df_injury['Primary_Partner_GSISID'] = df_injury['Primary_Partner_GSISID'].astype(int)

# Identify roles for player and partner
df_injury = df_injury.merge(df_punt_role, 
                                  left_on=['GameKey', 'PlayID', 'GSISID'],
                                 right_on=['GameKey', 'PlayID', 'GSISID'],
                                 how='left')

df_injury = df_injury.merge(df_punt_role, 
                                 suffixes=('', '_Partner'),
                                  left_on=['GameKey', 'PlayID', 'Primary_Partner_GSISID'],
                                 right_on=['GameKey', 'PlayID', 'GSISID'],
                                 how='left')
df_injury = df_injury.drop(['GSISID_Partner'], axis=1)
df_injury['Phase'] = df_injury.apply(lambda row: 
                                                set_phase(row), 
                                                axis=1)
results = [ 
           'downed',
           'fair catch', 
           'Touchback'
          ]

def get_result(row):
    match = re.search('.* for .*', row['PlayDescription']) # [A-Z]+? [0-9]+ # (to .*?)
    if match:
        return 'return'

    for result in results:
        if result in row['PlayDescription']:
            return result

    return 'others'

df_play_info['Result'] = df_play_info.apply(lambda row: get_result(row), axis=1)
print(len(df_play_info))
def get_penalty(row):
    if 'PENALTY' in row['PlayDescription']:
        return 'Yes'
    else:
        return 'No'

df_play_info['Penalty'] = df_play_info.apply(lambda row: get_penalty(row), axis=1)

# Show info
ax = sns.countplot(x="Penalty", 
                   order=df_play_info['Penalty'].value_counts().index,
                   data=df_play_info)
ax.set_title('Penalty Count in Punt Plays')
df_play_info['Penalty'].value_counts()
def identify_penalty(row):
    penalties = ['Offensive Holding', 
                 'Taunting',
                 'Disqualification',
                 'Running Into the Kicker',
                 'Interference',
                 'Unnecessary Roughness',
                 'Face Mask', 
                 'Neutral Zone Infraction',
                 'Horse Collar Tackle',
                 'Ineligible Downfield Kick',
                 'Player Out of Bounds on Punt',
                 'Defensive 12 On-field',
                 'Offensive 12 On-field',
                 'Chop Block',
                 'Illegal Block Above the Waist', 
                 'Illegal Blindside Block', 
                 'Illegal Touch', 
                 'Illegal Use of Hands', 
                 'Illegal Substitution',
                 'Illegal Formation',
                 'Illegal Motion',
                 'Illegal Shift',
                 'Clipping',
                 'Tripping',
                 'Invalid Fair Catch Signal',
                 'Delay of Game',
                 'Defensive Holding',
                 'Roughing the Kicker',
                 'Unsportsmanlike Conduct',
                 'Defensive Offside',
                'False Start']
    for penalty in penalties:
        if penalty in row['PlayDescription']:
            return penalty
    return 'Unknown'

def has_penalty(row):
    if 'PENALTY' in row['PlayDescription']:
        return 'Yes'
    return 'No'

df_play_info['Penalty'] = df_play_info.apply(lambda row: has_penalty(row), axis=1)
df_play_info['PenaltyID'] = df_play_info.apply(lambda row: identify_penalty(row), axis=1)
df_play_info_penalty = df_play_info[df_play_info['Penalty']=='Yes']

# Show info
sns.set(rc={'figure.figsize':(8,9)})
ax = sns.countplot(y="PenaltyID", 
                   order=df_play_info_penalty['PenaltyID'].value_counts().index,
                   data=df_play_info_penalty)
ax.set_title('Penalty Frequency')
df_injury_plays = df_injury.merge(df_play_info, 
                                  left_on=['GameKey', 'PlayID'],
                                 right_on=['GameKey', 'PlayID'],
                                 how='left')
df_injury_plays['Penalty'] = df_injury_plays.apply(lambda row: get_penalty(row), axis=1)
df_injury_penalty = df_injury_plays[df_injury_plays['Penalty']=='Yes']
print('Penalty count: ' + str(len(df_injury_penalty)))
# Graph
sns.set(rc={'figure.figsize':(6,4)})
df_injury_penalty['PenaltyID'].value_counts()
df_total = df_play_info['PenaltyID'].value_counts().reset_index(name='total')
df_injury_total = df_injury_penalty['PenaltyID'].value_counts().reset_index(name='injured')

# df_total.head()
df_merged = df_total.merge(df_injury_total, how='right')
df_merged = df_merged.fillna(0)
df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

ax = sns.barplot(x='ratio', y='index', 
#             order=df_merged['ratio'].value_counts().index,
            data=df_merged)
ax.set_title('Penalty Injury Risk')
df_cross_injured = pd.crosstab(df_injury_penalty['PenaltyID'], df_injury_penalty['Result'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Penalty-Event')
df_cross_injured = pd.crosstab(df_injury_penalty['Result'], df_injury_penalty['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity')
df_cross_injured = pd.crosstab(df_injury_penalty['Phase'], df_injury_penalty['Penalty'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Phase-Penalty Frequency')
df_injury_penalty_phase1 = df_injury_penalty[df_injury_penalty['Phase']==1]
df_injury_penalty_phase1['PenaltyID'].value_counts()
df_cross_injured = pd.crosstab(df_injury_penalty_phase1['Result'], df_injury_penalty_phase1['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity Before the Punt\n(With Penalties)')
df_injury_plays_phase1 = df_injury_penalty[df_injury_penalty['Phase']==1]
df_cross_injured = pd.crosstab(df_injury_plays_phase1['Primary_Impact_Type'], 
                               df_injury_plays_phase1['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Impact Type-Activity Before the Punt\n(With Penalties)')
df_injury_penalty_phase2 = df_injury_penalty[df_injury_penalty['Phase']==2]
df_injury_penalty_phase2['PenaltyID'].value_counts()
df_cross_injured = pd.crosstab(df_injury_penalty_phase2['Result'], df_injury_penalty_phase2['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity After the Punt\n(With Penalties)')
df_cross_injured = pd.crosstab(df_injury_penalty_phase2['Primary_Impact_Type'], 
                               df_injury_penalty_phase2['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Impact Type-Activity After the Punt\n(With Penalties)')
df_injury_no_penalty = df_injury_plays[df_injury_plays['Penalty']=='No']
len(df_injury_no_penalty)
# Graph
df_injury_no_penalty['Result'].value_counts()
df_cross_injured = pd.crosstab(df_injury_no_penalty['Result'], 
                               df_injury_no_penalty['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity\n(No Penalties)')
df_injury_no_penalty_phase1 = df_injury_no_penalty[df_injury_no_penalty['Phase']==1]
df_cross_injured = pd.crosstab(df_injury_no_penalty_phase1['Result'], 
                               df_injury_no_penalty_phase1['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity Before the Punt\n(No Penalties)')
df_injury_no_penalty_phase2 = df_injury_no_penalty[df_injury_no_penalty['Phase']==2]
df_cross_injured = pd.crosstab(df_injury_no_penalty_phase2['Result'], 
                               df_injury_no_penalty_phase2['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
ax.set_title('Event-Activity After the Punt\n(No Penalties)')
df_cross_injured = pd.crosstab(df_injury_no_penalty['Primary_Impact_Type'], 
                               df_injury_no_penalty['Player_Activity_Derived'])
df_cross_injured = df_cross_injured.fillna(0)

sns.heatmap(df_cross_injured, annot=True, fmt='.1g')
df_injury_np_passive = df_injury_no_penalty[(df_injury_no_penalty['Player_Activity_Derived']=='Blocked') |
                                            (df_injury_no_penalty['Player_Activity_Derived']=='Tackled')]
len(df_injury_np_passive)
# Graph
df_injury_np_passive['Primary_Impact_Type'].value_counts()
def get_side(role):
    left = ['GL', 'GLo', 'GLi', 'PLW', 'PLT', 'PLG',
           'VL', 'VLo', 'VLi',
           'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6',
           'PLL', 'PLL1', 'PLL2', 'PLL3', 'PLLi',
           'PPL', 'PPLo', 'PPLi']
    right = ['GR', 'GRo', 'GRi', 'PRG', 'PRT', 'PRW',
           'VR', 'VRo', 'VRi',
           'PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6',
           'PLR', 'PLR1', 'PLR2', 'PLR3',
            'PPR', 'PPRo', 'PPRi',
           ]
    center = ['PLS', 'PC', 'P', 'PDM',
                'PLM', 'PLM1',
                'PFB', 'PR']
    
    if role in left:
        return 'left'
    if role in right:
        return 'right'
    if role in center:
        return 'center'
    else:
        return ''

df_injury_plays['Player_Side'] = df_injury_plays.apply(lambda row: get_side(row['Role']), axis=1)
df_injury_plays['Partner_Side'] = df_injury_plays.apply(lambda row: get_side(row['Role_Partner']), axis=1)

def get_facing(row):
    if (row['Player_Side'] == 'left' and row['Partner_Side'] == 'right') or \
    (row['Player_Side'] == 'right' and row['Partner_Side'] == 'left') or \
    (row['Player_Side'] == 'center' and row['Partner_Side'] == 'center'):
        return 'Yes' # Facing each other
    elif (row['Player_Side'] == 'center' or row['Partner_Side'] == 'center'):
        return 'Off-center'
    return 'No'
    
df_injury_plays['Facing'] = df_injury_plays.apply(lambda row: get_facing(row), axis=1)
df_cross_injured = pd.crosstab(df_injury_plays['Primary_Impact_Type'], 
                               df_injury_plays['Facing'])
df_cross_injured = df_cross_injured.fillna(0)

sns.heatmap(df_cross_injured, annot=True, fmt='.2g')
df_total = df_play_info['Season_Type'].value_counts().reset_index(name='total')
df_injury_total = df_injury_plays['Season_Type'].value_counts().reset_index(name='injured')

# df_total.head()
df_merged = df_total.merge(df_injury_total, how='right')
df_merged = df_merged.fillna(0)
df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']
df_merged['ratio_safe'] = 100-df_merged['injured']

sns.barplot(x='ratio', y='index', 
#             order=df_merged['ratio'].value_counts().index,
            data=df_merged)
df_penalty = df_play_info[df_play_info['Penalty']=='Yes']

df_total = df_play_info['Season_Type'].value_counts().reset_index(name='total')
df_injury_total = df_penalty['Season_Type'].value_counts().reset_index(name='injured')

# df_total.head()
df_merged = df_total.merge(df_injury_total, how='right')
df_merged = df_merged.fillna(0)
df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']
df_merged['ratio_safe'] = 100-df_merged['injured']

sns.barplot(x='ratio', y='index', 
#             order=df_merged['ratio'].value_counts().index,
            data=df_merged)
df_penalty = df_injury_plays[df_injury_plays['Penalty']=='Yes']

df_total = df_injury_plays['Season_Type'].value_counts().reset_index(name='total')
df_injury_total = df_penalty['Season_Type'].value_counts().reset_index(name='injured')

# df_total.head()
df_merged = df_total.merge(df_injury_total, how='right')
df_merged = df_merged.fillna(0)
df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']
df_merged['ratio_safe'] = 100-df_merged['injured']

sns.barplot(x='ratio', y='index', 
#             order=df_merged['ratio'].value_counts().index,
            data=df_merged)