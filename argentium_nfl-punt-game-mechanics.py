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

df_video_review = pd.read_csv('../input/video_review.csv')

df_punt_role = pd.read_csv('../input/play_player_role_data.csv')
# Graph

# Copied from: https://stackoverflow.com/questions/51417483/mean-median-mode-lines-showing-only-in-last-graph-in-seaborn/51417635

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
df_yardline = df_play_info['YardLine'].str.split(" ", n = 1, expand = True)

df_play_info['yard_team'] = df_yardline[0]

df_play_info['yard_number'] = df_yardline[1].astype(float)



# Process Team Sides

df_home_visit = df_play_info['Home_Team_Visit_Team'].str.split("-", n = 1, expand = True)

df_play_info['home'] = df_home_visit[0]

df_play_info['visit'] = df_home_visit[1]



# Convert to coordinate system, origin at goal line

def convert_yardage(row):

    actual_yards = row['yard_number']

    if row['yard_team'] == row['home']:

        return actual_yards

    else:

        return 100 - actual_yards



# Convert to goal line distance

def convert_goal_distance(row):

    if row.loc[('Poss_Team')] == row.loc[('home')]:

        return row.loc[('Scrimmage_Line')]

    else:

        return 100 - row.loc[('Scrimmage_Line')]



df_play_info['Scrimmage_Line'] = df_play_info.apply(lambda row: convert_yardage(row), axis=1)

df_play_info['Goal_Line'] = df_play_info.apply(lambda row: convert_goal_distance(row), axis=1)



# Display Results

graph_distribution(df_play_info['Goal_Line'])

df_play_info['Goal_Line'].describe()
results = [ 

           'downed',

           'fair catch', 

           'Touchback',

           'out of bounds'

          ]



def get_result(row):

    for result in results:

        if result in row['PlayDescription']:

            return result



    match = re.search('.* for .*', row['PlayDescription'])

    if match:

        return 'runback'



    return 'others'



df_play_info['Result'] = df_play_info.apply(lambda row: get_result(row), axis=1)



# Graph

ax = sns.countplot(y="Result", 

                   order=df_play_info['Result'].value_counts().index,

                   data=df_play_info)



ax.set_title('Punt Play Outcome Frequency')

df_play_info['Result'].value_counts()
ax = sns.boxplot(x="Goal_Line", y="Result", 

            order=df_play_info.groupby(['Result'])['Goal_Line'].median().index,

            data=df_play_info)

ax.set_title('Goal Distance of Outcomes')



median = df_play_info.groupby(['Result'])['Goal_Line'].median()

median.head()
for event in results:

    df_play_info_standard = df_play_info.loc[(df_play_info['Result']!=event) &

                                            (df_play_info['Result']!='others')]

    df_play_info_standard['State'] = 'Standard'



    df_play_info_special = df_play_info[df_play_info['Result']==event]

    df_play_info_special['State'] = 'Event'

    df_concat = pd.concat([df_play_info_standard, df_play_info_special])



    # ANOVA

    mod = ols('Goal_Line ~ State',

                data=df_concat).fit()

    aov_table = sm.stats.anova_lm(mod, typ=2)

    print(str(event) + ":")

    print(aov_table)

    print()
special_circumstances = [ 

               'declined', 

               'No Play',

               'pass incomplete',

               'MUFFS catch',

                'RECOVERED',

                'PENALTY'

          ]



def get_others(row):

    for result in special_circumstances:

        if result in row.loc['PlayDescription']:

            return result

    return None



df_play_info['Special_Circumstance'] = df_play_info.apply(lambda row: 

                                                              get_others(row),

                                                              axis=1)



# Graph

ax = sns.countplot(y="Special_Circumstance", 

                   order=df_play_info['Special_Circumstance'].value_counts().index,

                   data=df_play_info)



ax.set_title('Random Outcomes Frequency')

df_play_info['Special_Circumstance'].value_counts()
def get_range(row):

    for i in range(20):

        number = 5*(i+1)

        if row['Goal_Line'] < number:

            return number

    return None



df_play_info['Range'] = df_play_info.apply(lambda row: 

                                          get_range(row),

                                          axis=1)



df_play_info['count'] = 1



# Remove the random events

df_play_info_norandom = df_play_info[(df_play_info['Result']!='others')]

df_play_info_range = df_play_info_norandom.groupby(['Range', 'Result']).agg({'count': 'sum'})



# Compute Percantage

df_play_info_range_percent = df_play_info_range.groupby(level=0).apply(lambda x:

                                                 100 * x / float(x.sum()))

df_play_info_range_percent = df_play_info_range_percent.rename(columns={'count': 'percent'})

df_play_info_range_percent = df_play_info_range_percent.reset_index()



# Statistics

for result in results:

    grouped_range = df_play_info_range_percent[df_play_info_range_percent['Result']==result]



    print(str(result) + ':')

    output = scipy.stats.pearsonr(grouped_range['Range'], 

                        grouped_range['percent'])

    print(output)

    output = scipy.stats.spearmanr(grouped_range['Range'], 

                        grouped_range['percent'])

    print(output)



# Return (Runbacks)

grouped_range = df_play_info_range_percent[df_play_info_range_percent['Result']=='runback']



print(str('runback') + ':')

output = scipy.stats.pearsonr(grouped_range['Range'], 

                    grouped_range['percent'])

print(output)

output = scipy.stats.spearmanr(grouped_range['Range'], 

                    grouped_range['percent'])

print(output)



# Graph

ax = sns.lineplot(x='Range', y='percent',

                     hue='Result',

                     data=df_play_info_range_percent)

ax.set_title('Outcome Probability vs Goal Distance')
# Computations

def get_pdistance(row):

    str_punt = '.* punts ([0-9]+) yard'

    distance = re.search(str_punt, row['PlayDescription'], re.IGNORECASE)

    if distance:

        return distance.group(1)

    else:

        return 0



df_play_info['Punts'] = df_play_info.apply(lambda row: get_pdistance(row), axis=1)

df_play_info['Punts'] = df_play_info['Punts'].astype(int)



# Remove null rows

df_play_info_punts = df_play_info.dropna(subset=['Punts'])



# Display Results

graph_distribution(df_play_info_punts['Punts'])

df_play_info_punts['Punts'].describe()
df_play_info_punts = df_play_info.dropna(subset=['Punts'])

df_play_info_punts_20 = df_play_info_punts[df_play_info_punts['Punts']<20]

print(1-len(df_play_info_punts_20)/len(df_play_info_punts))
df_play_info['LandingZone'] = df_play_info['Goal_Line'] + df_play_info['Punts']



# Display Results

graph_distribution(df_play_info['LandingZone'])

df_play_info['LandingZone'].describe()
def get_rdistance(row):

    if 'no gain' in row['PlayDescription']:

        return 0



    str_return = '.* for (-*[0-9]+)'

    distance = re.search(str_return, row['PlayDescription'], re.IGNORECASE)

    if distance:

        return int(distance.group(1))

    else:

        return 0



df_play_info['Returns'] = df_play_info.apply(lambda row: get_rdistance(row), axis=1)



# Select cases when there are returns

df_play_info_returns = df_play_info[df_play_info['Result']=='runback']



# Display Results

graph_distribution(df_play_info_returns['Returns'])

df_play_info_returns['Returns'].describe()
# Computations

df_play_info['PuntGain'] = df_play_info['Punts'] - df_play_info['Returns']



# Graphs

graph_distribution(df_play_info['PuntGain'])

df_play_info['PuntGain'].describe()
df_play_info['Next_Goal'] = df_play_info['Goal_Line'] + df_play_info['PuntGain']



# Display Results

graph_distribution(df_play_info['Next_Goal'])

df_play_info['Next_Goal'].describe()
df_injury = df_video_review.merge(df_play_info, 

                                  left_on=['GameKey', 'PlayID'],

                                 right_on=['GameKey', 'PlayID'],

                                 how='left')
print(len(df_injury))
ax = sns.countplot(y="Result", 

                   order=df_injury['Result'].value_counts().index,

                   data=df_injury)

ax.set_title('Frequency Count')

df_play_info['Result'].value_counts()
df_total = df_play_info['Result'].value_counts().reset_index(name='total')

df_injury_total = df_injury['Result'].value_counts().reset_index(name='injured')



df_merged = df_total.merge(df_injury_total, how='right')

df_merged = df_merged.fillna(0)

df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

df_merged = df_merged.sort_values(by=['ratio'])



ax = sns.barplot(x='ratio', y='index', 

            data=df_merged)

ax.set_title('Injury Risk per Outcome')
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



def set_role_category(role):

    for category in role_categories.keys():

        if str(role) in role_categories[category]:

            return str(category)

    return None



df_punt_role['Role_Category'] = df_punt_role.apply(lambda row: set_role_category(row['Role']), 

                                                axis=1)
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



df_punt_role['Team'] = df_punt_role.apply(lambda row: set_team(row['Role']), 

                                                axis=1)
# Punt Roles

# Convert to int data type

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

df_injury = df_injury.drop(['GSISID_Partner', 'Season_Year_Partner'], axis=1)
df_injury.isnull().sum()
df_injury_null = df_injury[pd.isnull(df_injury['Role_Partner'])]

df_injury_null.head()
df_injury = df_injury.fillna({'Friendly_Fire': 'Ground',

                             'Role_Partner': 'Unknown'})
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



# Merge with df_punt_role

df_injury['Phase'] = df_injury.apply(lambda row: 

                                                set_phase(row), 

                                                axis=1)
ax = sns.countplot(x="Phase", data=df_injury)

ax.set_title('Phase Injury Frequency')
df_phase1 = df_injury[df_injury['Phase']==1]

print(len(df_phase1))
# df_phase1['Result'].value_counts()

ax = sns.countplot(x="Result", data=df_phase1)

ax.set_title('Before the Kick: Outcome Frequency')
df_cross_injured = pd.crosstab(df_phase1['Player_Activity_Derived'], df_phase1['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.1g')

ax.set_title('Before the Kick: Activity Pairs')
df_phase1_tackled = df_phase1[df_phase1['Player_Activity_Derived'] == 'Tackled']

df_phase1_tackled.head()
df_phase1_blocking = df_phase1[df_phase1['Player_Activity_Derived'] == 'Blocking']



# Graph

df_cross_injured = pd.crosstab(df_phase1_blocking['Role'], 

                               df_phase1_blocking['Role_Partner'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('Before the Kick: Blocked Role Pairs')
ax = sns.countplot(y="Primary_Impact_Type", 

                   order = df_phase1_blocking['Primary_Impact_Type'].value_counts().index,

                   data=df_phase1_blocking)

ax.set_title('Before the Kick: Blocking Impact Type Frequency')
df_phase2 = df_injury[df_injury['Phase']==2]

print(len(df_phase2))
ax = sns.countplot(y="Role", 

                   order = df_phase2['Role'].value_counts().index,

                   data=df_phase2)

ax.set_title('After the Kick: Injury Frequency')
df_total = df_punt_role['Role'].value_counts().reset_index(name='total')

df_injury_total = df_phase2['Role'].value_counts().reset_index(name='injured')



# df_total.head()

df_merged = df_total.merge(df_injury_total, how='right')

df_merged = df_merged.fillna(0)

df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

df_merged = df_merged.sort_values(by=['ratio'], ascending=False)



ax = sns.barplot(x='ratio', y='index', 

            data=df_merged)

ax.set_title('After the Kick: Role Injury Ratio')

df_merged.head()
ax = sns.countplot(y="Role_Partner", 

                   order = df_phase2['Role_Partner'].value_counts().index,

                   data=df_phase2)

ax.set_title('After the Kick: Partner Frequency')
df_total = df_punt_role['Role'].value_counts().reset_index(name='total')

df_injury_total = df_phase2['Role_Partner'].value_counts().reset_index(name='injured')



# df_total.head()

df_merged = df_total.merge(df_injury_total, how='right')

df_merged = df_merged.fillna(0)

df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

df_merged = df_merged.sort_values(by=['ratio'], ascending=False)



ax = sns.barplot(x='ratio', y='index', 

            data=df_merged)

ax.set_title('After the Kick: Role Injury Ratio')

df_merged.head()
df_phase2_pr = df_phase2[(df_phase2['Role']=='PR') | (df_phase2['Role_Partner']=='PR')]



# Graph

df_cross_injured = pd.crosstab(df_phase2_pr['Player_Activity_Derived'], 

                               df_phase2_pr['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.1g')

ax.set_title('After the Kick: PR-related Activity Pairs')
df_phase2_pr = df_phase2[(df_phase2['Role']=='PR') | (df_phase2['Role_Partner']=='PR')]



# Graph

df_cross_injured = pd.crosstab(df_phase2_pr['Role'], df_phase2_pr['Role_Partner'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.1g')

ax.set_title('After the Kick: PR-related Partner Roles')
df_cross_injured = pd.crosstab(df_phase2_pr['Player_Activity_Derived'], df_phase2_pr['Primary_Impact_Type'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Activity Impact Types\n(PR-related Injuries)')
ax = sns.countplot(y='Role_Category', 

              order = df_phase2['Role_Category'].value_counts().index,

            data=df_injury)

ax.set_title('After the Kick: Role Category Frequency')
df_total = df_punt_role['Role_Category'].value_counts().reset_index(name='total')

df_injury_total = df_phase2['Role_Category'].value_counts().reset_index(name='injured')



# df_total.head()

df_merged = df_total.merge(df_injury_total, how='right')

df_merged = df_merged.fillna(0)

df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

df_merged = df_merged.sort_values(by=['ratio'], ascending=False)



ax = sns.barplot(x='ratio', y='index', 

            data=df_merged)

ax.set_title('After the Kick: Role Injury Ratio')

df_merged.head()
ax = sns.countplot(y='Role_Category_Partner', 

              order = df_injury['Role_Category_Partner'].value_counts().index,

            data=df_injury)

ax.set_title('After the Kick: Role Partner Frequency')
df_total = df_punt_role['Role_Category'].value_counts().reset_index(name='total')

df_injury_total = df_phase2['Role_Category_Partner'].value_counts().reset_index(name='injured')



# df_total.head()

df_merged = df_total.merge(df_injury_total, how='right')

df_merged = df_merged.fillna(0)

df_merged['ratio'] = 100*df_merged['injured'] / df_merged['total']

df_merged = df_merged.sort_values(by=['ratio'], ascending=False)



ax = sns.barplot(x='ratio', y='index', 

            data=df_merged)

ax.set_title('After the Kick: Role Partner Ratio')



df_merged.head()
# Graph

df_cross_injured = pd.crosstab(df_phase2['Role_Category'], 

                               df_phase2['Primary_Impact_Type'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Role Category and Impact Type')
df_phase2_gunner_player = df_phase2[(df_phase2['Role_Category']=='G')]



# Graph

df_cross_injured = pd.crosstab(df_phase2_gunner_player['Player_Activity_Derived'], 

                               df_phase2_gunner_player['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Activity Pairs\n(Gunner is Injured)')
df_phase2_gunner = df_phase2[(df_phase2['Role_Category']=='G') |

                            (df_phase2['Role_Category_Partner']=='G')]



df_phase2_gunner_tackling = df_phase2_gunner[(df_phase2_gunner['Player_Activity_Derived']=='Tackling') &

                                            (df_phase2_gunner['Primary_Partner_Activity_Derived']=='Tackling')]



df_phase2_gunner_tackling.head()
df_phase2_gunner = df_phase2[(df_phase2['Role_Category']=='G') |

                            (df_phase2['Role_Category_Partner']=='G')]



# Graph

df_cross_injured = pd.crosstab(df_phase2_gunner['Role_Category'], 

                               df_phase2_gunner['Role_Category_Partner'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Gunner Pairs')
df_phase2_gunner_partner = df_phase2[(df_phase2['Role_Category_Partner']=='G')]



# Graph

df_cross_injured = pd.crosstab(df_phase2_gunner_partner['Player_Activity_Derived'], 

                               df_phase2_gunner_partner['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Activity Pairs\n(Gunner is Partner Role)')
ax = sns.countplot(y='Primary_Impact_Type', 

              order = df_phase2_gunner_partner['Primary_Impact_Type'].value_counts().index,

            data=df_phase2_gunner_partner)

ax.set_title('After the Kick: Impact Types\n(Gunner is Partner Role)')



df_phase2_gunner_partner['Primary_Impact_Type'].value_counts()
ax = sns.countplot(y='Friendly_Fire', 

              order = df_phase2_gunner['Friendly_Fire'].value_counts().index,

            data=df_phase2_gunner)

ax.set_title('After the Kick: Friendly-Fires\n(Gunner is Involved)')



df_phase2_gunner['Friendly_Fire'].value_counts()
df_phase2_nopr = df_phase2[(df_phase2['Role']!='PR') &

                          (df_phase2['Role_Partner']!='PR')]



ax = sns.countplot(y='Friendly_Fire', 

              order = df_phase2_nopr['Friendly_Fire'].value_counts().index,

            data=df_phase2_nopr)

ax.set_title('After the Kick: Opponent Injuries')



df_phase2_nopr['Friendly_Fire'].value_counts()
# Graph

df_cross_injured = pd.crosstab(df_phase2_nopr['Team'], 

                               df_phase2_nopr['Team_Partner'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Team vs Team Injuries \n(Without PR Injuries)')
df_phase2_nopr_opponent = df_phase2_nopr[df_phase2_nopr['Friendly_Fire']=='No']



# Graph

df_cross_injured = pd.crosstab(df_phase2_nopr_opponent['Player_Activity_Derived'], 

                               df_phase2_nopr_opponent['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Opponent Activity Pairs\n(Without PR-related injuries)')
# Graph

df_cross_injured = pd.crosstab(df_phase2_nopr_opponent['Team'], 

                               df_phase2_nopr_opponent['Primary_Impact_Type'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Team vs Team Injuries \n(Without PR Injuries)')
df_phase2_nopr_opponent_nogunner = df_phase2_nopr_opponent[df_phase2_nopr_opponent['Role_Category']!='G']



# Graph

df_cross_injured = pd.crosstab(df_phase2_nopr_opponent_nogunner['Team'], 

                               df_phase2_nopr_opponent_nogunner['Primary_Impact_Type'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Team vs Team Injuries \n(Without PR or Gunner Injuries)')
# Graph

df_cross_injured = pd.crosstab(df_phase2['Team'], df_phase2['Friendly_Fire'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Team Opponent Injuries\n(Overall)')
df_friendly_fire = df_phase2[(df_phase2['Friendly_Fire']=='Yes')]



# Graph

df_cross_injured = pd.crosstab(df_friendly_fire['Player_Activity_Derived'], df_friendly_fire['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Friendly-fire Activity Pairs')
df_friendly_fire = df_phase2[(df_phase2['Friendly_Fire']=='Yes')]

df_friendly_fire_nogunner = df_friendly_fire[(df_friendly_fire['Role_Category']!='G') &

                                            (df_friendly_fire['Role_Category_Partner']!='G')]



# Graph

df_cross_injured = pd.crosstab(df_friendly_fire_nogunner['Player_Activity_Derived'], 

                               df_friendly_fire_nogunner['Primary_Partner_Activity_Derived'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Friendly-fire Activity Pairs\n(No Gunner Collisions)')
# Graph

df_cross_injured = pd.crosstab(df_friendly_fire_nogunner['Role_Category'], 

                               df_friendly_fire_nogunner['Role_Category_Partner'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Friendly-fire Role Pairs\n(No Gunner Collisions)')
# Graph

ax = sns.countplot(y="Primary_Impact_Type", 

                   order = df_friendly_fire_nogunner['Primary_Impact_Type'].value_counts().index,

                   data=df_friendly_fire_nogunner)

ax.set_title('After the Kick: Friendly-fire Impact Types\n(No Gunner)')

df_friendly_fire_nogunner['Primary_Impact_Type'].describe()
df_phase2_ground = df_phase2[df_phase2['Friendly_Fire']=='Ground']



# Graph

df_cross_injured = pd.crosstab(df_phase2_ground['Player_Activity_Derived'],

                               df_phase2_ground['Role'])

df_cross_injured = df_cross_injured.fillna(0)



ax = sns.heatmap(df_cross_injured, annot=True, fmt='.2g')

ax.set_title('After the Kick: Activity and Role Pairs\n(Ground Injuries)')
ax = sns.countplot(y="Player_Activity_Derived", 

                   order = df_phase2['Player_Activity_Derived'].value_counts().index,

                   data=df_phase2)

ax.set_title('After the Kick: Activity Injury Frequency')

df_phase2['Player_Activity_Derived'].describe()