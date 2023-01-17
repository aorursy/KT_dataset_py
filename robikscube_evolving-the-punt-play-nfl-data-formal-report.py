import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
import seaborn as sns
import math

from IPython.core.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from string import ascii_letters

plt.style.use('ggplot')

color1 = '#E24A33'
color2 = '#348ABD'
color3 = '#988ED5'
color4 = '#777777'

%matplotlib inline
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')

    return fig, ax
# Read in non-NGS data sources
ppd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
gd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
pprd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
vr = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
vfi = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
pi = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
vfi = vfi.rename(columns={'season' : 'season_year'})
all_dfs = [ppd, gd, pprd, vr, vfi, pi]
# Change column names so they are all lowercase 
# never have to guess about which letters are uppercase
for mydf in all_dfs:
    mydf.columns = [col.lower() for col in mydf.columns]
"""
Read in NGS Data (Only done once)
"""
read_ngs_data = False # Set to true to read NGS data (takes time and uses a lot of memory)
# Read in NGS data and combine.
# First try to read parquet file if it exists
def load_NGS_df(path = "../input/"):
    
    # gets all csv with NGS in their filename
    NGS_csvs = [path+file for file in os.listdir(path) if 'NGS' in file]
    
    df = pd.DataFrame() #initialize an empty dataframe
    
    # loop to csv then appends it to df
    for path_csv in NGS_csvs:
        _df = pd.read_csv(path_csv,low_memory=False)
        df = df.append(_df,ignore_index=True)
        del _df # deletes the _df to free up memory
        
    return df

if read_ngs_data:
    try:
        ngs = pd.read_parquet('../input/ngs_combined_with_role_lssnap.parquet')
    except:
        print('No parquet file- reading from CSVs')
        ngs = load_NGS_df()
        ngs = pd.merge(ngs, pprd) # Merge player position data
        ngs.columns = [col.lower() for col in ngs.columns]
        # Add rows for long snapper position at snap (for reference x,y)
        ngs_ls_at_snap = ngs.loc[(ngs['event'] == 'ball_snap') &
                             (ngs['role'] == 'PLS')]

        ngs = pd.merge(ngs,
                   ngs_ls_at_snap[['season_year','gamekey','playid','time','x','y']],
                   on=['season_year','gamekey','playid'],
                   how='left',
                   suffixes=('','_ls_at_snap'))
        ngs.to_parquet('../input/ngs_combined_with_role_lssnap.parquet')
"""
Create Dataframe with Generalized Punting Roles
include which team they are on (punting/returning)
"""
role_info_dict = {'GL': ['Gunner', 'Punting_Team'],
                  'GLi': ['Gunner', 'Punting_Team'],
                  'GLo': ['Gunner', 'Punting_Team'],
                  'GR': ['Gunner', 'Punting_Team'],
                  'GRi': ['Gunner', 'Punting_Team'],
                  'GRo': ['Gunner', 'Punting_Team'],
                  'P': ['Punter', 'Punting_Team'],
                  'PC': ['Punter_Protector', 'Punting_Team'],
                  'PPR': ['Punter_Protector', 'Punting_Team'],
                  'PPRi': ['Punter_Protector', 'Punting_Team'],
                  'PPRo': ['Punter_Protector', 'Punting_Team'],
                  'PDL1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL6': ['Defensive_Lineman', 'Returning_Team'],
                  'PFB': ['PuntFullBack', 'Returning_Team'],
                  'PLG': ['Punting_Lineman', 'Punting_Team'],
                  'PLL': ['Defensive_Backer', 'Returning_Team'],
                  'PLL1': ['Defensive_Backer', 'Returning_Team'],
                  'PLL3': ['Defensive_Backer', 'Returning_Team'],
                  'PLS': ['Punting_Longsnapper', 'Punting_Team'],
                  'PLT': ['Punting_Lineman', 'Punting_Team'],
                  'PLW': ['Punting_Wing', 'Punting_Team'],
                  'PRW': ['Punting_Wing', 'Punting_Team'],
                  'PR': ['Punt_Returner', 'Returning_Team'],
                  'PRG': ['Punting_Lineman', 'Punting_Team'],
                  'PRT': ['Punting_Lineman', 'Punting_Team'],
                  'VLo': ['Jammer', 'Returning_Team'],
                  'VR': ['Jammer', 'Returning_Team'],
                  'VL': ['Jammer', 'Returning_Team'],
                  'VRo': ['Jammer', 'Returning_Team'],
                  'VRi': ['Jammer', 'Returning_Team'],
                  'VLi': ['Jammer', 'Returning_Team'],
                  'PPL': ['Punter_Protector', 'Punting_Team'],
                  'PPLo': ['Punter_Protector', 'Punting_Team'],
                  'PPLi': ['Punter_Protector', 'Punting_Team'],
                  'PLR': ['Defensive_Backer', 'Returning_Team'],
                  'PRRo': ['Defensive_Backer', 'Returning_Team'],
                  'PDL4': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR4': ['Defensive_Lineman', 'Returning_Team'],
                  'PLM': ['Defensive_Backer', 'Returning_Team'],
                  'PLM1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR2': ['Defensive_Backer', 'Returning_Team'],
                  'PLR3': ['Defensive_Backer', 'Returning_Team'],
                  'PLL2': ['Defensive_Backer', 'Returning_Team'],
                  'PDM': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR6': ['Defensive_Lineman', 'Returning_Team'],
                  }
role_info = pd.DataFrame.from_dict(role_info_dict, orient='index',
                                   columns=['generalized_role', 'punting_returning_team']) \
    .reset_index() \
    .rename(columns={'index': 'role'})
# Play information preprocessing

penalties_list = ['Offensive Holding', 'Defensive 12 On-field', 'Illegal Block Above the Waist', 'Fair Catch Interference',
                  'Running Into the Kicker', 'Unnecessary Roughness', 'Illegal Touch Kick',
                  'Illegal Use of Hands', 'False Start', 'Out of Bounds on Punt', 'Horse Collar Tackle',
                  'Face Mask', 'Ineligible Downfield Kick', 'Illegal Substitution', 'Illegal Formation',
                  'Delay of Game', 'Illegal Blindside Block', 'Neutral Zone Infraction', 'Tripping',
                  'Defensive Holding', 'Roughing the Kicker', 'Unsportsmanlike Conduct', 'Defensive Offside',
                  'Interference with Opportunity to Catch', 'Illegal Motion', 'Chop Block', 'Clipping',
                  'Invalid Fair Catch Signal', 'Illegal Shift', 'Offensive 12 On-field', 'Taunting',
                  'Offensive Pass Interference', 'Disqualification', 'Defensive Pass Interference']

pi['PENALTY'] = (pi['playdescription'].str.contains('PENALTY')
                 | pi['playdescription'].str.contains('Penalty'))
pi['declined'] = pi['playdescription'].str.contains('declined')


pi['Offensive Holding'] = pi['playdescription'].str.contains(
    'Offensive Holding')
pi['Defensive 12 On-field'] = pi['playdescription'].str.contains(
    'Defensive 12 On-field')
pi['Illegal Block Above the Waist'] = pi['playdescription'].str.contains(
    'Illegal Block Above the Waist')
pi['Fair Catch Interference'] = pi['playdescription'].str.contains(
    'Fair Catch Interference')
pi['Running Into the Kicker'] = pi['playdescription'].str.contains(
    'Running Into the Kicker')
pi['Unnecessary Roughness'] = pi['playdescription'].str.contains(
    'Unnecessary Roughness')
pi['Illegal Touch Kick'] = pi['playdescription'].str.contains(
    'Illegal Touch Kick')
pi['Illegal Use of Hands'] = pi['playdescription'].str.contains(
    'Illegal Use of Hands')
pi['False Start'] = pi['playdescription'].str.contains('False Start')
pi['Out of Bounds on Punt'] = pi['playdescription'].str.contains(
    'Out of Bounds on Punt')
pi['Horse Collar Tackle'] = pi['playdescription'].str.contains(
    'Horse Collar Tackle')
pi['Face Mask'] = pi['playdescription'].str.contains('Face Mask')
pi['Ineligible Downfield Kick'] = pi['playdescription'].str.contains(
    'Ineligible Downfield Kick')
pi['Illegal Substitution'] = pi['playdescription'].str.contains(
    'Illegal Substitution')
pi['Illegal Formation'] = pi['playdescription'].str.contains(
    'Illegal Formation')
pi['Delay of Game'] = pi['playdescription'].str.contains('Delay of Game')
pi['Illegal Blindside Block'] = pi['playdescription'].str.contains(
    'Illegal Blindside Block')
pi['Neutral Zone Infraction'] = pi['playdescription'].str.contains(
    'Neutral Zone Infraction')
pi['Tripping'] = pi['playdescription'].str.contains('Tripping')
pi['Defensive Holding'] = pi['playdescription'].str.contains(
    'Defensive Holding')
pi['Roughing the Kicker'] = pi['playdescription'].str.contains(
    'Roughing the Kicker')
pi['Unsportsmanlike Conduct'] = pi['playdescription'].str.contains(
    'Unsportsmanlike Conduct')
pi['Defensive Offside'] = pi['playdescription'].str.contains(
    'Defensive Offside')
pi['Interference with Opportunity to Catch'] = pi['playdescription'].str.contains(
    'Interference with Opportunity to Catch')
pi['Illegal Motion'] = pi['playdescription'].str.contains('Illegal Motion')
pi['Chop Block'] = pi['playdescription'].str.contains('Chop Block')
pi['Clipping'] = pi['playdescription'].str.contains('Clipping')
pi['Invalid Fair Catch Signal'] = pi['playdescription'].str.contains(
    'Invalid Fair Catch Signal')
pi['Illegal Shift'] = pi['playdescription'].str.contains('Illegal Shift')
pi['Offensive 12 On-field'] = pi['playdescription'].str.contains(
    'Offensive 12 On-field')
pi['Taunting'] = pi['playdescription'].str.contains('Taunting')
pi['Offensive Pass Interference'] = pi['playdescription'].str.contains(
    'Offensive Pass Interference')
pi['Disqualification'] = pi['playdescription'].str.contains('Disqualification')
pi['Defensive Pass Interference'] = pi['playdescription'].str.contains(
    'Defensive Pass Interference')

# No Play
pi['No Play'] = pi['playdescription'].str.contains('No Play')

pi['count'] = 1

# Play challended / reviewed
pi['Challenged'] = pi['playdescription'].str.contains('challenged')
pi['Review'] = pi['playdescription'].str.contains('review')
pi['Upheld'] = pi['playdescription'].str.contains('Upheld')
pi['Reversed'] = pi['playdescription'].str.contains('REVERSED')

pi['Reversed'] = pi['playdescription'].str.contains('REVERSED')
pi['Out of Bounds'] = pi['playdescription'].str.contains('out of bounds')
pi['Touchback'] = pi['playdescription'].str.contains('Touchback')
pi['Fair Catch'] = pi['playdescription'].str.contains('fair catch')
pi['Muffed'] = pi['playdescription'].str.contains('MUFF')
pi['Downed'] = pi['playdescription'].str.contains('downed')
pi['Punt Blocked'] = pi['playdescription'].str.contains('BLOCKED')
pi['Touchdown'] = pi['playdescription'].str.contains('TOUCHDOWN')
pi['Returned for No Gain'] = pi['playdescription'].str.contains('no gain')
pi['Fumbled'] = pi['playdescription'].str.contains('FUMBLES')
pi['Pass Incomplete'] = pi['playdescription'].str.contains('pass incomplete')

pi['Returned'] = (~pi['No Play'] &
                  ~pi['Out of Bounds'] &
                  ~pi['Touchback'] &
                  ~pi['Fair Catch'] &
                  ~pi['Downed'] &
                  ~pi['Muffed'] &
                  ~pi['Punt Blocked'] &
                  #  ~pi['PENALTY'] &
                  ~pi['Returned for No Gain'] &
                  ~pi['Fumbled'] &
                  ~pi['Pass Incomplete'])

# Extract the number of yards the return was for from the play description
pi['Returned For'] = pi[~pi['No Play'] &
                        ~pi['Out of Bounds'] &
                        ~pi['Touchback'] &
                        ~pi['Fair Catch'] &
                        ~pi['Downed'] &
                        ~pi['Muffed'] &
                        ~pi['Punt Blocked'] &
                        #    ~pi['PENALTY'] &
                        ~pi['Returned for No Gain'] &
                        ~pi['Fumbled'] &
                        ~pi['Pass Incomplete']]['playdescription'] \
    .str.extract('(for .* yard)', expand=True).fillna(False)

# Cleanup ugly retrun yards and get int
pi['return_yards'] = pi['Returned For'].replace('for -2 yards. Lateral to C.Patterson to MIN 31 for 9 yards (W.Woodyard', 'for 9 yards') \
    .replace('for -4 yards. Lateral to R.Mostert to SEA 35 for 33 yard', 'for 33 yard') \
    .replace('for 10 yards (K.Byard', 'for 11 yard') \
    .replace('for 12 yards (A.Blake; W.Woodyard', 'for 12 yard') \
    .replace('for 14 yards (N.Palmer; K.Byard', 'for 14 yard') \
    .replace('for 44 yards (R.Blanton; C.Schmidt). Buffalo challenged the runner was in bounds ruling, and the play was REVERSED. C.Schmidt punts 35 yards to SEA 38, Center-G.Sanborn. T.Lockett ran ob at BUF 40 for 22 yard', 'for 22 yard') \
    .replace('for 2 yards (W.Woodyard', 'for 2 yard') \
    .replace('for -2 yards. Lateral to C.Patterson to MIN 31 for 9 yard', 'for 9 yard') \
    .replace('for 34 yards (C.Goodwin). Atlanta challenged the runner was in bounds ruling, and the play was REVERSED. M.Bosher punts 56 yards to NE 21, Center-J.Harris. J.Edelman ran ob at NE 47 for 26 yard', 'for 26 yard') \
    .dropna() \
    .str.replace('for ', '').str.replace('yard', '')
# Zero return yards for 'no gain'
pi.loc[pi['Returned for No Gain'], 'return_yards'] = 0
# pull just yards from string

# Clean up `s` from yards field
pi['return_yards'] = pi['return_yards'].str[:4].replace(
    's', ' ').replace('s', '')
pi.loc[~pi['return_yards'].isna() & pi['return_yards'].str.contains('s'), 'return_yards'] = \
    pi.loc[~pi['return_yards'].isna() & pi['return_yards'].str.contains('s')
           ]['return_yards'].str[:2]
pi['return_yards'] = pi['return_yards'].str.replace('(', '')
pi['return_yards'] = pd.to_numeric(pi['return_yards'])
pi.loc[pi['Returned for No Gain'], 'return_yards'] = 0
pi['Returned for Negative Gain'] = (pi['Returned']) & (pi['return_yards'] < 0)
pi['Returned for Positive Gain'] = (pi['Returned']) & (pi['return_yards'] > 0)

# Punting Distance
new = pi['playdescription'].str.split(pat='punts', n=1, expand=True)
pi['split1'] = new[0]
pi['split2'] = new[1]
pi['punt_distance'] = pi['split2'].str.extract('(\d+)')
pi.drop(columns=["split1", "split2"], inplace=True)
# pi['punt_distance'] = pi['punt_distance'].fillna(0)
pi['punt_distance'] = pd.to_numeric(pi['punt_distance'])

# Return yards calculation version 2 inspired by from https://www.kaggle.com/travisbuhrow/the-fair-catch-advancement-rule
ReturnYards = pi['playdescription'].str.split("for ", n=1, expand=True)
ReturnYards = ReturnYards.astype('str')
ReturnYards = ReturnYards[1].str.split(" yard", n=1, expand=True)
pi['Return_Yards'] = ReturnYards[0]
# pi.loc[(play_info.Return_Indicator == False),'Return_Yards'] = 0
# pi.loc[(play_info.Return_Yards.str.len() > 2),'Return_Yards'] = 0
pi['Return_Yards'] = pi['Return_Yards'].replace('None', np.nan)
pi.loc[pi['Return_Yards'].str.contains('no gain') == True, 'Return_Yards'] = 0
pi.loc[(pi.Return_Yards.str.len() > 3), 'Return_Yards'] = np.nan
pi['Return_Yards'] = pd.to_numeric(pi['Return_Yards'])

# Plot Penalties

ax = pi[penalties_list].sum() \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 10), color='grey', title='Punting Play Penalty Types')

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

plt.show()
injury_plays = pd.merge(pi, vr)

ax = injury_plays[penalties_list].sum().loc[(injury_plays[penalties_list].sum() != 0)] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 4), color='grey', title='Penalties on Punts Involving Injuries')

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

plt.show()
ax = pi[pi['No Play']][penalties_list].sum() \
    .sort_values() \
    .plot(kind='barh',
          figsize=(15, 8),
          color='grey',
          title='Penalties Causing Negation of Play')
rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

plt.show()
pi[pi['Challenged'] | pi['Review']] \
    .groupby(['Challenged'])[['Upheld','Reversed']] \
    .sum() \
    .sort_values('Challenged', ascending=False)
ax = (pi[['Returned for Negative Gain',
          'Returned for Positive Gain',
          'Reversed',
          'Out of Bounds',
          'Touchback',
          'Fair Catch',
          'Muffed',
          'Downed',
          'Punt Blocked',
          'Touchdown',
          'Returned for No Gain',
          'Fumbled',
          'Pass Incomplete']]
      .sum() / pi[['Returned for Negative Gain',
                   'Returned for Positive Gain',
                   'Reversed',
                   'Out of Bounds',
                   'Touchback',
                   'Fair Catch',
                   'Muffed',
                   'Downed',
                   'Punt Blocked',
                   'Touchdown',
                   'Returned for No Gain',
                   'Fumbled',
                   'Pass Incomplete']].count() * 100) \
    .sort_values(ascending=True) \
    .plot(kind='barh',
          color='grey',
          figsize=(15, 5),
          title='Result of Punt Plays (Multiple Results Possible per play)')

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.2f}%".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.

plt.show()
ax = pd.merge(pi, vr, on=['season_year',
                          'playid',
                          'gamekey'])[['Returned for Negative Gain',
                                       'Returned for Positive Gain',
                                       'Reversed',
                                       'Out of Bounds',
                                       'Touchback',
                                       'Fair Catch',
                                       'Muffed',
                                       'Downed',
                                       'Punt Blocked',
                                       'Touchdown',
                                       'Returned for No Gain',
                                       'Fumbled',
                                       'Pass Incomplete']] \
    .sum() \
    .sort_values(ascending=True) \
    .plot(kind='barh',
          color='grey',
          figsize=(15, 5),
          title='Result of Punts involving Concussions',
          grid=True)

rects = ax.patches
ax.grid(False)
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.

major_ticks = np.arange(0, 28, 2)
ax.set_xticks(major_ticks)
ax.grid(True)
plt.show()
sns.set(style="white")
plt.style.use('ggplot')
pi.loc[(~pi['Reversed']) & (~pi['PENALTY'])]['Return_Yards'] \
    .plot(kind='hist', figsize=(15, 5),
          bins=55, color='grey',
          title='Distribution of Return Yards on Punt Plays (Excluding Penalty and Reversed Plays)')
plt.show()
print('Median return distance is for : {:.2f} yards'.format(pi['Return_Yards'].median()))
print('Mean return distance is for   : {:.2f} yards'.format(pi['Return_Yards'].mean()))
# Return yards preprocessing
return_yards_dropna = pi.loc[(~pi['Reversed']) & (~pi['PENALTY'])][['Return_Yards']].dropna()
try:
    return_yards_dropna = return_yards_dropna.drop('Return_Summary', axis=1)
except:
    pass
return_yards_dropna.loc[return_yards_dropna['Return_Yards'] < 0, 'Return_Summary'] = 'Negative Return'
return_yards_dropna.loc[return_yards_dropna['Return_Yards'] == 0, 'Return_Summary'] = 'No Gain'
return_yards_dropna.loc[(return_yards_dropna['Return_Yards'] > 0) &
                        (return_yards_dropna['Return_Yards'] <= 5), 'Return_Summary'] = 'Return +0 to 5 Yards'
return_yards_dropna.loc[(return_yards_dropna['Return_Yards'] > 5) &
                        (return_yards_dropna['Return_Yards'] <= 10), 'Return_Summary']= 'Return +5-10 Yards'
return_yards_dropna.loc[(return_yards_dropna['Return_Yards'] > 10) &
                        (return_yards_dropna['Return_Yards'] <= 15), 'Return_Summary'] = 'Return +10-15 Yards'
return_yards_dropna.loc[(return_yards_dropna['Return_Yards'] > 15) &
                        (return_yards_dropna['Return_Yards'] <= 20), 'Return_Summary'] = 'Return +15-20 Yards'
return_yards_dropna.loc[(return_yards_dropna['Return_Yards'] > 20), 'Return_Summary'] = 'Return Over +20 Yards'

return_yards_dropna.groupby('Return_Summary').count()['Return_Yards']
return_yards_dropna['Return_Summary'] = pd.Categorical(return_yards_dropna['Return_Summary'],
                                                       ["Negative Return",
                                                        "No Gain",
                                                        "Return +0 to 5 Yards",
                                                        "Return +5-10 Yards",
                                                        "Return +10-15 Yards",
                                                        "Return +15-20 Yards",
                                                        "Return Over +20 Yards"])
ax = (return_yards_dropna.groupby('Return_Summary') \
    .count() / len(return_yards_dropna) * 100) \
    .sort_index(ascending=False) \
    .plot(kind='barh', figsize=(15, 5),
        color='grey',
         title='Percentage of Returns by Return Yards',
         legend=False)

rects = ax.patches
ax.grid(False)
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.2f}%".format(x_value)

    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.
ax.set_ylabel('')
ax.grid()
plt.show()
pi['punt_distance'].plot(kind='hist',
                         figsize=(15, 5),
                         bins=79,
                         title='Distribution of Punt Distances for all plays',
                        color='grey')
plt.show()
print('The mean distance of a punt is {:.2f} yards'.format(pi['punt_distance'].mean()))
print('The median distance of a punt is {:.2f} yards'.format(pi['punt_distance'].median()))
injury_plays_df = pd.merge(vr, pi)

injury_plays_df['punt_distance'].plot(kind='hist',
                         figsize=(15, 5),
                         bins=10,
                         title='Distribution of Punt Distances',
                        color='grey')
plt.show()
print('The mean distance of punts involving a concussion is {:.2f} yards'.format(injury_plays_df['punt_distance'].mean()))
print('The median distance of punts involving a concussion punt is {:.2f} yards'.format(injury_plays_df['punt_distance'].median()))
# Data Prep
injury_play_ngs = pd.read_parquet(
    '../input/nfl-punt-data-preprocessing-ngs-injury-plays/NGS-injury-plays.parquet')
gsisid_numbers = ppd.groupby('gsisid')['number'].apply(
    lambda x: "%s" % ', '.join(x))
gsisid_numbers = pd.DataFrame(gsisid_numbers).reset_index()
# Add Player Number and Direction
vr_with_number = pd.merge(
    vr, gsisid_numbers, how='left', suffixes=('', '_injured'))
vr_with_number['primary_partner_gsisid'] = vr_with_number['primary_partner_gsisid'].replace(
    'Unclear', np.nan).fillna(0).astype('int')
vr_with_number = pd.merge(vr_with_number, gsisid_numbers, how='left',
                          left_on='primary_partner_gsisid', right_on='gsisid', suffixes=('', '_primary_partner'))
vr = vr_with_number

vr_merged = pd.merge(vr, pprd)
vr_merged = pd.merge(vr_merged, role_info)


vr_merged = pd.merge(vr_merged, pprd, left_on=['season_year', 'gamekey', 'playid', 'primary_partner_gsisid'],
                     right_on=['season_year', 'gamekey', 'playid', 'gsisid'], how='left',
                     suffixes=('', '_primary_partner'))
vr_merged = pd.merge(vr_merged, role_info, left_on='role_primary_partner',
                     right_on='role', how='left', suffixes=('', '_primary_partner'))

vr_merged = vr_merged.fillna('None')
vr_merged['count'] = 1

vr_merged['generalized_role'] = vr_merged['generalized_role'].str.replace(
    '_', ' ')
vr_merged['generalized_role_primary_partner'] = vr_merged['generalized_role_primary_partner'].str.replace(
    '_', ' ')
ax = vr_merged.groupby('primary_impact_type').count() \
    .sort_values('count')['count'].plot(kind='barh',
                                        figsize=(15, 5),
                                        color='grey',
                                        title='Primary Impact Type on Concussion Plays')
rects = ax.patches
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{}".format(x_value)

    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.
ax.set_ylabel('')
plt.show()
vr_merged['derived_text'] = \
    vr_merged.apply(lambda row: 'Injured player was {}, Iinjuring player was {}'.format(row['player_activity_derived'],
                                                                                        row['primary_partner_activity_derived'])
                    if type(row['primary_partner_activity_derived']) is str
                    else 'Injured player was {}'.format(row['player_activity_derived']), axis=1)
ax = vr_merged.groupby('derived_text').count()['count'] \
    .sort_values() \
    .plot(kind='barh',
          figsize=(15, 5),
          color='grey',
          title='Activity of Players involved in Concussion')

rects = ax.patches
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{}".format(x_value)

    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.
ax.set_ylabel('')
ax.grid(True)
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))
vr_merged \
    .groupby('punting_returning_team').count()['count'] \
    .rename(index={'Punting_Team': 'Punting Team',
                   'Returning_Team': 'Returning Team'}) \
    .plot(kind='bar',
          title='Concussions by Team',
          color='grey',
          rot=0,
          ax=ax1)
vr_merged \
    .groupby('punting_returning_team_primary_partner').count()['count'] \
    .rename(index={'Punting_Team': 'Punting Team',
                   'Returning_Team': 'Returning Team'}) \
    .plot(kind='bar',
          title='Primary Partner by Team',
          color='grey',
          rot=0,
          ax=ax2)
ax1.set_xlabel('')
ax2.set_xlabel('')
ax1.set_ylabel('Count')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(20, 4))

ax1 = vr_merged.groupby('generalized_role') \
    .sum()['count'] \
    .sort_values() \
    .plot(kind='barh',
          color='grey',
          title='Count of Injured Players by Generalized Role',
          ax=ax1)

rects = ax1.patches
ax1.grid(False)
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    ax1.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.

ax2 = vr_merged.groupby('generalized_role_primary_partner') \
    .sum()['count'] \
    .sort_values() \
    .plot(kind='barh',
          color='grey',
          title='Count of Primary Partner in Injury by Generalized Role',
          ax=ax2)

rects = ax2.patches
ax2.grid(False)
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    ax2.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.
ax1.set_ylabel('')
ax2.set_ylabel('')
ax1.grid()
ax2.grid()
plt.subplots_adjust(left=0.02)
plt.show()
ax = pd.concat([vr_merged[['generalized_role_primary_partner',
                           'punting_returning_team_primary_partner']]
                .rename(columns={'generalized_role_primary_partner': 'generalized_role',
                                 'punting_returning_team_primary_partner': 'punting_returning_team'}),
                vr_merged[['generalized_role', 'punting_returning_team']]]) \
    .groupby(['generalized_role']) \
    .count() \
    .sort_values('punting_returning_team') \
    .drop('None') \
    .plot(kind='barh',
          legend=False,
          title='Roles involved in Concussion Plays (Injured or Partner)',
          figsize=(15, 5), color='grey')


rects = ax.patches
ax.grid(False)
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points",  # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
    # positive and negative values.
ax.set_xticklabels('')
ax.set_ylabel('Generalized Role')
ax.grid()
plt.show()
vr_merged.groupby(['generalized_role', 'generalized_role_primary_partner']) \
    .sum()['count'] \
    .reset_index() \
    .rename(columns={'generalized_role': 'Injured Player',
                     'generalized_role_primary_partner': 'Primary Partner'}) \
    .sort_values('count', ascending=False).reset_index(drop=True)
"""
This cell block contains functions for interacting with the NGS data.
Plotting compass of player angle and velocity along with the playing field
"""


def compass(angles, radii, arrowprops=None, ax=None):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    #angles, radii = cart2pol(u, v)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return ax


def trim_play_action(df):
    """
    Trims a play to only the duration of action
    """
    if len(df.loc[df['event'] == 'ball_snap']['time'].values) == 0:
        print('........No Snap for this play')
        ball_snap_time = df['time'].min()
    else:
        ball_snap_time = df.loc[df['event'] ==
                                'ball_snap']['time'].values.min()

    try:
        end_time = df.loc[(df['event'] == 'out_of_bounds') |
                          (df['event'] == 'downed') |
                          (df['event'] == 'tackle') |
                          (df['event'] == 'punt_downed') |
                          (df['event'] == 'fair_catch') |
                          (df['event'] == 'touchback') |
                          (df['event'] == 'touchdown')]['time'].values.max()
    except ValueError:
        end_time = df['time'].values.max()
    df = df.loc[(df['time'] >= ball_snap_time) & (df['time'] <= end_time)]
    return df


def plot_injury_play(season_year, gamekey, playid,
                     plot_velocity=False, ax3=None, display_url=False,
                     figsize_velocity=(5, 4),
                     **kwargs):
    vr_thisplay = vr.loc[(vr['season_year'] == season_year) &
                         (vr['playid'] == playid) &
                         (vr['gamekey'] == gamekey)]

    play = injury_play_ngs.loc[(injury_play_ngs['season_year'] == season_year) &
                               (injury_play_ngs['playid'] == playid) &
                               (injury_play_ngs['gamekey'] == gamekey)].copy()

    # Calculate velocity in meters per second
    play['dis_meters'] = play['dis'] / 1.0936  # Add distance in meters
    # Speed
    play['dis_meters'] / 0.01
    play['v_mps'] = play['dis_meters'] / 0.1

    # Filter to only duration of play
    play = trim_play_action(play)

    # play = pd.read_csv('../working/playlevel/during_play/{}-{}-{}.csv'.format(season_year, gamekey, playid))
    play['dir_theta'] = play['dir'] * np.pi / 180

    # Video footage link
    url_link = vfi.loc[(vfi['season_year'] == season_year) &
                       (vfi['playid'] == playid) &
                       (vfi['gamekey'] == gamekey)]['preview link (5000k)'].values[0]

    playdescription = vfi.loc[(vfi['season_year'] == season_year) &
                              (vfi['playid'] == playid) &
                              (vfi['gamekey'] == gamekey)]['playdescription'].values[0]
    if display_url:
        display(HTML("""<a href="{}">Play-Video-Link</a>""".format(url_link)))
        print('Injured player number {} was injured while {} with primary impact {}'
              .format(vr_thisplay['number'].values[0],
                      vr_thisplay['player_activity_derived'].values[0],
                      vr_thisplay['primary_impact_type'].values[0]))

    # Determine time of injury
    injured = play.loc[play['injured_player']]
    primarypartner = play.loc[play['primary_partner_player']]
    injury_time = None
    if len(primarypartner) != 0:
        inj_and_pp = pd.merge(injured[['time', 'x', 'y']], primarypartner[[
                              'time', 'x', 'y']], on='time', suffixes=('_inj', '_pp'))
        inj_and_pp['dis_from_eachother'] = np.sqrt(np.square(inj_and_pp['x_inj'] -
                                                             inj_and_pp['x_pp']) +
                                                   np.square(inj_and_pp['y_inj'] -
                                                             inj_and_pp['y_pp']))
        injury_time = inj_and_pp.sort_values('dis_from_eachother')[
            'time'].values[0]
    # PLOT
    fig, ax3 = create_football_field(**kwargs)

    # Plot path of injured player
    d = play.loc[play['injured_player']]
    injured_player_role = play.loc[play['injured_player']]['role'].values[0]
    d.plot('x', 'y', kind='scatter', ax=ax3,  zorder=5, color='blue', alpha=0.3,
           xlim=(0, 120), ylim=(0, 53.3),
           label='Injured Player Path - Role: {}'.format(injured_player_role))  # Plot injured player path
    play.loc[(play['punting_returning_team'] == 'Returning_Team') &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='purple', ax=ax3, zorder=5, style='+',
                                                  label='Returning Team Player')
    play.loc[(play['punting_returning_team'] == 'Punting_Team') &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='orange', ax=ax3, zorder=4, style='+',
                                                  label='Punting Team Player')
    start_pos = d.loc[d['time'] == d['time'].min()]
    inj_star_pos = ax3.scatter(start_pos['x'], start_pos['y'], color='red',
                               zorder=5, label='Injured Player Starting Position')
    end_pos = d.loc[d['time'] == d['time'].max()]
    ax3.scatter(end_pos['x'], end_pos['y'], color='black',
                zorder=5, label='Injured Player Ending Position')
    if injury_time:
        inj_pos = d.loc[d['time'] == injury_time]
    pp_player_role = None
    if len(primarypartner) != 0:
        pp_player_role = play.loc[play['primary_partner_player']
                                  ]['role'].values[0]
        play.loc[play['primary_partner_player']].plot('x', 'y', kind='scatter',
                                                      xlim=(0, 120), ylim=(0, 53.3),
                                                      ax=ax3, color='yellow', alpha=0.3, zorder=3,
                                                      label='Primary Partner Path - Role {}'.format(pp_player_role))
        ax3.scatter(inj_pos['x'],
                    inj_pos['y'],
                    color='red',
                    zorder=5,
                    s=50,
                    marker='+',
                    label='Aproximate Location of Injury')
    play_info_string = 'Season {} - Gamekey {} - Playid {}'.format(
        season_year, gamekey, playid)
    injured_player_string = 'Injured Player Number: {} - action {}' \
        .format(vr_thisplay['number'].values[0],
                vr_thisplay['player_activity_derived'].values[0])
    primary_partner_string = 'Primary Partner Player Number: {} - action {}' \
        .format(vr_thisplay['number_primary_partner'].values[0],
                vr_thisplay['primary_partner_activity_derived'].values[0])
    # Plot punt return path if not one of the players.
    if (injured_player_role != 'PR') and (pp_player_role != 'PR'):
        punt_returner = play.loc[play['role'] == 'PR']
        punt_returner.plot('x', 'y', kind='scatter', ax=ax3,  zorder=3, color='white', alpha=0.3,
                           label='Punt Returner Path')
    # print(playdescription)
    plt.suptitle(play_info_string, fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if plot_velocity:
        # Plot injured player compass

        fig3, (ax1, ax2) = plt.subplots(
            1, 2, subplot_kw=dict(polar=True), figsize=figsize_velocity)

        d = play.loc[play['injured_player']]
        role = d.role.values[0]

        ax1 = compass(d['dir_theta'], d['v_mps'],
                      arrowprops={'alpha': 0.3}, ax=ax1)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_title('Injured Player: {}'.format(role))
        # Color point of time when inujury happened
        if len(primarypartner) != 0:
            theta_at_inj = d.loc[d['time'] ==
                                 injury_time]['dir_theta'].values[0]
            dis_at_inj = d.loc[d['time'] == injury_time]['v_mps'].values[0]
            impact_arrow = ax1.annotate("",
                                        xy=(theta_at_inj, dis_at_inj), xytext=(
                                            0, 0),
                                        arrowprops={'color': 'orange'},
                                        label='Aproximate Point of Impact')  # use cir mean
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.suptitle('Velocity and Direction (Injured Player): {}'.format(role), x=0.52, y=1.01, fontsize=15)

        if len(primarypartner) != 0:
            # Plot primary partner compass
            d = play.loc[play['primary_partner_player']]
            role = d.role.values[0]
            ax2 = compass(d['dir_theta'], d['v_mps'],
                          arrowprops={'alpha': 0.3}, ax=ax2)
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_title('Primary Partner: {}'.format(role))
            # Color point of time when inujury happened
            theta_at_inj = d.loc[d['time'] ==
                                 injury_time]['dir_theta'].values[0]
            dis_at_inj = d.loc[d['time'] == injury_time]['v_mps'].values[0]
            ax2.annotate("", xy=(theta_at_inj, dis_at_inj), xytext=(
                0, 0), arrowprops={'color': 'orange'})  # use cir mean
            # plt.suptitle('Velocity and Direction (Primary Partner): {}'.format(role), x=0.52, y=1.01, fontsize=15)
            plt.show()
    return ax3
# Load preprocessed injury data (preprocessing done in seperate kernel)
injury_play_ngs = pd.read_parquet('../input/nfl-punt-data-preprocessing-ngs-injury-plays/NGS-injury-plays.parquet')
injury_play_ngs.loc[injury_play_ngs['role'] == 'PFB', 'punting_returning_team'] = 'Returning_Team' # Fix typo in preprocessing
# Plot all plays
fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(23,30))
axes_list = [item for sublist in axes for item in sublist] 

count = 0
for i, play in injury_play_ngs.groupby(['season_year','gamekey','playid']):
    
    #print(i)
    # Determine time of injury
    injured = play.loc[play['injured_player']]
    primarypartner = play.loc[play['primary_partner_player']]
    injury_time = None
    if len(primarypartner) != 0:
        inj_and_pp = pd.merge(injured[['time','x','y']],
                              primarypartner[['time','x','y']],
                              on='time',
                              suffixes=('_inj','_pp'))
        inj_and_pp['dis_from_eachother'] = np.sqrt(np.square(inj_and_pp['x_inj'] -
                                                             inj_and_pp['x_pp']) +
                                                   np.square(inj_and_pp['y_inj'] - 
                                                             inj_and_pp['y_pp']))
        injury_time = inj_and_pp.sort_values('dis_from_eachother')['time'].values[0]
    
    play = play.copy()
    # Filter to only duration of play
    play = trim_play_action(play)

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)
    figsize=(5, 3.3)
    ax = axes_list.pop(0)
    ax.add_patch(rect)

    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white', linewidth=0.5)
    # Endzones

    ez1 = patches.Rectangle((0, 0), 10, 53.3,
                            linewidth=0.1,
                            edgecolor='r',
                            facecolor='blue',
                            alpha=0.2,
                            zorder=0)
    ez2 = patches.Rectangle((110, 0), 120, 53.3,
                            linewidth=0.1,
                            edgecolor='r',
                            facecolor='blue',
                            alpha=0.2,
                            zorder=0)
    ax.add_patch(ez1)
    ax.add_patch(ez2)
    # Line Numbers
    for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=6, # fontname='Arial',
                     color='white')
            ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=6, # fontname='Arial',
                     color='white', rotation=180)

    hash_range = range(11, 110)
    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white', linewidth=0.4)
        ax.plot([x, x], [53.0, 52.5], color='white', linewidth=0.4)
        ax.plot([x, x], [22.91, 23.57], color='white', linewidth=0.4)
        ax.plot([x, x], [29.73, 30.39], color='white', linewidth=0.4)
        
        # Plot path of injured player
    d = play.loc[play['injured_player']]
    injured_player_role = play.loc[play['injured_player']]['role'].values[0]
    d.plot('x', 'y', kind='scatter', ax=ax,  zorder=5, color='blue', alpha=0.3,
           xlim = (0, 120), ylim=(0,53.3),
           label='Injured Player Path - Role: {}'.format(injured_player_role)) #Plot injured player path
    play.loc[(play['punting_returning_team'] == 'Returning_Team') &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='purple', ax=ax, zorder=5, style='+',
                                                  label='Returning Team Player')
    play.loc[(play['punting_returning_team'] == 'Punting_Team')  &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='orange', ax=ax, zorder=4, style='+',
                                                 label='Punting Team Player')
    start_pos = d.loc[d['time'] == d['time'].min()]
    inj_star_pos = ax.scatter(start_pos['x'], start_pos['y'], color='red',
                               zorder=5, label='Injured Player Starting Position')
    end_pos = d.loc[d['time'] == d['time'].max()]
    ax.scatter(end_pos['x'], end_pos['y'], color='black',
                zorder=5, label='Injured Player Ending Position')
    if injury_time:
        inj_pos = d.loc[d['time'] == injury_time]
    
    # Plot the primary partner path if it exists
    pp_player_role = None
    if len(primarypartner) != 0:
        pp_player_role = play.loc[play['primary_partner_player']]['role'].values[0]
        play.loc[play['primary_partner_player']].plot('x', 'y', kind='scatter',
                                                      xlim = (0, 120), ylim=(0,53.3),
                                                      ax=ax, color='yellow', alpha=0.3, zorder=3,
                                                      label='Primary Partner Path - Role {}'.format(pp_player_role))
        ax.scatter(inj_pos['x'],
                    inj_pos['y'],
                    color='red', 
                    zorder=5,
                    s=50,
                    marker='+',
                    label='Aproximate Location of Injury')
    # If injured player or partner are not the punt returner plot the PR path
    if (injured_player_role != 'PR') and (pp_player_role != 'PR'):
        punt_returner = play.loc[play['role'] == 'PR']
        punt_returner.plot('x', 'y', kind='scatter', ax=ax,  zorder=3, color='white', alpha=0.3,
                           label='Punt Returner Path')

    ax.axis('off')
    ax.get_legend().remove()

# Final 3 plots are nothing
ax38 = axes_list.pop(0)
ax38.axis('off')
ax39 = axes_list.pop(0)
ax39.axis('off')
ax40 = axes_list.pop(0)
ax40.axis('off')
plt.show()
plot_injury_play(2016, 5,   3129, figsize=(10, 5), display_url=True)
plt.show()
# plot_injury_play(2016, 29,  538, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 189, 3509, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 234, 3278, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 266, 2902, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 280, 3746, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 357, 3630, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 384, 183, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 397, 1526, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 399, 3312, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 506, 1988, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 585, 2208, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 601, 602, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 21, 2587, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 289, 2341, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 364, 2764, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 392, 1088, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 448, 2792, figsize=(10, 5))
# plt.show()
plot_injury_play(2017, 553, 1683, figsize=(10, 5), display_url=True)
plt.show()
# plot_injury_play(2017, 585, 733, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 54, 1045, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 231, 1976, figsize=(10, 5))
# plt.show()
# plot_injury_play(2017, 618, 2792, figsize=(10, 5))
# plt.show()
ax = plot_injury_play(2017, 607, 978, figsize=(10, 5), display_url=True)
plt.show()
# plot_injury_play(2016, 60, 905, figsize=(10, 5))
# plt.show()
# plot_injury_play(2016, 280, 2918, figsize=(10, 5))
# plt.show()
plot_injury_play(2016, 218, 3468, figsize=(10, 5), display_url=True)
plt.show()
# plot_injury_play(2017, 414, 1262, figsize=(10, 5))
# plt.show()
plot_injury_play(2016, 296, 2667, figsize=(10, 5), display_url=True)
plt.show()
# plot_injury_play(2017, 473, 2072, figsize=(10, 5))
# plt.show()
plot_injury_play(2016, 274, 3609, figsize=(10, 5), display_url=True)
plt.show()
plot_injury_play(2017, 448, 2792, figsize=(10, 5), plot_velocity=True,
                 figsize_velocity=(15, 5), display_url=True)
plt.show()
plot_injury_play(2017, 473, 2072, figsize=(10, 5), plot_velocity=True,
                 figsize_velocity=(15, 5), display_url=True)
plt.show()
fc = pd.read_parquet('../input/robmullanflpreprocessed/position_at_faircatch.parquet')
fc['y_rel_plus26'] = fc['y_rel'] + 26
prec = pd.read_parquet('../input/robmullanflpreprocessed/position_at_punt_recieved.parquet')
prec['y_rel_plus26'] = prec['y_rel'] + 26
fig, ax = create_football_field(endzones=False, linenumbers=False)
prec['y_rel_plus26'] = prec['y_rel'] + 26.65
prec['x_rel_left_to_right_plus40'] = prec['x_rel_left_to_right'] + 40
prec.plot(x='x_rel_left_to_right_plus40',
          y='y_rel_plus26',
          kind='scatter',
          ax=ax, alpha=0.1,
          title='All Players `Relative to Punt Returned',
          color='yellow')
# Add injured and primary partners
prec.loc[prec['injured_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Gunners Relative to Punt Returner (Injury Colored)',
          color='red', zorder=3, label='Injured Player')
prec.loc[prec['primary_partner_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Gunners Relative to Punt Returner (Injury Colored)',
          color='orange', zorder=3, label='Primary Partner Player')
caught_point = plt.plot(40, 26.65, '>', color='blue', markersize=10, zorder=5,
                        label='Punt Recieved (Relative Location)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

fig, ax = create_football_field(endzones=False, linenumbers=False)
fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc.plot(x='x_rel_left_to_right_plus40',
        y='y_rel_plus26',
        kind='scatter', ax=ax, alpha=0.1,
        title='All Players Relative to Faircatch',
        color='orange',
        zorder=2)
fc.loc[fc['injured_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Gunners Relative to Punt Returner (Injury Colored)',
          color='red', zorder=3, label='Injured Player')

fc.loc[fc['primary_partner_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Gunners Relative to Punt Returner (Injury Colored)',
          color='white', zorder=3, label='Primary Partner Player')
caught_point = plt.plot(40, 26.65, '>', color='blue', markersize=10,
                        zorder=5, label='Fair Catch (Relative Location)')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
fc['distance_from_rec'] = np.sqrt(np.square(fc['x_rel']) + np.square(fc['y_rel']))
prec['distance_from_rec'] = np.sqrt(np.square(prec['x_rel']) + np.square(prec['y_rel']))

plt.style.use('ggplot')

fc.loc[fc['punting_returning_team'] == 'Punting_Team']['distance_from_rec'] \
    .plot(kind='hist', figsize=(15,5),
          bins=50,
          title='Punting Team Distance from Punt Returner (Fair Catch)',
          color=color1)
plt.show()

prec.loc[prec['punting_returning_team'] == 'Punting_Team']['distance_from_rec'] \
    .plot(kind='hist',
          figsize=(15,5),
          bins=50,
          title='Punting Team Distance from Punt Returner (Punt Returned)',
          color=color2)
plt.show()
fig, ax = create_football_field(endzones=False, linenumbers=False)
prec['x_rel_left_to_right_plus40'] = prec['x_rel_left_to_right'] + 40
prec['y_rel_plus26'] = prec['y_rel'] + 26.65

prec.loc[prec['role'].isin(['GL', 'GLi', 'GR', 'GRi'])] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=0.2, title='Players Relative to Punt Returned Punting Team',
          color='yellow', zorder=3, label='Closest Player')
prec.loc[prec['role'].isin(['GL', 'GLi', 'GR', 'GRi']) & prec['injured_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Players Relative to Punt Returned Punting Team',
          color='red', zorder=3, label='Injured Player')
prec.loc[prec['role'].isin(['GL', 'GLi', 'GR', 'GRi']) & prec['primary_partner_player']] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=1, title='Gunners Relative to Punt Returner (Injury Colored)',
          color='orange', zorder=3, label='Primary Partner Player')
caught_point = plt.plot(40, 26.65, '>', color='blue', markersize=10,
                        zorder=5, label='Return Catch Point (Relative Location)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

fig, ax = create_football_field(endzones=False, linenumbers=False)

fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc['y_rel_plus26'] = fc['y_rel'] + 26.65

fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc.loc[fc['role'].isin(['GL', 'GLi', 'GR', 'GRi'])] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=0.2, title='Gunners Relative to Fair Catch Punting Team',
          color='orange', zorder=3, label='Closest Player to Returner')
caught_point = plt.plot(40, 26.65, '>', color='blue', markersize=10,
                        zorder=5, label='Fair Catch (Relative Location)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
fc_closest_player = fc.loc[fc['punting_returning_team'] == 'Punting_Team'] \
    .groupby(['season_year', 'gamekey', 'playid'])[['role', 'distance_from_rec']] \
    .min().reset_index()
prec_closest_player = prec.loc[prec['punting_returning_team'] == 'Punting_Team'] \
    .groupby(['season_year', 'gamekey', 'playid'])[['role', 'distance_from_rec']].min().reset_index()

fc_closest_player['distance_from_rec'].plot(kind='hist',
                                            figsize=(15, 5),
                                            bins=50,
                                            title='Closest Player Distance from Punt Returner (Fair Catch)',
                                            color=color1)
plt.show()
prec_closest_player['distance_from_rec'].plot(kind='hist',
                                              figsize=(15, 5),
                                              bins=50,
                                              title='Closest Player Distance from Punt Returner (Returned)',
                                              color=color2)
plt.show()
print('Median closest player to punter returner on a return is {:.2f} yards'.format(prec_closest_player['distance_from_rec'].median()))
print('Median closest player to punter returner on a fair catch is {:.2f} yards'.format(fc_closest_player['distance_from_rec'].median()))
fig, ax = create_football_field(endzones=False, linenumbers=False)

non_gunners_punting_team_roles = ['P', 'PC', 'PPR', 'PPRi', 'PPRo', 'PFB', 'PLG',
                                  'PLS', 'PLT', 'PLW', 'PRW', 'PRG', 'PRT', 'PPL',
                                  'PPLo', 'PPLi']

fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc['y_rel_plus26'] = fc['y_rel'] + 26.65

fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc.loc[fc['role'].isin(non_gunners_punting_team_roles)] \
    .plot(x='x_rel_left_to_right_plus40', y='y_rel_plus26', kind='scatter', ax=ax,
          alpha=0.2,
          title='Punting Team - Non-Gunners Relative to Fair Catch Punt Returner at Catch',
          color='orange',
          zorder=3)
caught_point = plt.plot(40, 26.65, '>', color='blue', markersize=10,
                        zorder=5, label='Fair Catch (Relative Location)')
plt.legend(handles=caught_point)
plt.show()

fig, ax = create_football_field(endzones=False, linenumbers=False)
prec['x_rel_left_to_right_plus40'] = prec['x_rel_left_to_right'] + 40
prec['y_rel_plus26'] = prec['y_rel'] + 26.65

prec.loc[~prec['role'].isin(non_gunners_punting_team_roles)] \
    .plot(x='x_rel_left_to_right_plus40',
          y='y_rel_plus26',
          kind='scatter', ax=ax,
          alpha=0.2,
          title='Players Relative to Punt Returned Punting Team', color='yellow', zorder=3)
prec.loc[prec['role'].isin(non_gunners_punting_team_roles) & prec['injured_player']] \
    .plot(x='x_rel_left_to_right_plus40',
          y='y_rel_plus26',
          kind='scatter',
          ax=ax,
          alpha=1,
          title='Players Relative to Punt Returned Punting Team',
          color='red',
          label='injured gunners',
          zorder=3)
prec.loc[prec['role'].isin(non_gunners_punting_team_roles) & prec['primary_partner_player']] \
    .plot(x='x_rel_left_to_right_plus40',
          y='y_rel_plus26',
          kind='scatter',
          ax=ax,
          alpha=1,
          title='Punting Team - Non-Gunners Relative to Punt Returner at Moment of Catch (Injury Colored)',
          color='orange',
          zorder=3)
caught_point = plt.plot(40, 26.65, '>',
                        color='blue',
                        markersize=10,
                        zorder=5,
                        label='Return Caught (Relative Location)')
plt.legend(handles=caught_point)
plt.show()
pprd['count'] = 1
#unique key to join on
pprd['play_unique'] = pprd['season_year'].astype('str').add(
    (pprd['gamekey']).astype('str')).add((pprd['playid']).astype('str'))
vr['play_unique'] = vr['season_year'].astype('str').add(
    (vr['gamekey']).astype('str')).add((vr['playid']).astype('str'))

play_roles = pprd.pivot_table(values='count', index='play_unique',
                              columns='role', aggfunc='mean').fillna(0).reset_index()
play_roles.drop(columns=['play_unique']) \
    .sum().sort_values() \
    .plot(kind='barh', figsize=(15, 10), title='Number of Plays with Role', color='grey')
plt.show()
role_count_df = pd.DataFrame(play_roles.drop(columns=['play_unique'])
                             .sum()
                             .sort_values()) \
    .reset_index() \
    .rename(columns={0: 'Role Count', 'Role': 'role'})
role_count_df = pd.merge(role_count_df, role_info)
role_count_df.loc[role_count_df['punting_returning_team'] == 'Returning_Team'] \
    .set_index('role')['Role Count'] \
    .plot(kind='barh',
          figsize=(15, 10),
          title='Returning Team - Number of Plays with Role',
          color='grey')
plt.show()
role_count_df.loc[role_count_df['punting_returning_team'] == 'Punting_Team'] \
    .set_index('role')['Role Count'] \
    .plot(kind='barh',
          figsize=(15, 10),
          title='Punting Team - Number of Plays with Role',
          color='grey')
plt.show()
play_roles['Number_of_jammers'] = play_roles[['VL', 'VLi',
                                              'VLo', 'VR', 'VRi', 'VRo']].sum(axis=1).astype('int')
ax = ((play_roles.groupby('Number_of_jammers').count())['play_unique'] / len(play_roles) * 100) \
    .plot(kind='bar',
          figsize=(15, 5),
          title='Percentage of Punt Plays by Jammers',
          color='grey',
          rot=0)
ax.set_ylabel('Percentage of All Punt Plays')
ax.set_xlabel('Number of Jammers')

plt.show()
vr_merged['injury_play'] = True

pi['play_unique'] = pi['season_year'].astype('str') \
    .add((pi['gamekey']).astype('str')) \
    .add((pi['playid']).astype('str'))

vr_merged['play_unique'] = vr_merged['season_year'].astype('str') \
    .add((vr_merged['gamekey']).astype('str')) \
    .add((vr_merged['playid']).astype('str'))

play_roles2 = pd.merge(pd.merge(play_roles, pi, on='play_unique', how='left', suffixes=('', 'y')),
                       vr_merged, on='play_unique', how='left', suffixes=('', 'y'))

ax = (play_roles2
      .groupby('Number_of_jammers')['injury_play'].sum() * 100 / play_roles.groupby('Number_of_jammers')['play_unique']
      .count()) \
    .drop([0, 1, 5]) \
    .plot(kind='bar',
          figsize=(15, 5),
          color='grey',
          title='Rate of Concussions by # of Jammers',
          rot=0)
ax.set_ylabel('Rate (%) of Injuries')
ax.set_xlabel('Number of Jammers')
plt.show()
# Fair catch merge
fc['play_unique'] = fc['season_year'].astype('str') \
    .add((fc['gamekey']).astype('str')) \
    .add((fc['playid']).astype('str'))
fc['x_rel_left_to_right_plus40'] = fc['x_rel_left_to_right'] + 40
fc['y_rel_plus26'] = fc['y_rel'] + 26.65
ngs_fair_catch = pd.merge(
    fc, play_roles2, on='play_unique', suffixes=('', 'y'))

# Reception data merge
prec['play_unique'] = prec['season_year'].astype('str') \
    .add((prec['gamekey']).astype('str')) \
    .add((prec['playid']).astype('str'))
prec['x_rel_left_to_right_plus40'] = prec['x_rel_left_to_right'] + 40
prec['y_rel_plus26'] = prec['y_rel'] + 26.65
ngs_return = pd.merge(prec, play_roles2, on='play_unique', suffixes=('', 'y'))
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 5))
ngs_return_small = ngs_return[['season_year', 'gamekey', 'playid',
                               'punting_returning_team', 'role', 'distance_from_rec',
                               'Number_of_jammers']]

# Return data filtered
ngs_return_small = ngs_return_small.loc[ngs_return_small['Number_of_jammers'].isin([
                                                                                   2, 3, 4])]

ngs_return_small.loc[ngs_return_small['punting_returning_team'] == 'Punting_Team'] \
    .groupby(['season_year', 'gamekey', 'playid', 'Number_of_jammers'])[['role', 'distance_from_rec']].min() \
    .reset_index() \
    .groupby('Number_of_jammers') \
    .mean()['distance_from_rec'].plot(kind='bar',
                                      color='grey',
                                      title='Distance from Returner (Return)',
                                      rot=0,
                                      ax=ax1)

ngs_fair_catch_small = ngs_fair_catch[['season_year', 'gamekey', 'playid',
                                       'punting_returning_team', 'role', 'distance_from_rec',
                                       'Number_of_jammers']]

ngs_fair_catch_small = ngs_fair_catch_small.loc[ngs_fair_catch_small['Number_of_jammers'].isin([
                                                                                               2, 3, 4])]
ngs_fair_catch_small.loc[ngs_fair_catch_small['punting_returning_team'] == 'Punting_Team'] \
    .groupby(['season_year', 'gamekey', 'playid', 'Number_of_jammers'])[['role', 'distance_from_rec']].min() \
    .reset_index() \
    .groupby('Number_of_jammers') \
    .mean()['distance_from_rec'].plot(kind='bar',
                                      color='grey',
                                      title='Distance from Returner (Fair Catch)',
                                      rot=0,
                                      ax=ax2)
ax1.set_xlabel('Number of Jammers')
ax2.set_xlabel('Number of Jammers')
ax1.set_ylabel('Average Distance of Opponents (yards)')
plt.show()
def create_generalized_role_paths(generalized_role, legend=True):
    fig, ax = create_football_field(
        linenumbers=False, endzones=False, fifty_is_los=True)

    role_df_all = pd.read_parquet(
        '../input/robmullanflpreprocessed/{}.parquet'.format(generalized_role))
    role_df_all['x-rel-snap-plus50'] = role_df_all['x-rel-snap'] + 60
    role_df_all['x-rel-snap-plus50'] = pd.to_numeric(
        role_df_all['x-rel-snap-plus50'])
    role_df_all.plot(x='x-rel-snap-plus50', y='y', kind='scatter',
                     ax=ax, alpha=0.05, color='white', s=1,
                     label='All {}s Routes'.format(generalized_role),
                     title='{} Routes on Punt Plays'.format(generalized_role.replace('_', ' ')))

    role_df_all[role_df_all['injured_player']].plot(x='x-rel-snap-plus50', y='y',
                                                    kind='scatter', s=2,
                                                    color='red', ax=ax, label='Injured Players Routes')
    if len(role_df_all[role_df_all['primary_partner_player']]) > 0:
        role_df_all[role_df_all['primary_partner_player']] \
            .plot(x='x-rel-snap-plus50', y='y',
                  kind='scatter', s=2,
                  color='orange', ax=ax, label='Primary Partner Routes')
    if legend is True:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.get_legend().remove()
    return fig, ax
fig, ax = create_generalized_role_paths('Punt_Returner', legend=False)
plt.show()
fig, ax = create_generalized_role_paths('Punting_Lineman', legend=False)
plt.show()
remaining_positions = ['Gunner',
                       'Punter',
                       'Punter_Protector', 'Defensive_Lineman',
                       'PuntFullBack',
                       #'Punting_Lineman',
                       'Defensive_Backer',
                       'Punting_Longsnapper', 'Punting_Wing',
                       #'Punt_Returner',
                       'Jammer']

for role in remaining_positions:
    try:
        # print(role)
        fig, ax = create_generalized_role_paths(role)
        plt.show()
    except Exception as e:
        print(e)
role_df_all = pd.read_parquet('../input/robmullanflpreprocessed/{}.parquet'.format('Gunner'))
role_df_all['play_unique'] = role_df_all['season_year'].astype('str') \
    .add((role_df_all['gamekey']).astype('str')) \
    .add((role_df_all['playid']).astype('str'))

gunner_routs_with_coverage = pd.merge(role_df_all, play_roles[['play_unique','Number_of_jammers']], how='left')

gunner_single_covered = \
    gunner_routs_with_coverage.loc[gunner_routs_with_coverage['Number_of_jammers'] == 2].copy()
gunner_double_covered = \
    gunner_routs_with_coverage.loc[gunner_routs_with_coverage['Number_of_jammers'] == 4].copy()

# Single Covered
fig, ax1 = create_football_field(linenumbers=False, endzones=False, fifty_is_los=True)
legend = False
gunner_single_covered['x-rel-snap-plus50'] = gunner_single_covered['x-rel-snap'] + 60
gunner_single_covered['x-rel-snap-plus50'] = pd.to_numeric(gunner_single_covered['x-rel-snap-plus50'])
gunner_single_covered.plot(x='x-rel-snap-plus50',y='y', kind='scatter',
                 ax=ax1, alpha=0.05, color='white', s=1,
                 label='Gunner Routes',
                 title='Gunner Routes Single Coverage')

ax1.get_legend().remove()

# Double Covered
fig, ax2 = create_football_field(linenumbers=False, endzones=False, fifty_is_los=True)
legend = False
gunner_double_covered['x-rel-snap-plus50'] = gunner_double_covered['x-rel-snap'] + 60
gunner_double_covered['x-rel-snap-plus50'] = pd.to_numeric(gunner_double_covered['x-rel-snap-plus50'])
gunner_double_covered.plot(x='x-rel-snap-plus50',y='y', kind='scatter',
                 ax=ax2, alpha=0.05, color='white', s=1,
                 label='Gunner Routes',
                 title='Gunner Routes Double Coverage')

ax2.get_legend().remove()
role_df_all_dline = pd.read_parquet(
    '../input/robmullanflpreprocessed/Defensive_Lineman.parquet')
role_df_all_puntline = pd.read_parquet(
    '../input/robmullanflpreprocessed/Punting_Lineman.parquet')
role_df_all = pd.concat([role_df_all_dline, role_df_all_puntline])
role_df_all['play_unique'] = role_df_all['season_year'].astype('str') \
    .add((role_df_all['gamekey']).astype('str')) \
    .add((role_df_all['playid']).astype('str'))

gunner_routs_with_coverage = pd.merge(
    role_df_all, play_roles[['play_unique', 'Number_of_jammers']], how='left')

gunner_single_covered = \
    gunner_routs_with_coverage.loc[gunner_routs_with_coverage['Number_of_jammers'] == 2].copy()
gunner_hybrid_covered = \
    gunner_routs_with_coverage.loc[gunner_routs_with_coverage['Number_of_jammers'] == 3].copy()
gunner_double_covered = \
    gunner_routs_with_coverage.loc[gunner_routs_with_coverage['Number_of_jammers'] == 4].copy()

# Single Covered
fig, ax1 = create_football_field(
    linenumbers=False, endzones=False, fifty_is_los=True)
legend = False
gunner_single_covered['x-rel-snap-plus50'] = gunner_single_covered['x-rel-snap'] + 60
gunner_single_covered['x-rel-snap-plus50'] = pd.to_numeric(
    gunner_single_covered['x-rel-snap-plus50'])
gunner_single_covered.plot(x='x-rel-snap-plus50', y='y', kind='scatter',
                           ax=ax1, alpha=0.05, color='white', s=1,
                           label='Punting Linemen Routes',
                           title='Punting Linemen Routes Single Coverage')
if legend is True:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax1.get_legend().remove()
plt.show()

# Hybrid
fig, ax3 = create_football_field(
    linenumbers=False, endzones=False, fifty_is_los=True)
legend = False
gunner_hybrid_covered['x-rel-snap-plus50'] = gunner_hybrid_covered['x-rel-snap'] + 60
gunner_hybrid_covered['x-rel-snap-plus50'] = pd.to_numeric(
    gunner_hybrid_covered['x-rel-snap-plus50'])
gunner_hybrid_covered.plot(x='x-rel-snap-plus50', y='y', kind='scatter',
                           ax=ax3, alpha=0.05, color='white', s=1,
                           label='Punting Linemen Routes',
                           title='Punting Linemen Routes Hybrid Coverage')
if legend is True:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax3.get_legend().remove()
plt.show()


# Double Covered
fig, ax2 = create_football_field(
    linenumbers=False, endzones=False, fifty_is_los=True)
legend = False
gunner_double_covered['x-rel-snap-plus50'] = gunner_double_covered['x-rel-snap'] + 60
gunner_double_covered['x-rel-snap-plus50'] = pd.to_numeric(
    gunner_double_covered['x-rel-snap-plus50'])
gunner_double_covered.plot(x='x-rel-snap-plus50', y='y', kind='scatter',
                           ax=ax2, alpha=0.05, color='white', s=1,
                           label='Punting Linemen Routes',
                           title='Punting Linemen Routes Double Coverage')
if legend is True:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
else:
    ax2.get_legend().remove()
# Pull NGS data for example of preprocessing
ngs = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
"""
Functions for calculating physics with play
"""

def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def add_play_physics(play):
    # Format columns
    play['time'] = pd.to_datetime(play['time'])
    # Distance
    play['dis_meters'] = play['dis'] / 1.0936  # Add distance in meters
    # Speed
    play['dis_meters'] / 0.01
    play['v_mps'] = play['dis_meters'] / 0.1
    # Angles to radians
    play['dir_radians'] = play['dir'].apply(math.radians)
    play['o_radians'] = play['o'].apply(math.radians)
    average_weight_nfl_pounds = 245.86
    average_weight_nfl_kg = average_weight_nfl_pounds * 0.45359237
    # http://webpages.uidaho.edu/~renaes/251/HON/Student%20PPTs/Avg%20NFL%20ht%20wt.pdf
    play['momentum'] = play['v_mps'] * average_weight_nfl_kg
    play['momentum_x'] = pol2cart(play['momentum'], play['dir_radians'])[0]
    play['momentum_y'] = pol2cart(play['momentum'], play['dir_radians'])[1]
    return play
# Example play that we will calculate the risk factor for manually
play = ngs.loc[(ngs['Season_Year'] == 2016) &
               (ngs['GameKey'] == 5) &
               (ngs['PlayID'] == 3129)].copy()
play.columns = [col.lower() for col in play.columns]
play = pd.merge(play, pprd, how='left')
play = add_play_physics(play)

playexpanded = pd.merge(play, play,
                        on=['season_year', 'gamekey', 'playid', 'time'],
                        suffixes=('', '_partner'))
playexpanded['opp_momentum'] = np.sqrt(np.square(
    playexpanded['momentum_x'] - playexpanded['momentum_x_partner']) +
    np.square(playexpanded['momentum_y'] - playexpanded['momentum_y_partner']))
playexpanded['dist'] = np.sqrt((playexpanded['x'] - playexpanded['x_partner']).apply(np.square) +
                               (playexpanded['y'] - playexpanded['y_partner']).apply(np.square))
playexpanded['risk_factor'] = playexpanded['opp_momentum'] / \
    playexpanded['dist']

fig, ax = create_football_field(figsize=(20, 8))
df = playexpanded.loc[(playexpanded['role'] == 'PLW') & (
    playexpanded['role_partner'] == 'PR')].copy()
# These values calculated below at aggreate of all plays
mean_of_all_risk_factors = 183.40329650430533
stddev_of_all_risk_factors = 274.16429253242063

df['risk_factor_normalized'] = df['risk_factor'] / \
    (mean_of_all_risk_factors + stddev_of_all_risk_factors)


df[['time', 'role', 'role_partner', 'v_mps', 'v_mps_partner', 'opp_momentum', 'x', 'y', 'x_partner', 'y_partner']] \
    .plot(kind='scatter', x='x', y='y', vmin=0, vmax=3,
          c=df['risk_factor_normalized'].tolist(),
          cmap='coolwarm',
          ax=ax)

df[['time', 'role', 'role_partner', 'v_mps', 'v_mps_partner', 'opp_momentum', 'x', 'y', 'x_partner', 'y_partner']] \
    .plot(kind='scatter', x='x_partner', y='y_partner', vmin=0, vmax=3,
          c=df['risk_factor_normalized'].tolist(),
          cmap='coolwarm',
          ax=ax)

playexpanded.loc[playexpanded['event'] == 'ball_snap'].plot(x='x', y='y',
                                                            kind='scatter',
                                                            ax=ax,
                                                            color='black',
                                                            marker='o',
                                                            zorder=3,
                                                            label='Players Position at Snap')

fig.get_axes()[1].remove()
fig.get_axes()[0].set_title('Injury Risk')
plt.title('Example 1: Player Pair with High Injury Risk', fontsize=20)
plt.show()
fig, ax = create_football_field(figsize=(20, 8))
df2 = playexpanded.loc[(playexpanded['gsisid'] == 31902) & (
    playexpanded['gsisid_partner'] == 28041)].copy()
# These values calculated below at aggreate of all plays
mean_of_all_risk_factors = 183.40329650430533
stddev_of_all_risk_factors = 274.16429253242063

df2['risk_factor_normalized'] = df2['risk_factor'] / \
    (mean_of_all_risk_factors + stddev_of_all_risk_factors)


df2[['time', 'role', 'role_partner', 'v_mps', 'v_mps_partner', 'opp_momentum', 'x', 'y', 'x_partner', 'y_partner']] \
    .plot(kind='scatter', x='x', y='y', vmin=0, vmax=3,
          c=df2['risk_factor_normalized'].tolist(),
          cmap='coolwarm',
          ax=ax)

df2[['time', 'role', 'role_partner', 'v_mps', 'v_mps_partner', 'opp_momentum', 'x', 'y', 'x_partner', 'y_partner']] \
    .plot(kind='scatter', x='x_partner', y='y_partner',
          c=df2['risk_factor_normalized'].tolist(), vmin=0, vmax=3,
          cmap='coolwarm',
          ax=ax)

playexpanded.loc[playexpanded['event'] == 'ball_snap'].plot(x='x', y='y',
                                                            kind='scatter',
                                                            ax=ax,
                                                            color='black',
                                                            style='.',
                                                            zorder=3,
                                                            label='Players Position at Snap')

fig.get_axes()[1].remove()
plt.title('Example 2: Player Pair with Low Injury Risk', fontsize=20)
plt.show()
# Plot both examples side by side
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)
      ) = plt.subplots(5, 2, sharey='row', sharex=True, figsize=(23, 12))

df.set_index('time')[['momentum']] \
    .plot(title='Example 1: Player 1 Momentum (kg m/s)', ax=ax1, color=color3)
df.set_index('time')[['momentum_partner']] \
    .plot(title='Example 1: Player 1 Momentum (kg m/s)', ax=ax3, color=color3)
df.set_index('time')[['opp_momentum']] \
    .plot(title='Example 1: Opposing Momentum', ax=ax5, color=color1)
df.set_index('time')[['dist']] \
    .plot(title='Example 1: Distance between Players', ax=ax7, color=color2)
df.set_index('time')[['risk_factor_normalized']] \
    .plot(title='Example 1: Normalized Injury Risk', ax=ax9, color=color4)
df2.set_index('time')[['momentum']] \
    .plot(title='Example 2: Player 2 Momentum (kg m/s)', ax=ax2, color=color3)
df2.set_index('time')[['momentum_partner']] \
    .plot(title='Example 2: Player 2 Momentum (kg m/s)', ax=ax4, color=color3)
df2.set_index('time')[['opp_momentum']] \
    .plot(title='Example 2: Opposing Momentum', ax=ax6, color=color1)
df2.set_index('time')[['dist']] \
    .plot(title='Example 2: Distance between Players', ax=ax8, color=color2)
df2.set_index('time')[['risk_factor_normalized']] \
    .plot(title='Example 2: Normalized Injury Risk', ax=ax10, color=color4)

# Remove Legends
ax1.get_legend().remove()
ax2.get_legend().remove()
ax3.get_legend().remove()
ax4.get_legend().remove()
ax5.get_legend().remove()
ax6.get_legend().remove()
ax7.get_legend().remove()
ax8.get_legend().remove()
ax9.get_legend().remove()
ax10.get_legend().remove()
plt.subplots_adjust(hspace=0.7)
plt.show()

sns.set(style="white")
plt.style.use('ggplot')

max_risk_partners = pd.read_parquet(
    '../input/robmullanflpreprocessed/max_risk_partners.parquet')
ax1 = max_risk_partners['dist'].plot(kind='hist',
                                     figsize=(15, 5),
                                     bins=100,
                                     color=color4,
                                     title="Distribution Of Players Distance to Eachother During Punt Plays")
ax1.set_xlabel('Distance between Players (yards)')
plt.show()

# Cap distance at 0.1, because it is physically impossible to be 0 distance from eachother and divide by zero issues
# Example of calculating risk factor
# Distance can't be zero
max_risk_partners.loc[max_risk_partners['dist'] < 0.1, 'dist'] = 0.1
max_risk_partners['dist'] = max_risk_partners['dist']
max_risk_partners['risk_factor'] = max_risk_partners['opp_momentum'] / \
    max_risk_partners['dist']


ax2 = max_risk_partners['opp_momentum'].plot(kind='hist',
                                             figsize=(15, 5),
                                             bins=500,
                                             color=color3,
                                             xlim=(0, 3000),
                                             title='Distribution of Opposing Forces between Players')
ax2.set_xlabel('Opposing Momentum between Players kg * (m/s)')
plt.show()
# Read the momentum and distance calculated for every play

injury_play_ngs = pd.read_parquet(
    '../input/nfl-punt-data-preprocessing-ngs-injury-plays/NGS-injury-plays.parquet')
gsisid_numbers = ppd.groupby('gsisid')['number'].apply(
    lambda x: "%s" % ', '.join(x))
gsisid_numbers = pd.DataFrame(gsisid_numbers).reset_index()
# Add Player Number and Direction
vr_with_number = pd.merge(
    vr, gsisid_numbers, how='left', suffixes=('', '_injured'))
vr_with_number['primary_partner_gsisid'] = vr_with_number['primary_partner_gsisid'].replace(
    'Unclear', np.nan).fillna(0).astype('int')
vr_with_number = pd.merge(vr_with_number, gsisid_numbers,
                          how='left',
                          left_on='primary_partner_gsisid',
                          right_on='gsisid', suffixes=('', '_primary_partner'))
vr = vr_with_number

# Mark the player pairs where one was injured
vr['injured_pair'] = True
risk_with_inj = pd.merge(max_risk_partners,
                         vr[['season_year', 'gamekey', 'playid', 'gsisid',
                             'injured_pair', 'primary_partner_gsisid']],
                         left_on=['season_year', 'gamekey',
                                  'playid', 'gsisid', 'gsisid_partner'],
                         right_on=['season_year', 'gamekey', 'playid',
                                   'gsisid', 'primary_partner_gsisid'],
                         how='left',
                         suffixes=('', '_y'))
risk_with_inj = pd.merge(risk_with_inj,
                         vr[['season_year', 'gamekey', 'playid', 'gsisid',
                             'injured_pair', 'primary_partner_gsisid']],
                         left_on=['season_year', 'gamekey',
                                  'playid', 'gsisid_partner', 'gsisid'],
                         right_on=['season_year', 'gamekey', 'playid',
                                   'gsisid', 'primary_partner_gsisid'],
                         how='left',
                         suffixes=('', '_y'))

risk_with_inj['either_injured'] = risk_with_inj[[
    'injured_pair', 'injured_pair_y']].sum(axis=1)
risk_with_inj['risk_factor_normalized'] = \
    risk_with_inj['risk_factor'] / (risk_with_inj['risk_factor'].mean() + risk_with_inj['risk_factor'].std())
ax = risk_with_inj['risk_factor_normalized'].plot(kind='hist',
                                                  bins=500,
                                                  xlim=(0, 10), figsize=(15, 5),
                                                  title='Distribution of All Player Risks',
                                                  label='Risk Factor (All Player Pairs)',
                                                  color=color2)

mean_risk_for_concussions = risk_with_inj.loc[risk_with_inj['injured_pair']
                                              == True]['risk_factor_normalized'].mean()
# ax.axvline(x=mean_risk_for_concussions, color=color1, label='Average Risk Concussions')
# add dots for concussion plays
inured_pairs = risk_with_inj.loc[risk_with_inj['injured_pair'] == True][[
    'risk_factor_normalized']]
inured_pairs['y_axis'] = 10000
inured_pairs.plot(x='risk_factor_normalized', y='y_axis',
                  ax=ax, kind='scatter',
                  color='red',
                  zorder=3,
                  label='Concussion Injury Pair')
ax.set_ylabel('Count')
ax.set_xlabel('Normalized Injury Risk Factor')

plt.legend()
plt.show()

risk_with_inj.loc[risk_with_inj['injured_pair'] == True][['season_year', 'gamekey', 'playid',
                                                          'generalized_role', 'generalized_role_partner',
                                                          'risk_factor_normalized']] \
    .sort_values('risk_factor_normalized', ascending=False) \
    .reset_index(drop=True)
sns.set(style="white")

risk_with_inj['generalized_role'] = risk_with_inj['generalized_role'].str.replace(
    'Punting_Longsnapper', 'Punting_Lineman')
risk_with_inj['generalized_role_partner'] = risk_with_inj['generalized_role_partner'].str.replace(
    'Punting_Longsnapper', 'Punting_Lineman')
# Format data to compare risk vs plays
risk_corr = risk_with_inj[['generalized_role',
                           'generalized_role_partner',
                           'risk_factor_normalized']] \
    .groupby(['generalized_role', 'generalized_role_partner']) \
    .median() \
    .reset_index() \
    .pivot(index='generalized_role',
           columns='generalized_role_partner',
           values='risk_factor_normalized') \
    .fillna(0)

# Remove underscores
risk_corr.index = risk_corr.index.str.replace('_', ' ')
risk_corr.columns = risk_corr.columns.str.replace('_', ' ')

# Generate a mask for the upper triangle
mask = np.zeros_like(risk_corr, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(risk_corr,
            mask=mask,
            cmap='coolwarm', center=0.25,
            square=True, linewidths=.5,
            vmax=0.5,
            cbar_kws={"shrink": .5})
plt.title('Median Risk of Generalized Role Pairs', fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.show()
plt.style.use('ggplot')

risk_sum_per_play = risk_with_inj \
    .groupby(['season_year', 'gamekey', 'playid'])['risk_factor_normalized'] \
    .mean() \
    .reset_index()

playrisk = pd.merge(risk_sum_per_play, pi, how='left')
playrisk = pd.merge(
    playrisk, vr[['season_year', 'gamekey', 'playid', 'injured_pair']], how='left')
ax = playrisk.plot(x='punt_distance',
                   y='risk_factor_normalized',
                   kind='scatter',
                   figsize=(15, 8), alpha=0.1)
playrisk.loc[playrisk['injured_pair'] == 1].plot(x='punt_distance',
                                                 y='risk_factor_normalized',
                                                 kind='scatter',
                                                 color=color1,
                                                 title='Injury Risk vs Punt Distance',
                                                 label='Play where a Concussion occurred',
                                                 ax=ax)

playrisk.groupby('punt_distance')['risk_factor_normalized'] \
    .mean() \
    .plot(ax=ax, color=color3, label='Average Play Risk at Punt Distance')
ax.set_xlabel('Punt Distance')
ax.set_ylabel('Normalized Risk Factor of Play')
plt.legend()
ax.set_xlim(20, 70)
plt.show()

playrisk['play_unique'] = playrisk['season_year'].astype('str').add((playrisk['gamekey']).astype('str')).add((playrisk['playid']).astype('str'))
# Ended up not liking these plots so hiding them
fig, (ax, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 6, figsize=(20, 5), sharey=True)
sns.boxplot(x='Fumbled', y='risk_factor_normalized', data=playrisk, ax=ax)
sns.boxplot(x='Returned for Negative Gain', y='risk_factor_normalized', data=playrisk, ax=ax1)
sns.boxplot(x='Returned', y='risk_factor_normalized', data=playrisk, ax=ax2)
sns.boxplot(x='Returned for No Gain', y='risk_factor_normalized', data=playrisk, ax=ax3)
sns.boxplot(x='Fair Catch', y='risk_factor_normalized', data=playrisk, ax=ax4)
sns.boxplot(x='Out of Bounds', y='risk_factor_normalized', data=playrisk, ax=ax5)
fig.suptitle('Risk Factor by Play Features')
plt.show()
def play_result(row):
    """
    Function to determine play outcome"""
    if row['Punt Blocked']:
        return 'Punt Blocked'
    elif row['Fair Catch']:
        return 'Fair Catch'
    elif row['Touchback']:
        return 'Touchback'
    elif row['Downed']:
        return 'Downed'
    elif row['Returned for Negative Gain']:
        return 'Negative Gain'
    elif row['Returned for Positive Gain']:
        return 'Positive Gain'
    elif row['Returned for No Gain']:
        return 'No Gain'
    elif row['Out of Bounds']:
        return 'Out of Bounds'
    elif row['PENALTY']:
        return 'Penalty'


playrisk['Play Result'] = playrisk.apply(play_result, axis=1)
playrisk_with_roles = pd.merge(
    playrisk, play_roles, how='left', on='play_unique')

playrisk_mean_by_jammers = playrisk_with_roles.groupby(['Number_of_jammers'])[['risk_factor_normalized']] \
    .agg([np.mean, 'count']) \
    .reset_index()
playrisk_mean_by_jammers.columns = [
    ' '.join(col).strip() for col in playrisk_mean_by_jammers.columns.values]

fig, ax = plt.subplots(1, figsize=(20, 7), sharey=True)

sns.swarmplot(x='Play Result',
              y='risk_factor_normalized',
              order=['No Gain', 'Positive Gain', 'Punt Blocked',
                     'Negative Gain', 'Downed', 'Touchback', 'Fair Catch',
                     'Out of Bounds', 'Penalty'],
              data=playrisk_with_roles.replace(
                  'Returned for No Gain', 'No Gain'),
              ax=ax,
              hue_order=[0, 1, 2, 3, 4, 5])
plt.xlabel('')
plt.ylabel('Play Risk Metric (Larger is at more risk for injury)')
plt.suptitle(
    'Risk of Plays by Outcome (Each dot represents a play)', fontsize=15)
plt.show()

playrisk_mean_by_type = playrisk.groupby(['Play Result'])[['risk_factor_normalized']] \
    .agg([np.mean, 'count']) \
    .reset_index()

playrisk_mean_by_type.columns = [
    ' '.join(col).strip() for col in playrisk_mean_by_type.columns.values]

playrisk_mean_by_type.loc[playrisk_mean_by_type['risk_factor_normalized count'] > 1]

playrisk_mean_by_type.sort_values('risk_factor_normalized mean') \
    .plot(x='Play Result',
          y='risk_factor_normalized mean',
          kind='barh',
          figsize=(15, 5),
          color='grey',
          legend=False,
          title='Normalized Risk Factor of Play by Result')
plt.ylabel('')
plt.show()

playrisk_mean_by_jammers.sort_index(ascending=True) \
    .plot(x='Number_of_jammers',
          y='risk_factor_normalized mean',
          kind='bar',
          figsize=(15, 5),
          color='grey',
          legend=False,
          rot=0,
          title='Normalized Risk Factor of Play by Result')
plt.ylabel('Normalized Play Injury Risk')
plt.xlabel('Number of Jammers')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold  # import KFold
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

X = playrisk_with_roles[['GL', 'GLi', 'GLo', 'GR', 'GRi', 'GRo', 'P', 'PC', 'PDL1', 'PDL2',
                         'PDL3', 'PDL4', 'PDL5', 'PDL6', 'PDM', 'PDR1', 'PDR2', 'PDR3',
                         'PDR4', 'PDR5', 'PDR6', 'PFB', 'PLG', 'PLL', 'PLL1', 'PLL2',
                         'PLL3', 'PLM', 'PLM1', 'PLR', 'PLR1', 'PLR2', 'PLR3', 'PLS', 'PLT',
                         'PLW', 'PPL', 'PPLi', 'PPLo', 'PPR', 'PPRi', 'PPRo', 'PR', 'PRG',
                         'PRT', 'PRW', 'VL', 'VLi', 'VLo', 'VR', 'VRi', 'VRo', 'punt_distance', 'return_yards']] \
    .rename(columns={'punt_distance': 'Punt Distance',
                     'return_yards': 'Return Yards'}).fillna(0)
y = playrisk_with_roles['risk_factor_normalized']

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)

clf.best_params_
lasso = Lasso(**clf.best_params_, fit_intercept=True)
lasso.fit(X, y)

model_coeffs = pd.DataFrame([X.columns, lasso.coef_]).T
model_coeffs.columns = ['Role in Play', 'Coefficient']

ax = model_coeffs.loc[model_coeffs['Coefficient'] != 0] \
    .set_index('Role in Play') \
    .sort_values('Coefficient') \
    .plot(kind='barh',
          figsize=(20, 6),
          color='grey',
          title='Coefficients of Model for Play Injury Risk (Zero coefficients not shown)')
ax.get_legend().remove()
plt.show()
