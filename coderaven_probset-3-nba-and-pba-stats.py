import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pba_stats = pd.read_excel('/kaggle/input/nba-and-pba-basketball-stats-sample-2019/pba_stats.xlsx')
nba_stats = pd.read_excel('/kaggle/input/nba-and-pba-basketball-stats-sample-2019/nba_stats.xlsx')
pba_stats
nba_stats
pba_field_goals = pba_stats[["PLAYERS", "FGm "]] # Field goals made
pba_field_goals
pba_min_fg = pba_field_goals['FGm '].min()
pba_max_fg = pba_field_goals['FGm '].max()

print(f'PBA Minimum Field Goals: {pba_min_fg}') 
print(f'PBA Maximum Field Goals: {pba_max_fg}') 
nba_field_goals = nba_stats[["PLAYERS", "FGm "]]
nba_field_goals
nba_min_fg = nba_field_goals['FGm '].min()
nba_max_fg = nba_field_goals['FGm '].max()

print(f'NBA Minimum Field Goals: {nba_min_fg}') 
print(f'NBA Maximum Field Goals: {nba_max_fg}') 
print(f'PBA Minimum Field Goals Made: {pba_min_fg}') 
print(f'NBA Minimum Field Goals Made: {nba_min_fg}') 
print('\n')
print(f'PBA Maximum Field Goals Made: {pba_max_fg}') 
print(f'NBA Maximum Field Goals Made: {nba_max_fg}') 
no_of_classes = 5

pba_fg_range = round(pba_max_fg - pba_min_fg, 2)
nba_fg_range = round(nba_max_fg - nba_min_fg, 2)

print(f'PBA FGA Range: {pba_fg_range}') 
print(f'NBA FGA Range: {nba_fg_range}') 
pba_class_width = round(pba_fg_range / no_of_classes) + 0.5
nba_class_width = round(nba_fg_range / no_of_classes)


print(f'PBA Class Width: {pba_class_width}') 
print(f'NBA Class Width: {nba_class_width}') 
pba_bottom_boundary = round(pba_min_fg - (0.5 * pba_class_width), 2)
nba_bottom_boundary = round(nba_min_fg - (0.5 * nba_class_width), 2)

print(f'PBA Bottom Boundary: {pba_bottom_boundary}') 
print(f'NBA Bottom Boundary: {nba_bottom_boundary}') 

nba_bottom_boundary = nba_min_fg
print(f'NBA Bottom Boundary: {nba_bottom_boundary}') 
pba_fg_groups = []
pba_fg_range_groups = []
for i in range(no_of_classes):
    calculatedGroup = pba_bottom_boundary + (pba_class_width * i)
    pba_fg_groups.append(calculatedGroup)
    
    groupRange = f"={calculatedGroup} - <{calculatedGroup + pba_class_width}"
    pba_fg_range_groups.append(groupRange)

print("PBA Field Goal Groups: ")
print(pba_fg_range_groups)

print("\n")

nba_fg_groups = []
nba_fg_range_groups = []

for i in range(no_of_classes):
    calculatedGroup = nba_bottom_boundary + (nba_class_width * i)
    nba_fg_groups.append(calculatedGroup)
    
    groupRange = f"={calculatedGroup} - <{calculatedGroup + nba_class_width}"
    nba_fg_range_groups.append(groupRange)
    
print("NBA Field Goal Groups: ")
print(nba_fg_range_groups)

pba_tally = {}
for i in range(no_of_classes):
    pba_tally[pba_fg_range_groups[i]] = []
    
for key, data in pba_field_goals.iterrows(): 
    # data[1] is where field goal value is stored
    for i in range(no_of_classes):
        if (data[1] >= pba_fg_groups[i] and data[1] < (pba_fg_groups[i] + pba_class_width)):
            pba_tally[pba_fg_range_groups[i]].append(1) # Add to tally if fits to class
        else:
            pba_tally[pba_fg_range_groups[i]].append(0) 
    
    
nba_tally = {}
for i in range(no_of_classes):
    nba_tally[nba_fg_range_groups[i]] = []
    
for key, data in nba_field_goals.iterrows(): 
    # data[1] is where field goal value is stored
    for i in range(no_of_classes):
        if (data[1] >= nba_fg_groups[i] and data[1] < (nba_fg_groups[i] + nba_class_width)):
            nba_tally[nba_fg_range_groups[i]].append(1) # Add to tally if fits to class
        else:
            nba_tally[nba_fg_range_groups[i]].append(0)

pba_tally_frame = pd.DataFrame.from_dict(pba_tally, orient='columns')
nba_tally_frame = pd.DataFrame.from_dict(nba_tally, orient='columns')
pba_df = pd.melt(pba_tally_frame, var_name='Field Goal Class', value_name='Count')
sns.countplot(data=pba_df.loc[pba_df['Count']==1], x='Field Goal Class', hue='Count').set_title("PBA Field Goals Histogram")
nba_df = pd.melt(nba_tally_frame, var_name='Field Goal Class', value_name='Count')
sns.countplot(data=nba_df.loc[nba_df['Count']==1], x='Field Goal Class', hue='Count').set_title("NBA Field Goals Histogram")