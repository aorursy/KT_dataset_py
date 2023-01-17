import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
play_df = pd.read_csv('../input/play_information.csv')
print(play_df.shape)
play_df.head(1)
# HOW MANY GAMES HAD NO PUNTS AND WHICH GAMES
stuff = []
# Collect all game id's in punt data
for element in play_df['GameKey']:
    stuff.append(element)
print('Number of games without a punt:', 666 - len(set(stuff)))

for element in [i for i in range(1, 667)]:
    if element not in set(stuff):
        print('Game', element, 'had no punts')
# Create condensed version of play data
keeper_columns = ['GameKey', 'PlayID', 'PlayDescription', 'Poss_Team', 'YardLine']
condensed_play_df = play_df[keeper_columns].copy()
def find_that_play_word(keyword, df):
    """Help to find keywords"""
    df[keyword] = 0
    count = 0
    for i, description in enumerate(df['PlayDescription']):
        game_key = df.loc[i, 'GameKey']
        play_id = df.loc[i, 'PlayID']
        # Find keyword in lowercased string of play description
        if description.lower().find(keyword) != -1:
#             print('Keyword', keyword, 'found for (game, play):', '(' + str(game_key) + ',' + str(play_id) + ')')
#             print('Play description:', description)
#             print('---')
                
            # One-hot encode with keyword
            df.loc[i, keyword] = 1
            count += 1

    print('# of', keyword, 'occuring on a punt play:', count)
find_that_play_word('fair catch', condensed_play_df)
find_that_play_word('touchback', condensed_play_df)
find_that_play_word('downed', condensed_play_df)
find_that_play_word(', out of bounds', condensed_play_df)
find_that_play_word('dead ball', condensed_play_df)
find_that_play_word('no play', condensed_play_df)
find_that_play_word('(punt formation) penalty on', condensed_play_df) # Picks up additional 'no play' type punts
# Reduce play_df even further 
where_condition = (
    (condensed_play_df['fair catch'] == 1) |
    (condensed_play_df['touchback'] == 1) |
    (condensed_play_df['downed'] == 1) |
    (condensed_play_df[', out of bounds'] == 1) |
    (condensed_play_df['dead ball'] == 1) |
    (condensed_play_df['no play'] == 1) |
    (condensed_play_df['(punt formation) penalty on'] == 1))
interesting_plays_df = condensed_play_df[~where_condition].reset_index(drop=True)

print('There are now', len(interesting_plays_df), '"interesting plays" from', len(condensed_play_df), 'punt plays')
print('Proportion of interesting punts:', len(interesting_plays_df)/len(condensed_play_df))
interesting_plays_df.head(1)
'''I only have this here for reference of what I've filtered by'''
uninteresting_keywords = ['fair catch', 'touchback.', 'downed', ', out of bounds', 'dead ball', 'no play',
                         '(punt formation) penalty on']
interesting_keywords = ['muffs', 'blocked by','touchdown.', 'fumble', 'ruling', 'fake punt',
                        'up the middle', 'pass', 'right end', 'left end', 'right guard',
                        'direct snap', 'touchdown nullified']
# 'Interesting outcomes'
find_that_play_word('muffs', interesting_plays_df)
find_that_play_word('blocked by', interesting_plays_df)
find_that_play_word('touchdown.', interesting_plays_df)
find_that_play_word('fumble', interesting_plays_df)
find_that_play_word('ruling', interesting_plays_df)
find_that_play_word('fake punt', interesting_plays_df)
find_that_play_word('safety', interesting_plays_df)
find_that_play_word('up the middle', interesting_plays_df)
find_that_play_word('pass', interesting_plays_df)
find_that_play_word('right end', interesting_plays_df)
find_that_play_word('left end', interesting_plays_df)
find_that_play_word('right guard', interesting_plays_df)
find_that_play_word('direct snap', interesting_plays_df)
find_that_play_word('touchdown nullified', interesting_plays_df)
# Create a dataset where plays are currently assumed to be actual punt returns 
where_condition = (
    (interesting_plays_df['muffs'] == 1) |
    (interesting_plays_df['blocked by'] == 1) |
    (interesting_plays_df['touchdown.'] == 1) |
    (interesting_plays_df['fumble'] == 1) |
    (interesting_plays_df['ruling'] == 1) |
    (interesting_plays_df['fake punt'] == 1) |
    (interesting_plays_df['safety'] == 1) |
    (interesting_plays_df['up the middle'] == 1) |
    (interesting_plays_df['pass'] == 1) |
    (interesting_plays_df['right end'] == 1) |
    (interesting_plays_df['left end'] == 1) |
    (interesting_plays_df['right guard'] == 1) |
    (interesting_plays_df['direct snap'] == 1) |
    (interesting_plays_df['touchdown nullified'] == 1))
remainder_df = interesting_plays_df[~where_condition].reset_index(drop=True)

# Isolate touchdowns that were from punt returns
where_condition = ((interesting_plays_df['touchdown.'] == 1) &
                   (interesting_plays_df['blocked by'] == 0) &
                   (interesting_plays_df['direct snap'] == 0) &
                   (interesting_plays_df['right guard'] == 0) &
                   (interesting_plays_df['fumble'] == 0) &
                   (interesting_plays_df['pass'] == 0))
td_df = interesting_plays_df[where_condition].reset_index(drop=True)

# Combine touchdown punt returns and regular punt returns
remainder_df = pd.concat([remainder_df, td_df], axis=0)

# Drop unnecessary columns
keeper_columns = ['GameKey', 'PlayID', 'PlayDescription', 'Poss_Team', 'YardLine']
remainder_df = remainder_df[keeper_columns]
remainder_df.reset_index(inplace=True, drop=True)
print(remainder_df.shape)
remainder_df.head()
# Create dataset of punt return plays
remainder_df.to_csv('play-punt_return.csv', index=False)

# Create dataset of fair catch plays
where_condition = ((condensed_play_df['fair catch'] == 1))
fc_df = condensed_play_df[where_condition].reset_index(drop=True)\

# Drop unnecessary columns
keeper_columns = ['GameKey', 'PlayID', 'PlayDescription', 'Poss_Team', 'YardLine']
fc_df = fc_df[keeper_columns]
fc_df.to_csv('play-fair_catch.csv', index=False)
# Just for reference if you want to filter return plays that have a penalty
find_that_play_word('penalty on', remainder_df)
'''
Need to parse through PlayDescription in order to get return distance of play and distance to touchdown
Patterns that return two distances for yardage on play are lateral plays
'''

# Regex for them patterns
punt_distance_pattern = re.compile(r'punts ((-?)\d+) yards? to(\s| \w+ )((-?)\d+)')
yards_gained_pattern = re.compile(r'for ((-?)\d+) yard')
no_yards_gained_pattern = re.compile(r'([A-Z]\w+) ((-?)\d+) (for no gain)')

remainder_df['punt distance'] = 0
remainder_df['side ball lands'] = ''
remainder_df['yardline received'] = 0
remainder_df['yardage on play'] = 0

for i, element in enumerate(remainder_df['PlayDescription']):

    punt_distance = punt_distance_pattern.findall(element) # ('Punt distance', '', 'Side Ball Lands', 'Yardline Received')
    yards_gained = yards_gained_pattern.findall(element)   # ('Yardage on Play', '', )
    no_gain = no_yards_gained_pattern.findall(element)
    
#     print(punt_distance)
#     print(yards_gained)
#     print(no_gain)
    
    # A play that results in yards gained or lossed
    if yards_gained != []:
        remainder_df.loc[i, 'punt distance'] = int(punt_distance[0][0])
        remainder_df.loc[i, 'side ball lands'] = punt_distance[0][2]
        remainder_df.loc[i, 'yardline received'] = int(punt_distance[0][3])
        
        # A normal return
        if len(yards_gained) == 1:
            remainder_df.loc[i, 'yardage on play'] = int(yards_gained[0][0])
            
        # For laterals
        else:
            remainder_df.loc[i, 'yardage on play'] = int(yards_gained[0][0]) + int(yards_gained[1][0])
            
    # A play that resulted in no gain in yards
    elif no_gain != []:
        remainder_df.loc[i, 'punt distance'] = int(punt_distance[0][0])
        remainder_df.loc[i, 'side ball lands'] = punt_distance[0][2]
        remainder_df.loc[i, 'yardline received'] = int(punt_distance[0][3])

#     print('---')
# Doing some hand processing of specific returns where the yardage gained on return was
# officially changed (I know not elegant, especially if dataframe indices change overtime)
culprits = [476, 891, 1062, 1064, 1096, 2193]
yard_changes = [14, 6, 0, 3, 0, 4]
for i, element in enumerate(culprits):
    remainder_df.loc[element, 'yardage on play'] = yard_changes[i]
def calculate_distance_to_td (data_sample):
    '''Calculate distance needed for touchdown for each play'''
    # Punts that land on the 50 yard line
    if data_sample['yardline received'] == 50:
        distance_to_touchdown = 50
    
    # Punting on punting team's side of field
    elif data_sample['Poss_Team'] == data_sample['YardLine'][:len(data_sample['Poss_Team'])]:
        # Ball remains on punt team's side of field
        if data_sample['side ball lands'] == data_sample['YardLine'][:len(data_sample['Poss_Team'])]:
            distance_to_touchdown = data_sample['yardline received']
        # Ball is punted to return team's side of field
        else:
            distance_to_touchdown = (50 - data_sample['yardline received']) + 50
            
    # Punting on opponents side of field
    else:
        distance_to_touchdown = (50 - data_sample['yardline received']) + 50
    return distance_to_touchdown
# Calculate the value of a punt return based solely on the proportion of yardage gained on the return
# Relative to how many yards are needed to score a touchdown from where the punt initially lands
remainder_df['reward'] = 0
for i in range(len(remainder_df)):
    yards_on_return = remainder_df.loc[i, 'yardage on play']
    distance_to_touchdown = calculate_distance_to_td(remainder_df.iloc[i, :])
    remainder_df.loc[i, 'reward'] = yards_on_return / distance_to_touchdown
#     print('Value of return:', yards_on_return / distance_to_touchdown)

remainder_df.head()
# Cleanup
keepers = ['GameKey', 'PlayID', 'PlayDescription', 'yardage on play', 'reward']
remainder_df = remainder_df[keepers]

# Create dataset for external usage
remainder_df.to_csv('play-punt_return-yardage.csv', index=False)
play_player_role_df = pd.read_csv('../input/play_player_role_data.csv')
print(play_player_role_df.shape)
play_player_role_df.tail(2)
# How many unique plays in play_player_role dataset?
print('# of unique plays according to play_player_role dataset:',
      len(play_player_role_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})))

# How many roles are there?
print('# of roles in dataset:', len(play_player_role_df['Role'].value_counts()))
