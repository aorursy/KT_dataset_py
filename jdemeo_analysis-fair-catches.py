import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sns.set()
# Load in NGS data, player role data, and play info
ngs_df = pd.read_csv('../input/ngsconcussion/NGS-fair_catch.csv')
play_player_role_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
play_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

# Merge datasets
ngs_df = pd.merge(ngs_df, play_player_role_df,
                  how="inner",
                  on=['GameKey', 'PlayID', 'GSISID'])

ngs_df = pd.merge(ngs_df, play_df,
                  how="inner",
                  on=['GameKey', 'PlayID'])

# Cleanup
keepers = ['GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'dis', 'Event', 'Role', 'PlayDescription']
ngs_df = ngs_df[keepers]
ngs_df.head()
# NGS Unique_ids
ngs_ids = ngs_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})
ngs_ids.shape
'''ONLY RUN THE FOLLOWING TWO BLOCKS TO GET AN IDEA OF THE COURSE OF EVENTS FOR A PARTICULAR PLAY'''

# def isolate_play(df, game_key, play_id):
#     '''Create a dataframe of a particular play'''
#     where_condition = ((df['GameKey'] == game_key) &
#                        (df['PlayID'] == play_id))
#     new_df = df[where_condition].copy()
#     new_df.sort_values(by=['Time'], inplace=True)
#     new_df.reset_index(drop=True, inplace=True)
#     return new_df

# def course_of_events(df):
#     '''Get list of events in order of occurrence for a particular play'''
#     events = []
#     for i in range(len(df)):
#         event = df.loc[i, 'Event']
#         if event not in events:
#             events.append(event)
           
#     print('Play Description:', df.loc[0, 'PlayDescription'])
#     print('---')
#     print('Game Events:', events)
#     print('-----------------------------------------------')
# # Iterate through ids to get events for each play
# for i in range(len(ngs_ids)):
#     game_key = ngs_ids.loc[i, 'GameKey']
#     play_id = ngs_ids.loc[i, 'PlayID']
#     the_play = isolate_play(ngs_df, game_key, play_id)
#     course_of_events(the_play)
def event_df_creation(df, event):
    '''Get a new dataframe with data pertinent to a particular event'''
    new_df = df[df['Event'] == event].reset_index(drop=True)
    unique_ids = new_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})
    return new_df, unique_ids
# Let's indicate what team the player is playing on based off player role
return_team_positions = ['PR', 'PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDR1', 'PDR2', 'PDR3', 'PDR4', 'VL', 'VR', 
                         'PLL', 'PLR', 'VRo', 'VRi', 'VLi', 'VLo', 'PLM', 'PLR1', 'PLR2', 'PLL1', 'PLL2',
                         'PFB', 'PDL5', 'PDR5', 'PDL6', 'PLR3', 'PLL3', 'PDR6', 'PLM1', 'PDM']
punt_team_positions = ['P', 'PLS', 'PPR', 'PLG', 'PRG', 'PLT', 'PRT', 'PLW', 'PRW', 'GL', 'GR',
                       'GRo', 'GRi', 'GLi', 'GLo', 'PC', 'PPRo', 'PPRi', 'PPL', 'PPLi', 'PPLo']

def label_team(df):
    '''Label each player by the team they play on'''
    df['team'] = ''
    print('Determining player roles')

    for i, role in enumerate(df['Role']):
        if role in return_team_positions:
            df.loc[i, 'team'] = 'return team'
        elif role in punt_team_positions:
            df.loc[i, 'team'] = 'punt team'
        else:
            df.loc[i, 'team'] = 'unknown'
def calculate_player_proximity(role_x, role_y, player_x, player_y):
    '''Calculate euclidean distance between two players'''
    leg_x = (role_x - player_x) ** 2
    leg_y = (role_y - player_y) ** 2
    hypotenuse = np.sqrt(leg_x + leg_y)
    return hypotenuse
def calculate_x_proximity(role_x, player_x):
    '''Calculate distance of a player to a particular role only by yardline'''
    return np.abs(role_x - player_x)
def calculate_proximity_for_play(df, unique_ids, role):
    '''Calculate proximity of each player to the player of a particular role'''
    # Create feature for player proximity
    df['proximity_to_' + role + '_circle'] = 0
    df['proximity_to_' + role + '_x'] = 0
    
    print('Calculating player proximities to', role)
    
    # Go through each data point in particular NGS dataset
    for i in range(len(df)):
        
        # Play Information
        game_key = df.loc[i, 'GameKey']
        play_id = df.loc[i, 'PlayID']
        
        # Get one unique set of data points related to a single (GameKey, PlayID) pair
        where_condition = ((df['GameKey'] == game_key) &\
                           (df['PlayID'] == play_id))
        just_view = df[where_condition].reset_index()
        
        # Get coordinates of a player with a particular role
        if any(just_view['Role'] == role):
            role_x = just_view.loc[just_view['Role'] == role, 'x'].values[0]
            role_y = just_view.loc[just_view['Role'] == role, 'y'].values[0]
            
        # Plays that don't actually have the particular role represented
        else:
            continue

        # Current Player coordinates
        position_x = df.loc[i, 'x']
        position_y = df.loc[i, 'y']

        # Calculate proximity
        proximity_hypo = calculate_player_proximity(role_x, role_y, position_x, position_y)
        proximity_x = calculate_x_proximity(role_x, position_x)
        df.loc[i, 'proximity_to_' + role + '_circle'] = proximity_hypo
        df.loc[i, 'proximity_to_' + role + '_x'] = proximity_x
def calculate_closest_player(df, unique_ids, column):
    '''Find who the closest player on the punt team is and create new id set'''
    unique_ids[column] = 0
    good_indexes = []
    role = 'PR'
    print('Determining closest player to', role)
    
    for i in range(len(unique_ids)):
        
        # Play information
        game_key = unique_ids.loc[i, 'GameKey']
        play_id = unique_ids.loc[i, 'PlayID']

        # Get one unique set of data points related to a single (GameKey, PlayID) pair
        where_condition = ((df['GameKey'] == game_key) &\
                           (df['PlayID'] == play_id) &\
                           (df['team'] == 'punt team'))
        just_view = df[where_condition].reset_index(drop=True)
        
        # Take minimum of series and Error handling where the NGS data had no punt team :(
        try:
            unique_ids.loc[i, column] = min(just_view[column])
            good_indexes.append(i)
        except ValueError:
            continue
    
    # Create new set of ids
    new_ids = unique_ids.loc[good_indexes, :].copy()
    new_ids.reset_index(inplace=True, drop=True)
    
    return new_ids
event_df, event_ids = event_df_creation(ngs_df, 'fair_catch')
label_team(event_df)
calculate_proximity_for_play(event_df, event_ids, 'PR')
new_ids = calculate_closest_player(event_df, event_ids, 'proximity_to_PR_circle')
new_ids = calculate_closest_player(event_df, new_ids, 'proximity_to_PR_x')
new_ids.to_csv('FC-proximity.csv', index=False)
'''Plot of distribution distance of closest punt team player to punt receiver'''
bins = [i for i in range(0, 25, 1)]
plt.hist(new_ids['proximity_to_PR_x'], bins=bins)

plt.title('Distribution of closest player on punt team to punt receiver')
plt.xlabel('Yards')
plt.ylabel('count')
plt.show()

new_ids['proximity_to_PR_x'].describe()
'''Plot of distribution distance of closest punt team player to punt receiver'''
bins = [i for i in range(0, 25, 1)]
plt.hist(new_ids['proximity_to_PR_circle'], bins=bins)

plt.title('Distribution of closest player on punt team to punt receiver')
plt.xlabel('Yards')
plt.ylabel('count')
plt.show()

new_ids['proximity_to_PR_circle'].describe()
print('Count of fair catches with punt team player greater than 8 yards away by yardline:',
      new_ids[new_ids['proximity_to_PR_x'] > 8].shape[0])
print('Count of fair catches with punt team player greater than 8 yards away by Euclidean distance:',
      new_ids[new_ids['proximity_to_PR_circle'] > 10].shape[0])
pr_proximity = pd.read_csv('../input/ngsconcussion/PR-proximity.csv')
pr_proximity.head()
fc_proximity = pd.read_csv('../input/ngsconcussion/FC-proximity.csv')
fc_proximity.head()
plt.figure(figsize=(10, 6))
sns.distplot(pr_proximity['proximity_to_PR_x'], kde=False)
sns.distplot(fc_proximity['proximity_to_PR_x'], kde=False)
plt.axvline(8, color='black', linestyle='dashed', linewidth=1)

plt.xlim(0, 40)
plt.legend(['8 Yards', 'Fair Catches', 'Punt Returns'], fontsize=12)
plt.title('Distribution of closest punt team player to punt receiver', fontsize=16)
plt.xlabel('Yards', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('fc_pr_distributions.png', bbox_inches='tight')
plt.show()
