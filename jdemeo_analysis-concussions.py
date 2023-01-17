import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import Image, display
sns.set()
# Load in concussion data deduced from video review of the play
video_review_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
print(video_review_df.shape)
video_review_df.head(1)
# Load in player related data
player_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
play_player_role_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')

# Combine Relevant Player Information, Position, Role, Number
master_player_df = pd.merge(player_df, play_player_role_df,
                          how='inner',
                          on=['GSISID']).drop(columns=['Season_Year'])
master_player_df.head()
# Check summary counts to see if any are needed
print(video_review_df['Player_Activity_Derived'].value_counts())
print('---')
print(video_review_df['Turnover_Related'].value_counts())
print('---')
print(video_review_df['Primary_Impact_Type'].value_counts())
print('---')
print(video_review_df['Primary_Partner_Activity_Derived'].value_counts())
print('---')
print(video_review_df['Friendly_Fire'].value_counts())

where_condition = ((video_review_df['Primary_Partner_GSISID'] == 'Unclear') |
                   (video_review_df['Primary_Partner_GSISID'].isna()))
video_review_df[where_condition]
# Now that we have a descriptive idea of what's going on, I'm gonna just drop these columns
# and also clear up the unclear designation and convert it to NaN
droppers = ['Player_Activity_Derived', 'Turnover_Related', 'Primary_Partner_Activity_Derived', 
            'Friendly_Fire', 'Season_Year']
video_review_df.drop(columns=droppers, inplace=True)

# Remove 'Unclear' designation
video_review_df.loc[33, 'Primary_Partner_GSISID'] = 'NaN'
'''Concussion Video Footage'''
video_footage_injury_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
print(video_footage_injury_df.shape)
video_footage_injury_df.head(1)
# Preprocess to allow for easier joins
rename_columns = {'gamekey': 'GameKey', 'playid': 'PlayID', 'season': 'Season_Year'}
video_footage_injury_df.rename(columns=rename_columns, inplace=True)

# Combine Video Review and Video Injury DataFrames to have the injured player and partner player data
injury_play = pd.merge(video_review_df, video_footage_injury_df, 
                       how='inner', 
                       on=['GameKey', 'PlayID'])

# Lets Drop Some More Data I consider uncritical for getting a feel for the data
droppers = ['Season_Year', 'Type', 'Week', 'Home_team', 'Visit_Team', 'Qtr']
injury_play.drop(columns=droppers, inplace=True)

injury_play.head(1)
# Join Info (their jersey number, position, role) on the concussed player
injury_play = pd.merge(injury_play, master_player_df,
                           how='inner',
                           on=['GSISID', 'GameKey', 'PlayID'])

print('Shape:', injury_play.shape)
injury_play.head(1)
# Drop Certain Rows After Identifying Concussed Players Jersey Number from video
drop_rows = [1, 7, 17, 20, 23, 26, 27, 28, 29, 32, 33, 36, 38, 39, 45, 46, 47, 50, 52, 55, 57]
injury_play.drop(labels=drop_rows, inplace=True)
injury_play.reset_index(drop=True, inplace=True)

# Convert Primary_Partner_GSISID from str to float
injury_play['Primary_Partner_GSISID'] = injury_play['Primary_Partner_GSISID'].astype('float')
injury_play.shape
# 0 = No, 1 = Yes, 2 = inconclusive
# When concussions are occuring
after_ball_punted = [1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1]
after_punt_received = [1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1]
fake = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# My un-expert labeling for uncalled penalties and where the new use-of-helmet rule would apply
uncalled_penalty = [2,0,2,0,1,0,0,1,0,1,1,0,0,0,0,0,1,2,0,0,1,1,2,1,0,0,0,1,0,0,0,0,0,0,0,0,0]
helmet_rule = [1,1,1,1,0,0,0,0,1,0,1,1,1,0,0,1,2,0,1,1,1,0,1,0,0,0,0,0,0,1,1,1,1,1,0,1,0]
# Add one hot encodings
injury_play['after_ball_punted'] = after_ball_punted
injury_play['after_punt_received'] = after_punt_received
injury_play['fake'] = fake
injury_play['uncalled_penalty'] = uncalled_penalty
injury_play['helmet_rule'] = helmet_rule
injury_play.to_csv('injury_play.csv', index=False)
columns_of_interest = ['after_punt_received', 'uncalled_penalty', 'helmet_rule']
for column in columns_of_interest:
    print(injury_play[injury_play['after_ball_punted'] == 1][column].value_counts())
    print('---')
# Load in NGS data, player role data, and play info
ngs_concussion = pd.read_csv('../input/ngsconcussion/NGS-concussion.csv')
play_player_role_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
play_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

# Merge datasets
ngs_concussion = pd.merge(ngs_concussion, play_player_role_df,
                  how="inner",
                  on=['GameKey', 'PlayID', 'GSISID'])

ngs_concussion = pd.merge(ngs_concussion, play_df,
                  how="inner",
                  on=['GameKey', 'PlayID'])

# Cleanup
keepers = ['GameKey', 'PlayID', 'GSISID', 'Time', 'x', 'y', 'dis', 'Event', 'Role', 'PlayDescription']
ngs_concussion = ngs_concussion[keepers]
ngs_ids = ngs_concussion.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})
print('Number of unique plays in NGS dataset:', ngs_ids.shape[0])
ngs_concussion.head()
# Filter for after_punt concussions
where_condition = (injury_play['after_ball_punted'] == 1)
after_punt_df = injury_play[where_condition]
after_punt_df.reset_index(inplace=True, drop=True)
after_punt_ids = after_punt_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})

# Get appropriate NGS data
ngs_after_punt = pd.merge(ngs_concussion, after_punt_ids,
                          how='inner',
                          on=['GameKey', 'PlayID'])
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
#     print('')
#     print('Game Events:', events)
#     print('-----------------------------------------------')
# # Iterate through ids to get events for each play
# def whats_going_on(df, ids):
#     for i in range(len(ids)):
#         game_key = ids.loc[i, 'GameKey']
#         play_id = ids.loc[i, 'PlayID']
#         print('(', game_key, ',', play_id, ')')
#         the_play = isolate_play(df, game_key, play_id)
#         course_of_events(the_play)
        
# whats_going_on(ngs_after_punt, after_punt_ids)
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

    for i, role in enumerate(df['Role']):
#         print(i, role)
        if role in return_team_positions:
            df.loc[i, 'team'] = 'return team'
        elif role in punt_team_positions:
            df.loc[i, 'team'] = 'punt team'
        else:
            df.loc[i, 'team'] = 'unknown'
def calculate_player_proximity(role_x, role_y, player_x, player_y):
    '''Calculate distance of a player to a particular role'''
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
    
    # Go through each data point in particular NGS dataset
    for i in range(len(df)):
#         print(i)
        
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
        
    # Calculate closest player (adds to ids dataframe)
    calculate_closest_player(df, unique_ids, 'proximity_to_' + role + '_circle')
    calculate_closest_player(df, unique_ids, 'proximity_to_' + role + '_x')
def calculate_closest_player(df, unique_ids, column):
    '''Find who the closest player on the punt team and create new id set'''
    unique_ids[column] = 0
    good_indexes = []
    
    for i in range(len(unique_ids)):
#         print(i)
        
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
def all_event_proximities(ngs_df, events):
    data_frames = []
    
    for event in events:
        event_df, event_ids = event_df_creation(ngs_df, event)
        label_team(event_df)
        calculate_proximity_for_play(event_df, event_ids, 'PR')
        data_frames.append(event_ids)
        
    # Combine ids of particular play events
    new_ids = pd.concat(data_frames,
                        axis=0)
    new_ids.reset_index(inplace=True, drop=True)
    
    return new_ids
# Events of interest
events = ['punt_received', 'punt_downed', 'kick_received', 'punt_land']

new_ids = all_event_proximities(ngs_after_punt, events)
new_ids.head()
'''Plot of distribution distance of closest player to punt receiver'''
def proximity_distribution(df, column):
    bins = [i for i in range(0, 30, 1)]
    plt.hist(df[column], bins=bins)

    plt.title(column)
    plt.xlabel('Yards')
    plt.ylabel('count')
    plt.show()

    print(new_ids[column].describe())
proximity_distribution(new_ids, 'proximity_to_PR_circle')
proximity_distribution(new_ids, 'proximity_to_PR_x')
# Number of concussion plays affected by a rule with 8 yard restricted zone for PR by yardline distance
print(new_ids[new_ids['proximity_to_PR_x'] <= 8].shape[0])

# Number of concussion plays affected by a rule with 10 yard restricted zone for PR by euclidean distance to PR
print(new_ids[new_ids['proximity_to_PR_circle'] <= 10].shape[0])
# Combine with injury_play data to get one-hot encoding labels
combo = pd.merge(new_ids, after_punt_df,
                 how="inner",
                 on=['GameKey', 'PlayID'])

# Get Number of concussions that a particular rule may pertain too
print('Concussion Coverage by Particular Rules')
print('---------------------------------------')
print('Count of use-of-helmet rule:', combo[combo['helmet_rule'] == 1].shape[0])
print('Count of uncalled penalty rules:', combo[combo['uncalled_penalty'] == 1].shape[0])
print('Count of PR restricted zone rule:', combo[combo['proximity_to_PR_x'] <= 8].shape[0])

# Get overlap of rule counts
where_condition_1 = ((combo['helmet_rule'] == 1) &
                    (combo['uncalled_penalty'] == 1))
where_condition_2 = ((combo['helmet_rule'] == 1) &
                    (combo['proximity_to_PR_x'] <= 8))
where_condition_3 = ((combo['uncalled_penalty'] == 1) &
                    (combo['proximity_to_PR_x'] <= 8))
where_condition_4 = ((combo['proximity_to_PR_x'] <= 8) &
                    (combo['helmet_rule'] == 1) & 
                    (combo['uncalled_penalty'] == 1))

print('Count for use-of-helmet and uncalled penalty rules:', combo[where_condition_1].shape[0])
print('Count for use-of-helmet and PR restricted zone rules:', combo[where_condition_2].shape[0])
print('Count for uncalled penalty and PR restricted zone rules:', combo[where_condition_3].shape[0])
print('Count for use-of-helmet, uncalled penalty, and PR restricted zone rules:', combo[where_condition_4].shape[0])
from matplotlib_venn import venn3, venn3_circles
# Make the diagram
plt.figure(figsize=(10, 6))
labels = ['Use-of-Helmet Rule', 'Uncalled Penalty Rules', 'PR Restricted Zone Rule']
v = venn3(subsets = (5, 4, 11, 5, 2, 2, 0), set_labels=labels, )
c = venn3_circles(subsets = (5, 4, 11, 5, 2, 2, 0), linestyle='dashed')

for text in v.set_labels:
    text.set_fontsize(16)
for text in v.subset_labels:
    text.set_fontsize(18)
plt.title('Rule Coverage of Concussions After the Punt', fontsize=18)
plt.savefig('venn_diagram.png', bbox_inches='tight')
plt.show()
