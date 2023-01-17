import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import Image, display
player_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
play_player_role_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
video_review_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
print(video_review_df.shape)
# Combine Relevant Player Information, Position, Role, Number
master_player_df = pd.merge(player_df, play_player_role_df,
                          how='inner',
                          on=['GSISID']).drop(columns=['Season_Year'])
master_player_df.head()
# Check summary counts to see if any are even needed
print(video_review_df['Player_Activity_Derived'].value_counts())
print('---')
print(video_review_df['Turnover_Related'].value_counts())
print('---')
print(video_review_df['Primary_Impact_Type'].value_counts())
print('---')
print(video_review_df['Primary_Partner_Activity_Derived'].value_counts())
print('---')
print(video_review_df['Friendly_Fire'].value_counts())
# Now that we have a descriptive idea of what's going on, I'm gonna just drop these columns
# and also clear up the unclear designation and convert it to NaN
droppers = ['Player_Activity_Derived', 'Turnover_Related', 'Primary_Partner_Activity_Derived', 
            'Friendly_Fire', 'Season_Year']
video_review_df.drop(columns=droppers, inplace=True)

# Remove 'Unclear' designation
video_review_df.loc[33, 'Primary_Partner_GSISID'] = 'NaN'
'''Control Video Footage'''
video_footage_control_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
print(video_footage_control_df.shape)
video_footage_control_df.tail(1)
# # Use for printing out video links
# for i in range(len(video_footage_control_df)):
#     print(video_footage_control_df.loc[i, 'Preview Link'])
'''Concussion Video Footage'''
video_footage_injury_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
print(video_footage_injury_df.shape)
video_footage_injury_df.head(1)
# # Injury Video Links; search for player who is injured; watch the film, be the film....
# for i in range(len(injured_players)):
#     print(i, injured_players['PREVIEW LINK (5000K)'][i])
# Preprocess to allow easier join between video review data and the actual footage data
rename_columns = {'gamekey': 'GameKey', 'playid': 'PlayID', 'season': 'Season_Year'}
video_footage_injury_df.rename(columns=rename_columns, inplace=True)

# Combine Video Review and Video Injury DataFrames to have the injured player and partner player data
injury_play = pd.merge(video_review_df, video_footage_injury_df, 
                       how='inner', 
                       on=['GameKey', 'PlayID'])
injury_play.head(1)
# Lets Drop Some More Data I consider uncritical for getting a feel for the data
droppers = ['Season_Year', 'Type', 'Week', 'Home_team', 'Visit_Team', 'Qtr']
injury_play.drop(columns=droppers, inplace=True)
# Join Info (their jersey number, position, role) on Players Who are Injured
injured_players = pd.merge(injury_play, master_player_df,
                           how='inner',
                           on=['GSISID', 'GameKey', 'PlayID'])

print('Shape:', injured_players.shape)
injured_players.head(1)
# Drop Certain Rows After Identifying Concussed Players Jersey Number from video
drop_rows = [1, 7, 17, 20, 23, 26, 27, 28, 29, 32, 33, 36, 38, 39, 45, 46, 47, 50, 52, 55, 57]
injured_players.drop(labels=drop_rows, inplace=True)
injured_players.reset_index(drop=True, inplace=True)
# Read in only concussion data (dataset formed in a different notebook)
# Contains NGS data for plays involving a concussion
ngs_concussion = pd.read_csv('../input/ngsconcussion/NGS-concussion.csv')
ngs_concussion.head()
# Convert Primary_Partner_GSISID from str to float
injured_players['Primary_Partner_GSISID'] = injured_players['Primary_Partner_GSISID'].astype('float')
from IPython.display import display, HTML

def make_html(game_key, play_id):
    return '<img src="{}" style="display:inline;margin:1px"/>'\
    .format('../input/ngsconcussion/' + str(game_key) + '_' + str(play_id) + '.gif')
# Map Routes of concussed player and partner player
# and give approximate speeds throughout their route
for i in range(len(injured_players)):
    # Get necessary values for query of NGS data
    game_key = injured_players.loc[i, 'GameKey']
    play_id = injured_players.loc[i, 'PlayID']
    concussed_id = injured_players.loc[i, 'GSISID']
    partner_id = injured_players.loc[i, 'Primary_Partner_GSISID']
    print('GameKey:', game_key, 'PlayID:', play_id)
    print('Play Description:', injured_players.loc[i,'PlayDescription'])
    print('Primary Impact Type:', injured_players.loc[i, 'Primary_Impact_Type'])
    print('Concussed:', concussed_id, 'Role:', injured_players.loc[i, 'Role'])
    print('Partner:', partner_id)
    # Visualizing play with .gif file
    display(HTML(''.join(make_html(game_key, play_id))))
    
    # Concussed player
    where_condition = (
        (ngs_concussion['GameKey'] == game_key)&\
        (ngs_concussion['PlayID'] == play_id) &\
        (ngs_concussion['GSISID'] == concussed_id))
    concussion = ngs_concussion[where_condition].copy()
    # Reorder by Time and reset index
    concussion.sort_values(by=['Time'], inplace=True)
    concussion.reset_index(drop=True, inplace=True)
    
    # Partner player
    where_condition = (
        (ngs_concussion['GameKey'] == game_key)&\
        (ngs_concussion['PlayID'] == play_id) &\
        (ngs_concussion['GSISID'] == partner_id))
    partner = ngs_concussion[where_condition].copy()
    partner.sort_values(by=['Time'], inplace=True)
    partner.reset_index(drop=True, inplace=True)

    # Variables for Mapping
    concussion_x = concussion['x']
    concussion_y = concussion['y']
    partner_x = partner['x']
    partner_y = partner['y']
    speed1 = concussion['dis'] / 0.1
    speed2 = partner['dis'] / 0.1
    
    # Mapping of play
    sns.set()
    plt.figure(figsize=(10,5))
    cmap = plt.get_cmap('coolwarm')
    plt.scatter(concussion_x, concussion_y, c=speed1, cmap=cmap, alpha=0.5)
    if partner_id != 'NaN':
        plt.scatter(partner_x, partner_y, c=speed2, cmap=cmap, alpha=0.5)
    plt.clim(0, 12)
    plt.colorbar(label='yards/sec')
    # Normal length of field is 120 yards
    plt.xlim(-10, 130)
    plt.xticks(np.arange(0, 130, step=10),
               ['End', 'Goal Line', '10', '20', '30', '40', '50', '40', '30', '20', '10', 'Goal Line', 'End'])
    # Normal width is 53.3 yards
    plt.ylim(-10, 65)
    plt.yticks(np.arange(0, 65, 53.3), ['Sideline', 'Sideline'])
    plt.title('Playing Field')
    plt.xlabel('yardline')
    plt.ylabel('width of field')
    plt.show()
    print('---')
