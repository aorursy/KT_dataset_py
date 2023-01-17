import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.dpi'] = 100
init_notebook_mode(connected=True) 

#print(os.listdir('../input/NFL-Punt-Analytics-Competition'))

data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
players = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
gd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
play_info = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
video = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
positions = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')

data['GSISID'].apply(str)
data['Primary_Partner_GSISID'].apply(str)
players['GSISID'].apply(str)
gd['Turf'] = gd['Turf'].str.lower()
gd['GameWeather'] = gd['GameWeather'].str.lower()

for index, row in data.iterrows():
    role = players[(players.GSISID == row['GSISID']) & (players.GameKey == row['GameKey']) & (players.PlayID == row['PlayID'])]['Role'].tolist()[0]
    
    data.loc[index, 'game_clock'] = play_info[(play_info.PlayID == row['PlayID']) & (play_info.GameKey == row['GameKey'])]['Game_Clock'].tolist()[0]
    data.loc[index, 'video_url'] = video[(video.gamekey == row['GameKey']) & (video.playid == row['PlayID'])]['PREVIEW LINK (5000K)'].tolist()[0]
    data.loc[index, 'game_day'] = gd[(gd.GameKey == row['GameKey'])]['Game_Day'].tolist()[0]
    data.loc[index, 'turf'] = gd[(gd.GameKey == row['GameKey'])]['Turf'].tolist()[0]
    data.loc[index, 'start_time'] = gd[(gd.GameKey == row['GameKey'])]['Start_Time'].tolist()[0]
    data.loc[index, 'weather'] = gd[(gd.GameKey == row['GameKey'])]['GameWeather'].tolist()[0]
    data.loc[index, 'temperature'] = gd[(gd.GameKey == row['GameKey'])]['Temperature'].tolist()[0]
    data.loc[index, 'player_role'] = role
    data.loc[index, 'ball_possession'] = (role in ['GL','PLW','PLT','PLG','PLS','PRG','PRT','PRW','GR','PC','PPR','P'])
    #data.loc[index, 'description'] = video[(video.gamekey == row['GameKey']) & (video.playid == row['PlayID'])]['PlayDescription'].tolist()[0]
    data.loc[index, 'description'] = play_info[(play_info.PlayID == row['PlayID']) & (play_info.GameKey == row['GameKey'])]['PlayDescription'].tolist()[0]
    data.loc[index, 'illegal'] = ('illegal' in video[(video.gamekey == row['GameKey']) & (video.playid == row['PlayID'])]['PlayDescription'].tolist()[0].lower().split(' '))
    
    gsisid = row['GSISID']
    data.loc[index, 'player_pos'] = positions[positions.GSISID == gsisid]['Position'].tolist()[0]
    #print(i, gsisid, positions[positions.GSISID == gsisid]['Position'].tolist()[0])
    
    if str(row['Primary_Partner_GSISID']) != 'nan' and str(row['Primary_Partner_GSISID']) != 'Unclear':
        gsisid = int(row['Primary_Partner_GSISID'])
        data.loc[index, 'partner_pos'] = positions[positions.GSISID == gsisid]['Position'].tolist()[0]
        data.loc[index, 'partner_role'] = players[(players.GSISID == int(row['Primary_Partner_GSISID'])) & (players.GameKey == row['GameKey']) & (players.PlayID == row['PlayID'])]['Role'].tolist()[0]
        #data.loc[index, 'ROLES'] = '{}-{}'.format(str(data.loc[index]['player_role']),str(data.loc[index]['PP_GSISROLE']))
import tqdm
PATH = '../input/NFL-Punt-Analytics-Competition/'

dtypes = {'Season_Year': 'int16',
         'GameKey': 'int16',
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

df_list = []

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv(f'{PATH}'+i, usecols=col_names,dtype=dtypes)
    
    df_list.append(df)
import gc
# Merge all dataframes into one dataframe
ngs = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list
gc.collect()
ngs = ngs.drop(columns=['Event'])
def find_direction(angle):
    if angle < 45 and angle > -45:
        return 'front'
    elif (angle < -45 and angle > -135):
        return  'left'
    elif (angle > 45 and angle < 135):
        return 'right'
    else:
        return 'behind'

def find_half(angle):
    if angle > -90 and angle < 90:
        return 'front'
    else:
        return 'behind'
    

def find_impact(PlayID, GameKey, GSISID, Partner_GSISID, data_index):
    player_coords = ngs[(ngs.GSISID == GSISID) & (ngs.GameKey == GameKey) & (ngs.PlayID == PlayID)]
    partner_coords = ngs[(ngs.GSISID == Partner_GSISID) & (ngs.GameKey == GameKey) & (ngs.PlayID == PlayID)]

    player_coords = player_coords.sort_values(by=['Time'])
    partner_coords = partner_coords.sort_values(by=['Time'])

    p1 = player_coords.index.tolist()
    p2 = partner_coords.index.tolist()
    
    for i in range(0, len(p1)):
        player_index = p1[i]
        partner_index = p2[i]

        distance = math.sqrt((player_coords.loc[player_index]['x'] - partner_coords.loc[partner_index]['x'])**2 + (player_coords.loc[player_index]['y'] - partner_coords.loc[partner_index]['y'])**2)

        if distance < 1.2:
            
            rad = math.atan2(partner_coords.loc[partner_index]['y'] - player_coords.loc[player_index]['y'], partner_coords.loc[partner_index]['x'] - player_coords.loc[player_index]['x'])
            rad = (rad * 180) / math.pi
            if rad < 0:
                rad += 360
                
            rad2 = math.atan2(player_coords.loc[player_index]['y'] - partner_coords.loc[partner_index]['y'], player_coords.loc[player_index]['x'] - partner_coords.loc[partner_index]['x'])
            rad2 = (rad2 * 180) / math.pi
            if rad2 < 0:
                rad2 += 360
                
            collision_angle = player_coords.loc[player_index]['dir']  - rad
            collision_angle_partner = partner_coords.loc[partner_index]['dir']  - rad2
            
            data.loc[data_index, 'collision_angle'] = collision_angle
            data.loc[data_index, 'collision_angle_partner'] = collision_angle_partner
            data.loc[data_index, 'player_speed'] = player_coords.loc[player_index]['dis'] * 0.9144 * 10
            data.loc[data_index, 'partner_speed'] = partner_coords.loc[partner_index]['dis'] * 0.9144 * 10
            data.loc[data_index, 'player_x'] = player_coords.loc[player_index]['x']
            data.loc[data_index, 'player_y'] = player_coords.loc[player_index]['y']
            data.loc[data_index, 'partner_x'] = partner_coords.loc[partner_index]['x']
            data.loc[data_index, 'partner_y'] = partner_coords.loc[partner_index]['y']
            data.loc[data_index, 'collision_time1'] = player_coords.loc[player_index]['Time']
            data.loc[data_index, 'collision_time2'] = partner_coords.loc[partner_index]['Time']
            data.loc[data_index, 'player_dir'] = player_coords.loc[player_index]['dir']
            data.loc[data_index, 'player_o'] = player_coords.loc[player_index]['o']
            data.loc[data_index, 'partner_dir'] = partner_coords.loc[partner_index]['dir']
            data.loc[data_index, 'partner_o'] = partner_coords.loc[partner_index]['o']
            data.loc[data_index, 'distance'] = distance
            
            partner_side = find_direction(collision_angle)
            partner_half = find_half(collision_angle)
            
            player_side = find_direction(collision_angle_partner)
            player_half = find_half(collision_angle_partner)            
                
            data.loc[data_index, 'partner_side'] = partner_side
            data.loc[data_index, 'partner_half'] = partner_half
            
            data.loc[data_index, 'player_side'] = player_side
            data.loc[data_index, 'player_half'] = player_half
            
            avg_player_speed = 0
            avg_partner_speed = 0
            
            """
            for y in range(0, i):
                p_index = p1[y]
                p2_index = p2[y]
                
                avg_player_speed += player_coords.loc[p_index]['dis'] * 0.9144
                avg_partner_speed += partner_coords.loc[p2_index]['dis'] * 0.9144
                
            #data.loc[data_index, 'player_avg_speed'] = (avg_player_speed / i) * 10
            #data.loc[data_index, 'partner_avg_speed'] = (avg_partner_speed / i) * 10
            """
            # Nombre de joueurs dans le périmètre à l'impact 5 yards
            
            perimeter_players = 0
            for index in ngs[(ngs.GameKey == GameKey) & (ngs.PlayID == PlayID) & (ngs.Time == player_coords.loc[player_index]['Time'])].index:
                perimeter_player = ngs.loc[index]
                distance = math.sqrt((player_coords.loc[player_index]['x'] - perimeter_player[(perimeter_player.GameKey == GameKey)]['x'])**2 + (player_coords.loc[player_index]['y'] - perimeter_player[(perimeter_player.GameKey == GameKey)]['y'])**2)
                
                if distance < 3:
                    perimeter_players += 1
            
            perimeter_players -= 2
            data.loc[data_index, 'perimeter_players'] = perimeter_players
            
            return True
        

for i in data.index.tolist():
    PlayID = data.loc[i]['PlayID']
    GameKey = data.loc[i]['GameKey']
    GSISID = float(data.loc[i]['GSISID'])
    
    if str(data.loc[i]['Primary_Partner_GSISID']) != 'nan' and str(data.loc[i]['Primary_Partner_GSISID']) != 'Unclear':
        Partner_GSISID = float(data.loc[i]['Primary_Partner_GSISID'])
        
        find_impact(PlayID, GameKey, GSISID, Partner_GSISID, i)

data.head()
# Creating a dataframe to show turf relative concussions rate
graph_data = pd.DataFrame()

for turf in gd['Turf'].unique():
    # data cleaning
    if isinstance(turf, str) and turf.startswith('nat'):
        turf = 'natural grass'
    
    if gd[(gd.Turf == turf)]['Turf'].count() >= 5:

        graph_data.loc[turf, 'Total'] = gd[(gd.Turf == turf)]['Turf'].count()
        graph_data.loc[turf, 'Concussions'] = data[(data.turf == turf)]['turf'].count()
        graph_data.loc[turf, 'NoConcussions'] = gd[(gd.Turf == turf)]['Turf'].count() - data[(data.turf == turf)]['turf'].count()
    
# Sorting the data!
graph_data = graph_data.sort_values(by=['Total'])

ind = np.arange(len(graph_data))
width = 0.7

# Putting a part of the plot above the other one
p1 = plt.bar(ind, graph_data['Concussions'].tolist(), width, yerr=None, color='red')
p2 = plt.bar(ind, graph_data['NoConcussions'].tolist(), width, yerr=None, color='lightblue', bottom=graph_data['Concussions'].tolist())

# Labelling, legends, etc...
plt.ylabel('Match count')
plt.title('Concussion relative to total games on each turf (at least 5 games played on it)')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.yticks(np.arange(0, 250, 25))
plt.legend((p1[0], p2[0]), ('Match with a punt concussion', 'Match with no punt concussion'))

plt.show()
# Creating a dataframe to show weather relative concussions rate
graph_data = pd.DataFrame()
for weather in gd['GameWeather'].unique().tolist():
    if type(weather) is str and gd[(gd.GameWeather == weather)]['GameWeather'].count() >= 5:
        
        graph_data.loc[weather, 'Total'] = gd[(gd.GameWeather == weather)]['GameWeather'].count()
        graph_data.loc[weather, 'Concussions'] = data[(data.weather == weather)]['weather'].count()
        graph_data.loc[weather, 'NoConcussions'] = gd[(gd.GameWeather == weather)]['GameWeather'].count() - data[(data.weather == weather)]['weather'].count()
    

# Sorting the data!
graph_data = graph_data.sort_values(by=['Total'])

ind = np.arange(len(graph_data))
width = 0.7

# Putting a part of the plot above the other one
p1 = plt.bar(ind, graph_data['Concussions'].tolist(), width, yerr=None, color='red')
p2 = plt.bar(ind, graph_data['NoConcussions'].tolist(), width, yerr=None, color='lightblue', bottom=graph_data['Concussions'].tolist())

# Labelling, legends, etc...
plt.ylabel('Match count')
plt.title('Concussion relative to total games for each weather (at least 5 games played on it)')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.yticks(np.arange(0, 170, 15))
plt.legend((p1[0], p2[0]), ('Match with a punt concussion', 'Match with no punt concussion'))

plt.show()
# Creating a dataframe to show start time relative concussions rate
graph_data = pd.DataFrame()

for start_time in gd['Start_Time'].unique().tolist():
    if gd[(gd.Start_Time == start_time)]['Start_Time'].count() >= 5:
        graph_data.loc[start_time, 'Total'] = gd[(gd.Start_Time == start_time)]['Start_Time'].count()
        graph_data.loc[start_time, 'Concussions'] = data[(data.start_time == start_time)]['start_time'].count()
        graph_data.loc[start_time, 'NoConcussions'] = gd[(gd.Start_Time == start_time)]['Start_Time'].count() - data[(data.start_time == start_time)]['start_time'].count()


# Sorting the data!
graph_data = graph_data.sort_values(by=['Total'])

ind = np.arange(len(graph_data))
width = 0.7

# Putting a part of the plot above the other one
p1 = plt.bar(ind, graph_data['Concussions'].tolist(), width, yerr=None, color='red')
p2 = plt.bar(ind, graph_data['NoConcussions'].tolist(), width, yerr=None, color='lightblue', bottom=graph_data['Concussions'].tolist())

# Labelling, legends, etc...
plt.ylabel('Match count')
plt.title('Concussion relative to total games for each start time (at least 5 games played on it)')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.yticks(np.arange(0, 210, 15))
plt.legend((p1[0], p2[0]), ('Match with a punt concussion', 'Match with no punt concussion'))

plt.show()
# Creating a dataframe to show game day relative concussions rate
graph_data = pd.DataFrame()
for game_day in gd['Game_Day'].unique().tolist():
    graph_data.loc[game_day, 'Total'] = gd[(gd.Game_Day == game_day)]['Game_Day'].count()
    graph_data.loc[game_day, 'Concussions'] = data[(data.game_day == game_day)]['game_day'].count()
    graph_data.loc[game_day, 'NoConcussions'] = gd[(gd.Game_Day == game_day)]['Game_Day'].count() - data[(data.game_day == game_day)]['game_day'].count()
    

# Sorting the data!
graph_data = graph_data.sort_values(by=['Total'])

ind = np.arange(len(graph_data))
width = 0.7

# Putting a part of the plot above the other one
p1 = plt.bar(ind, graph_data['Concussions'].tolist(), width, yerr=None, color='red')
p2 = plt.bar(ind, graph_data['NoConcussions'].tolist(), width, yerr=None, color='lightblue', bottom=graph_data['Concussions'].tolist())

# Labelling, legends, etc...
plt.ylabel('Match count')
plt.title('Concussion relative to total games for each game')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.yticks(np.arange(0, 450, 25))
plt.legend((p1[0], p2[0]), ('Match with a punt concussion', 'Match with no punt concussion'))

plt.show()
sns.countplot(y='player_role', data=data, order=data['player_role'].value_counts().index)
plt.title('Role victim of concussion count')
plt.ylabel('Roles')
plt.show()
sns.countplot(y='partner_role', data=data, order=data['partner_role'].value_counts().index)
plt.title('Partner role count involved in concussion')
plt.ylabel('Roles')
plt.show()
corr_data = data[(data['partner_role'] == 'PR') | (data['player_role'] == 'PR')]
corr_data = corr_data.drop(['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Primary_Partner_GSISID','collision_time1','collision_time2', 'video_url', 'description'], axis=1)
subcorr = corr_data.corr()

for column in corr_data.columns:
    if column not in subcorr.columns.tolist():
        corr_data[column] = corr_data[column].astype('category').cat.codes

corr = corr_data.corr()
f, ax = plt.subplots(figsize=(16, 8))
heatmap = sns.heatmap(corr)
sns.countplot(y='Player_Activity_Derived', data=data[(data['player_role'] == 'PR')], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Punt returner (as victim) activity')
plt.ylabel('Activity')
plt.show()
sns.countplot(y='Player_Activity_Derived', data=data[(data['partner_role'] == 'PR')], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Victim activity when the partner is a Punt Returner')
plt.ylabel('Activity')
plt.show()
sns.countplot(y='Primary_Impact_Type', data=data[(data['player_role'] == 'PR')], order=data['Primary_Impact_Type'].value_counts().index)
plt.title('Punt returner (as victims) impact type')
plt.ylabel('Impact type')
plt.show()
sns.countplot(y='Primary_Impact_Type', data=data[(data['partner_role'] == 'PR')], order=data['Primary_Impact_Type'].value_counts().index)
plt.title('Victim impact type when the partner is a Punt Returner')
plt.ylabel('Impact type')
plt.show()
sns.countplot(y='Primary_Impact_Type', data=data[(data['partner_role'] == 'PR') | (data['player_role'] == 'PR')], order=data['Primary_Impact_Type'].value_counts().index)
plt.title('Sum of both cases')
plt.show()
# merging punt returners partners
part1 = data[(data['player_role'] == 'PR')]['partner_role']
part2 = data[(data['partner_role'] == 'PR')]['player_role']

partners = pd.DataFrame()
for i in part1.index:
    partners.loc[i, 'partner_role'] = part1.loc[i]

for i in part2.index:
    partners.loc[i, 'partner_role'] = part2.loc[i]

sns.countplot(y='partner_role', data=partners, order=partners['partner_role'].value_counts().index)
plt.title('Punt returner partners')
plt.ylabel('Role')
plt.show()
sns.countplot(y='partner_side', data=data[(data['player_role'] == 'PR') & (data['Player_Activity_Derived'] == 'Tackled')], order=data['player_side'].value_counts().index)
plt.title('Side from which Punt Returners (as victims) are tackled')
plt.ylabel('side')
plt.show()
sns.countplot(y='player_side', data=data[(data['partner_role'] == 'PR') & (data['Player_Activity_Derived'] == 'Tackling')], order=data['player_side'].value_counts().index)
plt.title('Side from which the player (as victim) tackles the punt returner')
plt.ylabel('side')
plt.show()
part1 = data[(data['player_role'] == 'PR')]['player_speed']
part2 = data[(data['partner_role'] == 'PR')]['partner_speed']

result = pd.DataFrame()
for i in part1.index:
    result.loc[i, 'speed'] = part1.loc[i]

for i in part2.index:
    result.loc[i, 'speed'] = part2.loc[i]

result = result.dropna()
sns.distplot(result['speed'])
plt.title('Punt returners (as victims) speed distribution')
plt.xlabel('speed in m/s')
plt.show()
part1 = data[(data['player_role'] == 'PR')]['partner_speed']
part2 = data[(data['partner_role'] == 'PR')]['player_speed']

result = pd.DataFrame()
for i in part1.index:
    result.loc[i, 'speed'] = part1.loc[i]

for i in part2.index:
    result.loc[i, 'speed'] = part2.loc[i]

result = result.dropna()
sns.distplot(result['speed'])
plt.title('Victims speed distribution when tackling')
plt.xlabel('speed in m/s')
plt.show()
corr_data = data[(data['partner_role'] != 'PR') & (data['player_role'] != 'PR')]
corr_data = corr_data.drop(['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Primary_Partner_GSISID','collision_time1','collision_time2', 'video_url', 'description'], axis=1)
subcorr = corr_data.corr()

for column in corr_data.columns:
    if column not in subcorr.columns.tolist():
        corr_data[column] = corr_data[column].astype('category').cat.codes

corr = corr_data.corr()
f, ax = plt.subplots(figsize=(18, 9))
heatmap = sns.heatmap(corr)
sns.countplot(y='ball_possession', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR')], order=data['ball_possession'].value_counts().index)
plt.title('Ball possession')
plt.ylabel('Victim from punting team ? (PRs excluded)')
plt.show()
sns.countplot(y='Player_Activity_Derived', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR')], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Both sides - Victim activity (PRs excluded)')
plt.ylabel('Activity')
plt.show()
has_ball = True
sns.countplot(y='Player_Activity_Derived', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == has_ball)], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Punting team - Victim activity (PRs excluded)')
plt.ylabel('Activity')
plt.show()
has_ball = False
sns.countplot(y='Player_Activity_Derived', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == has_ball)], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Returning team - Victim activity (PRs excluded)')
plt.ylabel('Activity')
plt.show()
sns.countplot(y='Primary_Impact_Type', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR')], order=data['Primary_Impact_Type'].value_counts().index)
plt.title('Impact types (PRs excluded)')
plt.ylabel('Impact')
plt.show()
sns.countplot(y='player_role', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == True)], order=data['player_role'].value_counts().index)
plt.title('Punting team - Victim role (PRs excluded)')
plt.ylabel('Role')
plt.show()
sns.countplot(y='player_role', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == False)], order=data['player_role'].value_counts().index)
plt.title('Returning team - Victim role (PRs excluded)')
plt.ylabel('Role')
plt.show()
sns.countplot(y='partner_role', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == False)], order=data['player_role'].value_counts().index)
plt.title('Returning team - Blocked partners (PRs excluded)')
plt.ylabel('Role')
plt.show()
sns.countplot(y='partner_role', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['ball_possession'] == True)], order=data['partner_role'].value_counts().index)
plt.title('Player roles involved in concussions (PRs excluded)')
plt.show()
sns.distplot(data[(data['player_role'] != 'PR') & (data['Player_Activity_Derived'] =='Tackling')]['partner_speed'].dropna())
sns.distplot(data[(data['player_role'] != 'PR') & (data['Player_Activity_Derived'] =='Tackling')]['player_speed'].dropna(), hist_kws={'alpha':0.25})
plt.legend(['Target speed','Victim speed'])
plt.title('Tackling speed distribution (PRs excluded)')
plt.xlabel('speed m/s')
plt.show()
sns.distplot(data[(data['player_role'] != 'PR') & (data['Player_Activity_Derived'] == 'Blocking')]['partner_speed'].dropna())
sns.distplot(data[(data['player_role'] != 'PR') & (data['Player_Activity_Derived'] == 'Blocking')]['player_speed'].dropna(), hist_kws={'alpha':0.25})
plt.legend(['Target speed','Victim speed'])
plt.title('Blocking speed distribution (PRs excluded)')
plt.xlabel('speed m/s')
plt.show()
sns.countplot(y='partner_side', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR')], order=data['player_side'].value_counts().index)
plt.title('Side from which partner comes leading to concussion')
plt.ylabel('side')
plt.show()
sns.countplot(y='Player_Activity_Derived', data=data[(data['player_role'] != 'PR') & (data['partner_role'] != 'PR') & (data['player_side'] == 'behind')], order=data['Player_Activity_Derived'].value_counts().index)
plt.title('Activities leading to concussion when coming from behind the target (PRs excluded)')
plt.ylabel('Activity')
plt.show()
sns.countplot(y='partner_side', data=data[(data['Friendly_Fire'] == 'No')], order=data['partner_side'].value_counts().index)
plt.title('Side from which partner coming from which leads to concussion | Excluded friendly fire')
plt.ylabel('side')
plt.show()
sns.countplot(y='illegal', data=data, order=data['illegal'].value_counts().index)
plt.title('Illegal move leading to concussion count')
plt.ylabel('Move illegal ?')
plt.show()
for i in play_info.index:
    if 'punts' in play_info.loc[i, 'PlayDescription'].split(' '):
        description = play_info.loc[i, 'PlayDescription'].split(' ')
        play_info.loc[i, 'punt_dist'] = int(description[description.index('punts') + 1])
    else:
        play_info.loc[i, 'punt_dist'] = -1
        
for i in data.index:
    if 'punts' in data.loc[i, 'description'].split(' '):
        description = data.loc[i, 'description'].split(' ')
        data.loc[i, 'punt_dist'] = int(description[description.index('punts') + 1])
    else:
        data.loc[i, 'punt_dist'] = -1

sns.distplot(play_info[play_info.punt_dist > 0]['punt_dist'])
sns.distplot(data[data.punt_dist > 0]['punt_dist'], hist_kws={'alpha':0.25})        
plt.title('Punt distance distribution')
plt.legend(['All punts', 'Punt with concussion'])
plt.xlabel('Punt distance in yards')
plt.show()
sample = data['perimeter_players'].dropna()
sns.distplot(sample)
plt.title('Player in perimeter distribution')
plt.xlabel('Number of players in a 3 yards perimeter')
plt.show()
concussion_gamekey = data['GameKey'].values.tolist()
concussion_punts = 0
normal_punts = 0

normal_gamekey = play_info['GameKey'].unique().tolist()
for i in normal_gamekey:
    if i in concussion_gamekey:
        concussion_punts += len(play_info[play_info.GameKey == i]['Play_Type'])
    else:
        normal_punts += len(play_info[play_info.GameKey == i]['Play_Type'])
        
print('Average punt in a game with concussion {} ({} punts / {} games)'.format(concussion_punts/ len(concussion_gamekey),concussion_punts , len(concussion_gamekey)))
print('Average punt in a game with no concussion {} ({} punts / {} games)'.format(normal_punts/ len(normal_gamekey), normal_punts , len(normal_gamekey)))