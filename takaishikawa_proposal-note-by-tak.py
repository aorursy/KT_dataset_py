from IPython.display import YouTubeVideo
YouTubeVideo("voFHkaRWXvg")
import numpy as np
import pandas as pd
import glob
from plotly import offline
import plotly.graph_objs as go
import warnings; warnings.filterwarnings("ignore")

from IPython.display import HTML
pd.set_option('max.columns', None)
offline.init_notebook_mode()
config = dict(showLink=False)
import numpy as np 
import pandas as pd 

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
"""
https://www.kaggle.com/slamer/speed-calculation-quick
"""

import numpy as np
import pandas as pd
import glob

def calculate_speeds(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x.values / 1.0936132983
        data_selected.y = data_selected.y.values / 1.0936132983
    # Might have used shift pd function ?
    data_selected_diff = data_selected.diff()
    if dt==None:
        # Time is now a timedelta and need to be converted
        data_selected_diff.Time = data_selected_diff.Time.apply(lambda x: (x.total_seconds()))
        data_selected_diff['Speed'] = np.sqrt(data_selected_diff.x.values **2 + data_selected_diff.y.values **2) / data_selected_diff.Time.values
    else:
        # Need to be sure about the time step...
        data_selected_diff['Speed'] = (data_selected_diff.x.values **2 + data_selected_diff.y.values **2).astype(np.float64).apply(np.sqrt) / dt.values
    #data_selected_diff.rename(columns={'Time':'TimeDelta'}, inplace=True)
    #return data_selected_diff
    df['TimeDelta'] = data_selected_diff.Time
    df['Speed'] = data_selected_diff.Speed
    return df[1:]

def remove_wrong_values(df, tested_columns=['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'TimeDelta'], cutspeed=None):
    dump = df.copy()
    colums = dump.columns
    mask = []
    for col in tested_columns:
        dump['shift_'+col] = dump[col].shift(-1)
        mask.append("( dump['shift_"+col+"'] == dump['"+col+"'])")
    mask =eval(" & ".join(mask))
    # Keep results where next rows is equally space
    dump = dump[mask]
    dump = dump[colums]
    if cutspeed!=None:
        dump = dump[dump.Speed < cutspeed]
    return dump

def get_speed(df):
    df_with_speed = df.copy()
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df_with_speed.Time = pd.to_datetime(df_with_speed.Time, format =date_format)
    df_with_speed.sort_values(sortBy, inplace=True)
    df_with_speed = calculate_speeds(df_with_speed, SI=True)
    cut_speed=100 / 9.58 # World record 9,857232 m/s for NFL
    df_with_speed = remove_wrong_values(df_with_speed, cutspeed=cut_speed)
    return df_with_speed
import pandas as pd
import glob
from plotly import offline
import plotly.graph_objs as go

offline.init_notebook_mode()
config = dict(showLink=False)

import os
import pandas as pd
import numpy as np
from IPython.display import HTML
import seaborn as sns
# import squarify
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
import plotly.figure_factory as ff
#Always run this the command before at the start of notebook
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# import cufflinks as cf
# cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
import warnings
warnings.filterwarnings('ignore')
sns.set_style("white")
plt.style.use('seaborn')


"""
Returns a dict for a Football themed Plot.ly layout 
""" 
def load_layout():
    layout = dict(
        title = "Player Activity",
        plot_bgcolor='darkseagreen',
        showlegend=True,
        xaxis=dict(
            autorange=False,
            range=[0, 120],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            tickmode='array',
            tickvals=[10,20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            ticktext=['Goal', 10, 20, 30, 40, 50, 40, 30, 20, 10, 'Goal'],
            showticklabels=True
        ),
        yaxis=dict(
            title='',
            autorange=False,
            range=[-3.3,56.3],
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            linewidth=1,
            mirror=True,
            ticks='',
            showticklabels=False
        ),
        shapes=[
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=0,
                x1=120,
                y1=0,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=0,
                y0=53.3,
                x1=120,
                y1=53.3,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=10,
                y0=0,
                x1=10,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=20,
                y0=0,
                x1=20,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=30,
                y0=0,
                x1=30,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=40,
                y0=0,
                x1=40,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=50,
                y0=0,
                x1=50,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=60,
                y0=0,
                x1=60,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=70,
                y0=0,
                x1=70,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=80,
                y0=0,
                x1=80,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=90,
                y0=0,
                x1=90,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),dict(
                type='line',
                layer='below',
                x0=100,
                y0=0,
                x1=100,
                y1=53.3,
                line=dict(
                    color='white'
                )
            ),
            dict(
                type='line',
                layer='below',
                x0=110,
                y0=0,
                x1=110,
                y1=53.3,
                line=dict(
                    color='white',
                    width=10
                )
            )
        ]
    )
    return layout
    
    
"""
Loading and plotting functions
"""

def load_player_role(GSISID):
    player_role = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
    player_role = player_role[player_role['GSISID'] == GSISID]
    return player_role


def load_plays_for_game(GameKey):
    """
    Returns a dataframe of play data for a given game (GameKey)
    """
    play_information = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
    play_information = play_information[play_information['GameKey'] == GameKey]
    return play_information


def load_game_and_ngs(ngs_file=None, GameKey=None):
    """
    Returns a dataframe of player movements (NGS data) for a given game
    """
    if ngs_file is None:
        print("Specifiy an NGS file.")
        return None
    if GameKey is None:
        print('Specify a GameKey')
        return None
    # Merge play data with NGS data    
    plays = load_plays_for_game(GameKey)
    ngs = pd.read_csv(ngs_file, low_memory=False)
    merged = pd.merge(ngs, plays, how="inner", on=["GameKey", "PlayID", "Season_Year"])
    return merged


def plot_play(game_df, PlayID, player1=None, player2=None, custom_layout=False):
    """
    Plots player movements on the field for a given game, play, and two players
    """
    game_df = game_df[game_df.PlayID==PlayID]
    
    GameKey=str(pd.unique(game_df.GameKey)[0])
    HomeTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[0]
    VisitingTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[1]
    YardLine = game_df[(game_df.PlayID==PlayID) & (game_df.GSISID==player1)]['YardLine'].iloc[0]
    
    traces=[]   
    if (player1 is not None) & (player2 is not None):
        game_df = game_df[ (game_df['GSISID']==player1) | (game_df['GSISID']==player2)]
        for player in pd.unique(game_df.GSISID):
            player = int(player)
            trace = go.Scatter(
                x = game_df[game_df.GSISID==player].x,
                y = game_df[game_df.GSISID==player].y,
                name='GSISID '+str(player),
                mode='markers'
            )
            traces.append(trace)
    else:
        print("Specify GSISIDs for player1 and player2")
        return None
    
    if custom_layout is not True:
        layout = load_layout()
        layout['title'] =  HomeTeam + \
        ' vs. ' + VisitingTeam + \
        '<br>Possession: ' + \
        YardLine.split(" ")[0] +'@'+YardLine.split(" ")[1]
    data = traces
    fig = dict(data=data, layout=layout)
    play_description = game_df[(game_df.PlayID==PlayID) & (game_df.GSISID==player1)].iloc[0]["PlayDescription"]
    print("\n\n\t",play_description)
    offline.iplot(fig, config=config)
    

"""
https://www.kaggle.com/garlsham/nfl-concussion-analysis-inc-suggested-changes
"""

def get_NGS_data():
    dir_path = os.getcwd()
    files = os.listdir('../input/')
    substring = 'NGS'
    
    NGS_files = []
    for item in files:
        if substring in item:
            NGS_files.append(item)
    
    data = pd.read_csv('../input/' + NGS_files[0])    
    for file in NGS_files[1:]:
        temp_data = pd.read_csv('../input/' + file)
        data.append(temp_data)
        
    return data

def vid_review_data():
    dir_path = os.getcwd()
    vid_data =  pd.read_csv('../input/NFL-Punt-Analytics-Competition/' + 'video_review.csv')
        
    return vid_data

def vid_injury_data():
    dir_path = os.getcwd()
    vid_data =  pd.read_csv('../input/NFL-Punt-Analytics-Competition/' + 'video_footage-injury.csv')
        
    return vid_data

def player_role_data():
    dir_path = os.getcwd()
    role_data =  pd.read_csv('../input/NFL-Punt-Analytics-Competition/' + 'play_player_role_data.csv')
        
    return role_data

def player_punt_data():
    dir_path = os.getcwd()
    punt_data =  pd.read_csv('../input/NFL-Punt-Analytics-Competition/' + 'player_punt_data.csv')
        
    return punt_data

def play_info_data():
    dir_path = os.getcwd()
    play_data =  pd.read_csv('../input/NFL-Punt-Analytics-Competition/' + 'play_information.csv')
        
    return play_data
    
    
def perform_merge(data1, data2, columns):
    merged_data = pd.merge(data1, data2, left_on=columns, right_on=columns, suffixes=['','_1'])
    return merged_data


def visiting_data(data):
    score_away = []
    away_team = []
    for item in data['Score_Home_Visiting']:
        scores = item.split('-')
        temp =  int(scores[1])
        score_away.append(temp)
        
    for item in data['Home_Team_Visit_Team']:
        teams = item.split('-')
        temp =  teams[1].strip()
        away_team.append(temp)
        
    data['visiting_team'] = away_team    
    data['visit_score'] = score_away
    
    return data

def home_data(data):
    home_score = []
    home_team = []
    for item in data['Score_Home_Visiting']:
        scores = item.split('-')
        temp =  int(scores[0])
        home_score.append(temp)
        
    for item in data['Home_Team_Visit_Team']:
        teams = item.split('-')
        temp =  teams[0].strip()
        home_team.append(temp)

    data['home_team'] = home_team 
        
    data['home_score'] = home_score
    
    return data

def score_difference(data):
    data['score_diff'] = abs(data['home_score'] - data['visit_score'])
    
    return data


def punt_received(data):
    yards_gained = []
    
    for row in data['PlayDescription']:
        temp = row.split('punts')[1].split(' ')[1]
        yards_gained.append(int(temp))
    
    data['kicked_to'] = yards_gained
    data['kicked_to'] = data['kicked_to'] + data['kicked_from']
    return data

def punt_from(data):
    yardline = []
    
    for row in data['YardLine']:
        temp = row.split(' ')[1]
        yardline.append(int(temp))
    
    data['kicked_from'] = yardline
    return data

def opposition_team(data): 
    opposition = []
    
    for item in data.iterrows():
        teams = item[1]['Home_Team_Visit_Team'].split('-')
        poss_team = item[1]['Poss_Team']
        for element in teams:
            if poss_team != element:
                opposition.append(element)
    data['oppostion'] = opposition

    return data


def draw_pitch(data, col1, col2, title, poss_team, oppostion):
    #layout sourced from https://fcpython.com/visualisation/drawing-pitchmap-adding-lines-circles-matplotlib
    #pitch is 53 yards by 100 yards excluding two 10 yard touchdown zones.
    labels = ['Goal','10','20','30','40','50','40','30','20','10','Goal']
    fig = plt.figure(facecolor='white', figsize=(12.5,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor('green')
    plt.yticks([]) # disable yticks
    
    start_x = -10
    bottom_y = 0
    top_y = 53
    
    ticks = [item * 10 for item in range(0,11)]
    #(x1,x2) (y1,y2)
    
    plt.plot([-10, 110],[0, 0], color='white', linewidth=4)
    plt.plot([-10, 110],[53, 53], color='white', linewidth=4)

    
    for item in range(0,28):
        if item == 0:
            plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
        
        if item >=1  and item <= 28:
            if item % 2 == 1:
                if item == 0 or item == 27:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5
                else:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linestyle="dashed")
                    start_x = start_x + 5
                
            else:
                if start_x >=0 and start_x < 110:
                    plt.plot([start_x, start_x],[bottom_y, top_y], color='white', linewidth=4)
                    start_x = start_x + 5 
                    
    y_value = []
    for i in range(len(data)):
        y_value.append(10 + i * 5)
                    
    for item in range(len(data)):
        plt.scatter(data[col1][item], y_value[item], s=80, color="red")
        plt.scatter(data[col2][item], y_value[item], s=80, color="yellow")
        ax.text(data[col1][item], y_value[item], poss_team[item], ha='left', size=12.5, color='black')
        ax.text(data[col2][item], y_value[item], oppostion[item], ha='left', size=12.5, color='black')

    plt.xticks(ticks, labels, size=15)
    plt.title(title, fontsize=20)
    plt.show()
def plot_count_category(df, column):
    x = df[column].value_counts().index
    y = df[column].value_counts()
    trace = go.Bar(
        x=x,
        y=y
    )
    data = [trace]
    offline.iplot(data, config=config)
    
def plot_punt_team(df, column, g1, g2):
    trace1 = go.Bar(
        x=df[df["punt_position_team"]==g1][column].value_counts().index,
        y=df[df["punt_position_team"]==g1][column].value_counts(),
        name=g1
    )
    trace2 = go.Bar(
        x=df[df["punt_position_team"]==g2][column].value_counts().index,
        y=df[df["punt_position_team"]==g2][column].value_counts(),
        name=g2
    )
    data = [trace1, trace2]
    layout = go.Layout(barmode='group')
    fig = go.Figure(data=data, layout=layout)
    offline.iplot(fig, config=config)
    
def plot_punt_initial(df, column, g1, g2, g3):
    trace1 = go.Bar(
        x=df[df["punt_position_initial"]==g1][column].value_counts().index,
        y=df[df["punt_position_initial"]==g1][column].value_counts(),
        name=g1
    )
    trace2 = go.Bar(
        x=df[df["punt_position_initial"]==g2][column].value_counts().index,
        y=df[df["punt_position_initial"]==g2][column].value_counts(),
        name=g2
    )
    trace3 = go.Bar(
        x=df[df["punt_position_initial"]==g3][column].value_counts().index,
        y=df[df["punt_position_initial"]==g3][column].value_counts(),
        name=g3
    )
    data = [trace1, trace2, trace3]
    layout = go.Layout(barmode='group')
    fig = go.Figure(data=data, layout=layout)
    offline.iplot(fig, config=config)
# EDA of concussion plays
video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
cols = ["Primary_Impact_Type", "Primary_Partner_GSISID", "Primary_Partner_Activity_Derived", "Friendly_Fire"]
for col in cols:  # change Unclear to NaN
    if col == "Primary_Partner_GSISID":
        video_review[col].iloc[33] = float("NaN")
    else:
        video_review[col].iloc[33] = str("NaN")
# video_review.to_csv("../output/video_review.csv", index=False)
video_review["PAD_PIT"] = [pad + "_" + pit.replace("Helmet-", "") for pad, pit in zip(video_review["Player_Activity_Derived"], video_review["Primary_Impact_Type"])]
# video_review.head(3)
plot_count_category(video_review, 'Player_Activity_Derived')
# Helmet-to-helmet and heltmet-to-body impacts result in the most coccussions
plot_count_category(video_review, 'Primary_Impact_Type')
plot_count_category(video_review, 'PAD_PIT')
files = []
gk_min = []
gk_max = []
for file in sorted(os.listdir("../input/NFL-Punt-Analytics-Competition/")):
    if "NGS" in file: 
        files += [file]
        gk = pd.read_csv('../input/NFL-Punt-Analytics-Competition/{}'.format(file)).GameKey
        gk_min += [np.min(gk)]
        gk_max += [np.max(gk)]
df_gamekey = pd.DataFrame({"filename": files, "gamekey_min": gk_min, "gamekey_max": gk_max}).sort_values("gamekey_min")
# df_gamekey.to_csv("../output/df_gamekey.csv", index=False)

# df_gamekey = pd.read_csv("../output/df_gamekey.csv")
# df_gamekey  # NGS-2017-reg-wk7-12.csv に 406 が混じってる
layout = load_layout()
# df_gamekey = pd.read_csv("../output/df_gamekey.csv")

def watch_video():
    role_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
    
    player_info = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
    grouped = player_info.groupby('GSISID').agg('count')
    grouped_player_info = player_info.groupby('GSISID').agg({
        'Number': lambda x: ','.join(x.replace(to_replace='[^0-9]', value='', regex=True).unique()), 
        'Position': lambda x: ','.join(x.unique())})
    
    video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv', na_values='Unclear', dtype={'Primary_Partner_GSISID': np.float64}) 
    video_footage_injury = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')

    videos = pd.merge(video_review, video_footage_injury, left_on=['PlayID','GameKey'], right_on=['playid','gamekey'])
    videos = pd.merge(videos, grouped_player_info, how='left', left_on='GSISID', right_on='GSISID')
    videos = pd.merge(videos, role_data, how='left', on=["Season_Year", "GameKey", "PlayID", "GSISID"])
    videos.rename({'Number': 'Injured Player Number(s)', 'Position': 'Injured Player Position'}, axis=1, inplace=True)
    videos = pd.merge(videos, grouped_player_info, how='left', left_on='Primary_Partner_GSISID', right_on='GSISID')
    videos.rename({'Number': 'Other Player Number(s)', 'Position': 'Other Player Position'}, axis=1, inplace=True)
    # Remove duplicate columns
    # videos.drop(['gamekey', 'playid', 'season'], axis=1, inplace=True)

    # Remove some columns that distract in manual viewing
    watch = videos.drop(['GameKey', 'PlayID', 'GSISID', 'Season_Year', 'Primary_Partner_GSISID', 'Type', 'Week', 'Qtr', 'PlayDescription', 'Home_team', 'Visit_Team'], axis=1)
    watch.rename({'Player_Activity_Derived': 'Injured Player Action', 'Primary_Partner_Activity_Derived': 'Other Player Action', 'Turnover_Related': 'From Turnover', 'Friendly_Fire': 'Friendly Fire'}, axis=1, inplace=True)
    watch['PREVIEW LINK (5000K)'] = watch['PREVIEW LINK (5000K)'].apply(lambda x: '<a href="{0}" target="__blank">video link</a>'.format(x))
    return watch

df_watch = watch_video()
pd.set_option('display.max_colwidth', -1)
HTML(df_watch.to_html(escape=False))
punt_coverage_positions = ["P", "PPR","PPRi", "PPRo", "PPL", "PPLi", "PPLo","PC", "PLW","PRW","GL", "GLi",
                           "GLo",  "GR",   "GRi",  "GRo","GR","PLT","PLG","PLS","PRG","PRT"]
punt_return_positions = ["PR", "PFB", "VL", "PDL1", "PDL2", "PDL3", "PDL4", "PDL5", "PDL6",
                         "PDR1", "PDR2", "PDR3", "PDR4", "PDR5", "PDR6","PLL", "PLL1", "PLL2", "PLL3",
                         "PLR","PLM1", "PLR1", "PLR2", "PLR3","PLM","VRi","VRo","VR","VLi", "VLo", "PDM"]
df_watch["punt_position_team"] = ["punt_coverage_positions" if role in punt_coverage_positions 
                              else "punt_return_positions" if role in punt_return_positions
                              else "NaN" for role in df_watch["Role"]]

front_positions = ["PLS", "PRG", "PRT", "PRW", "PLG", "PLT", "PLW", "PDR", "PDL", "PLR", "PLL", "PLM"]
back_positions = ["P", "PPR", "PPL", "PC", "PR", "PFB"]
side_positions = ["GL", "GR", "VL", "VRi", "VRo"]
df_watch["punt_position_initial"] = ["front_positions" if role in front_positions 
                                     else "back_positions" if role in back_positions
                                     else "side_positions" if role in side_positions
                                     else "NaN" for role in df_watch["Role"]]

df_watch["action_position"] = [action + "_" + position for action, position in zip(df_watch["Injured Player Action"], df_watch["Injured Player Position"])]
df_watch["action_role"] = [action + "_" + role for action, role in zip(df_watch["Injured Player Action"], df_watch["Role"])]
plot_count_category(df_watch, "Role")
plot_count_category(df_watch, "action_role")
plot_count_category(df_watch, "punt_position_team")
plot_count_category(df_watch, "punt_position_initial")
plot_punt_team(df_watch, "Role", "punt_coverage_positions", "punt_return_positions")
plot_punt_initial(df_watch, "Role", "front_positions", "back_positions", "side_positions")
def load_punt_data():
    player_info = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')

    grouped = player_info.groupby('GSISID').agg('count')
    grouped_player_info = player_info.groupby('GSISID').agg({
        'Number': lambda x: ','.join(x.replace(to_replace='[^0-9]', value='', regex=True).unique()), 
        'Position': lambda x: ','.join(x.unique())})
    return grouped_player_info

def load_role_data():
    return pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')

def load_plot(index, video=True):
    i = index
    # for i in range(len(video_review)):
    gk = int(video_review["GameKey"].iloc[i])
    for j in range(len(df_gamekey)):
        if gk <= int(df_gamekey["gamekey_max"].iloc[j]):
            filepath = df_gamekey.filename.iloc[j]
            break
    # Load the movements of players in GameKey 280. 
    df_game = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/{}'.format(filepath), GameKey=gk)
    df_game = get_speed(df_game)
    df_game = df_game.astype({"GSISID": int})
    grouped_player_info = load_punt_data().reset_index()
    role_data = load_role_data()
    df_game = df_game.merge(grouped_player_info, how='left', on="GSISID")
    df_game = df_game.merge(role_data, how='left', on=["Season_Year", "GameKey", "PlayID", "GSISID"])
    del grouped_player_info, role_data

    # Plot a single play, with two players
    print("Injured GSISID: {}, Injured Player Number: #{}, Concerned Player Number: #{}, Play Type: {}, Primary Impact: {}"
          .format(video_review["GSISID"].iloc[i],
                  df_watch["Injured Player Number(s)"].iloc[i],
                  df_watch["Other Player Number(s)"].iloc[i],
                  video_review["Player_Activity_Derived"].iloc[i], 
                  video_review["Primary_Impact_Type"].iloc[i]))
#     print("{} {}".format(df_game["Speed"].iloc[i]))
    
    plot_play(game_df=df_game, 
              PlayID=int(video_review["PlayID"].iloc[i]), 
              player1=int(video_review["GSISID"].iloc[i]), 
              player2=float(video_review["Primary_Partner_GSISID"].iloc[i]))

    if video == False:
        return reduce_mem_usage(df_game, verbose=False)

    # Check video
    if video == True:
        preview_link = df_watch["PREVIEW LINK (5000K)"].iloc[i]
        pl_start = preview_link.find('http')
        pl_end = preview_link.find('mp4') + 3

        video_url = preview_link[pl_start:pl_end]
        return HTML('<video width="560" height="315" controls> <source src="{}" type="video/mp4"></video>'.format(video_url))
for i in range(3):
    load_plot(i)
video_review_detail = pd.read_csv("../input/kaggle-nfl-tak/kaggle-nfl-video-review.csv", header=1, nrows=37)
video_review_detail
plot_count_category(video_review_detail, "Tackle Correct?")
video_review_detail["pt_tc"] = [pt + "_" + tc for pt, tc in zip(
                                video_review_detail["Play Type"].fillna("NaN"), 
                                video_review_detail["Tackle Correct?"].fillna("NaN"))]

plot_count_category(video_review_detail, "pt_tc")
