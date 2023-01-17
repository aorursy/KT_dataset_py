# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
import os
print(os.listdir("../input"))
game_data = pd.read_csv('../input/game_data.csv')
game_data.head()
import random
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
game_data['Temperature']= game_data['Temperature'].fillna(game_data['Temperature'].mean())
game_data.head()

video_review= pd.read_csv('../input/video_review.csv')
video_review.head()
NGS= pd.read_csv('../input/NGS-2016-post.csv')
NGS.head()
NGS = NGS.fillna(0)
NGS.head()
NGS1= NGS.truncate(before=1, after=1000)
def convert_to_mph(dis, converter):
    mph = dis * converter
    return mph
def get_speed(ng_data, playId, gameKey, player, partner):
    ng_data = pd.read_csv(ng_data,low_memory=False)
    ng_data['mph'] = convert_to_mph(ng_data['dis'], 20.455)
    player_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                               & (ng_data.GSISID == player)].sort_values('Time')
    partner_data = ng_data.loc[(ng_data.GameKey == gameKey) & (ng_data.PlayID == playId) 
                              & (ng_data.GSISID == partner)].sort_values('Time')
    player_grouped = player_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    player_grouped['Player_Involved'] = 'player_injured'
    partner_grouped = partner_data.groupby(['GameKey','PlayID','GSISID'], 
                               as_index = False)['mph'].agg({'max_mph': max,
                                                             'avg_mph': np.mean
                                                            })
    partner_grouped['Player_Involved'] = 'primary_partner'
    return pd.concat([player_grouped, partner_grouped], axis = 0)[['Player_Involved',
                                                                   'max_mph',
                                                                   'avg_mph']].reset_index(drop=True)
def load_layout():
    """
    Returns a dict for a Football themed Plot.ly layout 
    """
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

layout = load_layout()
Player_position= go.Scatter(x=NGS1.x,y=NGS1.y)

fig=go.Figure(data=[Player_position],layout=layout)
py.iplot(fig)



# Loading and plotting functions

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
    
video_review1 = pd.merge(video_review,NGS, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1
import glob
from plotly import offline
import plotly.graph_objs as go


pd.set_option('max.columns', None)
offline.init_notebook_mode()
config = dict(showLink=False)
NGS_pre= pd.read_csv('../input/NGS-2016-pre.csv')
NGS_pre.head()
NGS_pre1= NGS_pre.truncate(before=1, after=1000)

trace1= go.Scatter(x=NGS_pre1.x,y=NGS_pre1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_pre, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-pre.csv', 3129, 5, 31057, 32482))

# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-pre.csv', 2587, 21, 29343, 31059))

# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-pre.csv', 538, 29, 31023, 31941))
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-pre.csv', 1212, 45, 33121, 28429))
print('Primary Impact:',video_review1.iloc[5]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[5]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[5]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[5]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-pre.csv', 905, 60, 30786, 29815))
NGS_reg= pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review= pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-reg-wk1-6.csv', 2342, 144, 32410, 23259))
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NGS-2016-reg-wk1-6.csv', 3663, 149, 28128, 29629))
NGS_reg= pd.read_csv('../input/NGS-2016-reg-wk7-12.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
NGS_pre1= NGS_pre.truncate(before=1, after=1000)
NGS_pre.head()
NGS_pre1= NGS_pre.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_pre1.x,y=NGS_pre1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
