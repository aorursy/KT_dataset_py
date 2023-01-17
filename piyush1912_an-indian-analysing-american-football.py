# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
game_data = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')

game_data.head()
import random
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
game_data['Temperature']= game_data['Temperature'].fillna(game_data['Temperature'].mean())

game_data.head()
from datetime import datetime
import pandas_datareader.data as web


data = [go.Scatter(x=game_data.Game_Date, y=game_data.Temperature)]

py.iplot(data)
pitch_count = game_data['Game_Site'].value_counts()
data = [go.Bar(
    x = pitch_count.index,
    y = pitch_count.values,
    marker = dict(color = random_colors(25))
)]
layout = dict(
         title= "Most Occurring Game Venue "
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
day_count = game_data['Game_Day'].value_counts()
trace = go.Pie(labels=day_count.index, values=day_count.values, hole=0.6)
layout = go.Layout(
    title='Game Day Percentage'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
pitch_count = game_data['StadiumType'].value_counts()
data = [go.Bar(
    x = pitch_count.index,
    y = pitch_count.values,
    marker = dict(color = random_colors(25))
)]
layout = dict(
         title= "Different types of Stadium"
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
pitch_count = game_data['GameWeather'].value_counts()
data = [go.Bar(
    x = pitch_count.index,
    y = pitch_count.values,
    marker = dict(color = random_colors(25))
)]
layout = dict(
         title= "Different types of Weathers"
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
pitch_count = game_data['Turf'].value_counts()
data = [go.Bar(
    x = pitch_count.index,
    y = pitch_count.values,
    marker = dict(color = random_colors(25))
)]
layout = dict(
         title= "Different types of Turfs"
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
video_review= pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.head()
NGS= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-post.csv')
NGS.head()
NGS = NGS.fillna(0)
NGS1= NGS.truncate(before=1, after=1000)
import base64
with open("../input/picture2/61jq6q0FyJL._SY606_.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
#add the prefix that plotly will want when using the string as source
encoded_image = "data:image/png;base64," + encoded_string
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
NGS_pre= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
NGS_pre.head()
NGS_pre1= NGS_pre.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_pre1.x,y=NGS_pre1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_pre, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game5 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=5)
game21 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=21)
game29 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=29)
game45 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=45)
game54 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=54)
game60 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',GameKey=60)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 3129, 5, 31057, 32482))
plot_play(game_df=game5, PlayID=3129, player1=31057, player2=32482 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 2587, 21, 29343, 31059))
plot_play(game_df=game21, PlayID=2587, player1=29343, player2=31059 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 538, 29, 31023, 31941))
plot_play(game_df=game29, PlayID=538, player1=31023, player2=31941 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 1212, 45, 33121, 28429))
plot_play(game_df=game45, PlayID=1212, player1=33121, player2=28429 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[4]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[4]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[4]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[4]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 1045, 54, 32444, 31756))
plot_play(game_df=game54, PlayID=1045, player1=32444, player2=31756 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[5]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[5]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[5]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[5]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv', 905, 60, 30786, 29815))
plot_play(game_df=game60, PlayID=905, player1=30786, player2=29815 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game149 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv',GameKey=149)
game144 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv',GameKey=144)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv', 2342, 144, 32410, 23259))
plot_play(game_df=game144, PlayID=2342, player1=32410, player2=23259 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv', 3663, 149, 28128, 29629))
plot_play(game_df=game149, PlayID=3663, player1=28128, player2=29629 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game189 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',GameKey=189)
game218 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',GameKey=218)
game231 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',GameKey=231)
game234 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',GameKey=234)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv', 3509, 189, 27595, 31950))
plot_play(game_df=game189, PlayID=3509, player1=27595, player2=31950 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv', 3468, 218, 28987, 31950))
plot_play(game_df=game218, PlayID=3468, player1=28987, player2=31950 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv', 1976, 231, 32214, 32807))
plot_play(game_df=game231, PlayID=1976, player1=32214, player2=32807 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv', 3278, 234, 28620, 27860))
plot_play(game_df=game234, PlayID=3278, player1=28620, player2=27860 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',low_memory=False)
NGS_reg.head()
NGS_reg= NGS_reg.fillna(0)
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game266 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=266)
game274 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=274)
game280 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=280)
game281 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=281)
game289 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=289)
game296 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',GameKey=296)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 2902, 266, 23564, 31844))
plot_play(game_df=game266, PlayID=2902, player1=23564, player2=31844 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 3609, 274, 23742, 31785))
plot_play(game_df=game274, PlayID=3609, player1=23742, player2=31785 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 2918, 280, 32120, 32725))
plot_play(game_df=game280, PlayID=2918, player1=32120, player2=32725 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 1526, 281, 28987, 30789))
plot_play(game_df=game281, PlayID=1526, player1=28987, player2=30789 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[4]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[4]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[4]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[4]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 2341, 289, 32007, 32998))
plot_play(game_df=game289, PlayID=2341, player1=32007, player2=32998 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[5]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[5]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[5]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[5]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv', 2667, 296, 32783, 32810))
plot_play(game_df=game296, PlayID=2667, player1=32783, player2=32810 )
NGS_post= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-post.csv')
NGS_post.head()
NGS_post1= NGS_post.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_post1.x,y=NGS_post1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_post, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1
NGS_pre= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv')
NGS_pre.head()
NGS_pre1= NGS_pre.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_pre1.x,y=NGS_pre1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_pre, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game357 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',GameKey=357)
game364 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',GameKey=364)
game384 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',GameKey=384)
game392 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',GameKey=392)
game397 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',GameKey=397)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv', 3630, 357, 30171, 29384))
plot_play(game_df=game357, PlayID=3630, player1=30171, player2=29384 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv', 2764, 364, 32323, 31930))
plot_play(game_df=game364, PlayID=2764, player1=32323, player2=31930 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv', 183, 384, 33813, 33841))
plot_play(game_df=game384, PlayID=183, player1=33813, player2=33841 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv', 1088, 392, 32615, 31999))
plot_play(game_df=game392, PlayID=1088, player1=32615, player2=31999 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[4]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[4]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[4]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[4]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv', 1526, 397, 32894, 31763))
plot_play(game_df=game397, PlayID=1526, player1=32894, player2=31763 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game399 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',GameKey=399)
game414 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',GameKey=414)
game448 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',GameKey=448)
game473 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',GameKey=473)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv', 3312, 399, 26035, 27442))
plot_play(game_df=game399, PlayID=3312, player1=26035, player2=27442 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv', 1262, 414, 33941, 27442))
plot_play(game_df=game414, PlayID=1262, player1=33941, player2=27442 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv', 2792, 448, 33838, 31317))
plot_play(game_df=game448, PlayID=2792, player1=33838, player2=31317 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv', 2072, 473, 29492, 33445))
plot_play(game_df=game473, PlayID=2072, player1=29492, player2=33445 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game506 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv',GameKey=506)
game553 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv',GameKey=553)
game567 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv',GameKey=567)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv', 1988, 506, 27060, 30780))
plot_play(game_df=game506, PlayID=1988, player1=27060, player2=30780 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv', 1683, 553, 32820, 25503))
plot_play(game_df=game553, PlayID=1683, player1=32820, player2=25503 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv', 1407, 567, 32403, 32891))
plot_play(game_df=game567, PlayID=1407, player1=32403, player2=32891 )
NGS_reg= pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv')
NGS_reg.head()
NGS_reg1= NGS_reg.truncate(before=1, after=1000)
trace1= go.Scatter(x=NGS_reg1.x,y=NGS_reg1.y)

fig=go.Figure(data=[trace1],layout=layout)
py.iplot(fig)
video_review1 = pd.merge(video_review,NGS_reg, on=['Season_Year','GameKey','PlayID','GSISID'])
video_review1= video_review1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
game585 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',GameKey=585)
game601 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',GameKey=601)
game607 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',GameKey=607)
game618 = load_game_and_ngs('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',GameKey=618)
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[0]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[0]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[0]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[0]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv', 2208, 585, 33069, 29414))
plot_play(game_df=game585, PlayID=2208, player1=33069, player2=29414 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[1]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[1]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[1]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[1]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv', 602, 601, 33260, 31697))
plot_play(game_df=game601, PlayID=602, player1=33260, player2=31697 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[2]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[2]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[2]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[2]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv', 978, 607, 29793, 32114))
plot_play(game_df=game607, PlayID=978, player1=29793, player2=32114 )
# Plot a single play, with two players
print('Primary Impact:',video_review1.iloc[3]["Primary_Impact_Type"]) 
print('Primary Activity:',video_review1.iloc[3]["Player_Activity_Derived"] )
print('Partners Activity:',video_review1.iloc[3]["Primary_Partner_Activity_Derived"] )
print('Players from same team :',video_review1.iloc[3]["Friendly_Fire"] )
print(get_speed('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv', 2792, 618, 31950, 326))
plot_play(game_df=game618, PlayID=2792, player1=31950, player2=32677 )
play_info= pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
play_info.head()
play_info1 = pd.merge(video_review,play_info, on=['Season_Year','GameKey','PlayID'])
play_info1.sort_values('PlayID', ascending=False).drop_duplicates('GameKey').sort_index()
punt_count = play_info1['Quarter'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Most concussion Quarter'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
punt_count = play_info1['Poss_Team'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Team percentage Punt play which lead to injury'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
punt_data= pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
punt_data.head()
punt_data1 = pd.merge(video_review,punt_data, on=['GSISID'])
punt_data1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
punt_count = punt_data1['Position'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Concussion Position Percentage'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
player_role= pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
player_role.head()
player_role1 = pd.merge(video_review,player_role, on=['Season_Year','GameKey','PlayID','GSISID'])
player_role1.sort_values('GSISID', ascending=False).drop_duplicates('GameKey').sort_index()
role_count = player_role1['Role'].value_counts()
data = [go.Bar(
    x = role_count.index,
    y = role_count.values,
    marker = dict(color = random_colors(100))
)]
layout = dict(
         title= "Role count leading to Injury"
)
fig = dict( data=data, layout=layout )
iplot( fig, validate=False )
punt_count = player_role1['Role'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Player role percentage'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
video_footage_control= pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-control.csv')
video_footage_control.head()
video_footage_control.rename(columns={'season': 'Season_Year', 'gamekey': 'GameKey', 'playid': 'PlayID'}, inplace=True)
player_role1 = pd.merge(video_review,video_footage_control, on=['GameKey','PlayID','Season_Year'])
player_role1.sort_values('PlayID', ascending=False).drop_duplicates('GameKey').sort_index()
video_footage_injury= pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
video_footage_injury.head()
video_footage_injury.rename(columns={'season': 'Season_Year', 'gamekey': 'GameKey', 'playid': 'PlayID'}, inplace=True)
player_role1 = pd.merge(video_review,video_footage_injury, on=['GameKey','PlayID','Season_Year'])
player_role1.sort_values('PlayID', ascending=False).drop_duplicates('GameKey').sort_index()
punt_count = player_role1['Home_team'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Home team injury involvement percentage'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
punt_count = player_role1['Visit_Team'].value_counts()
trace = go.Pie(labels=punt_count.index, values=punt_count.values, hole=0.6)
layout = go.Layout(
    title='Visit team injury involvement percentage'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
trace0 = go.Scatter(
    x=player_role1.Home_team,
    y=player_role1.Player_Activity_Derived,
    mode='markers',
    marker=dict(
        color=random_colors(50)

    )
)
layout = dict(
            title='Home team players activity leading to concussion '
)
fig = go.Figure(data=[trace0], layout=layout)
py.iplot(fig, filename='bubblechart-color')
trace0 = go.Scatter(
    x=player_role1.Visit_Team,
    y=player_role1.Player_Activity_Derived,
    mode='markers',
    marker=dict(
        color=random_colors(50)

    )
)
layout = dict(
            title='Away team players activity leading to concussion '
)
fig = go.Figure(data=[trace0], layout=layout)
py.iplot(fig, filename='bubblechart-color')

video_review= video_review.fillna(0)

video_review= video_review[['Player_Activity_Derived','Turnover_Related','Primary_Impact_Type','Primary_Partner_Activity_Derived','Friendly_Fire']]
s = pd.crosstab(video_review['Primary_Impact_Type'],
                video_review['Friendly_Fire'] ,normalize='index').style.background_gradient(cmap='RdBu', low=.1, high=0).highlight_null('red')
s
s = pd.crosstab(video_review['Player_Activity_Derived'],
                video_review['Primary_Partner_Activity_Derived'] ,normalize='index').style.background_gradient(cmap='RdBu', low=.1, high=0).highlight_null('red')
s
video_review_count = video_review['Primary_Impact_Type'].value_counts()
trace = go.Pie(labels=video_review_count.index, values=video_review_count.values, hole=0.6)
layout = go.Layout(
    title='Most Impact Type'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
video_review_count = video_review['Player_Activity_Derived'].value_counts()
trace = go.Pie(labels=video_review_count.index, values=video_review_count.values, hole=0.6)
layout = go.Layout(
    title='Most activity percentage leading to concussion injury'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")
serious_tackle= video_review.groupby(['Primary_Partner_Activity_Derived', 'Friendly_Fire'])
partner_count = serious_tackle['Primary_Partner_Activity_Derived'].value_counts()
trace = go.Pie(labels=partner_count.index, values=partner_count.values, hole=0.6)
layout = go.Layout(
    title='Percentage of partners serious play'
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename="plotting-library")