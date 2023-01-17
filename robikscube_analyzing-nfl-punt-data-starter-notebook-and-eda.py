import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import os
plt.style.use('ggplot')

# For interactive plots
from plotly import offline
import plotly.graph_objs as go


pd.set_option('max.columns', None)
offline.init_notebook_mode()
config = dict(showLink=False)
# Read the input data
ppd = pd.read_csv('../input/player_punt_data.csv')
gd = pd.read_csv('../input/game_data.csv')
pprd = pd.read_csv('../input/play_player_role_data.csv')
vr = pd.read_csv('../input/video_review.csv')
vfi = pd.read_csv('../input/video_footage-injury.csv')
pi = pd.read_csv('../input/play_information.csv')
gd.head()
gd.plot(kind='scatter', x='Week', y='Temperature', figsize=(15, 5), title='NFL Game Data Week vs. Temperature')
plt.show()
gd['count'] = 1
gd.groupby('Turf') \
    .count()[['count']] \
    .sort_values('count', ascending=False) \
    .plot(kind='bar', figsize=(15, 5), rot=85, title='Count of Games by Turf Type')
plt.show()
gd.groupby('Start_Time') \
    .count()[['count']] \
    .plot(kind='bar', figsize=(15, 5), rot=85, title='Count of Games by Start Time', color='g')
plt.show()
vr.shape
vr.head()
vr['count'] = 1
vr.groupby('Player_Activity_Derived') \
    .count()[['count']] \
    .sort_values('count', ascending=False) \
    .plot(kind='barh', figsize=(15, 5), title='Count of Player Activity Derived')

plt.show()
vr['count'] = 1
vr.groupby('Primary_Partner_Activity_Derived') \
    .count()[['count']] \
    .sort_values('count', ascending=False) \
    .plot(kind='barh', figsize=(15, 5), title='Count of Primary Partner Activity Derived', color='g')

plt.show()
vr['count'] = 1
vr.groupby('Primary_Impact_Type') \
    .count()[['count']] \
    .sort_values('count', ascending=False) \
    .plot(kind='barh', figsize=(15, 5), title='Count of Primary Impact Type', color='b')
plt.show()
pi.head()
pi['count'] = 1
pi.groupby('Poss_Team').count()[['count']] \
    .sort_values('count', ascending=False) \
    .plot(kind='bar', figsize=(15, 5), title='Count of punts per team', color='k')
plt.show()
# They are all punts!!! :D
pi['Play_Type'].unique()
# Loading and plotting functions

def load_plays_for_game(GameKey):
    """
    Returns a dataframe of play data for a given game (GameKey)
    """
    play_information = pd.read_csv('../input/play_information.csv')
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
# Load the movements of players in GameKey 280. 
game280 = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=280)
# Plot a single play, with two players
plot_play(game_df=game280, PlayID=2918, player1=32864, player2=32725)
def plot_play_all_players(game_df, PlayID, custom_layout=False):
    """
    Plots player movements on the field for a given game, play, and two players
    """
    game_df = game_df[game_df.PlayID==PlayID]
    
    GameKey=str(pd.unique(game_df.GameKey)[0])
    HomeTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[0]
    VisitingTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[1]
    player1 = game_df[(game_df.PlayID==PlayID)]['GSISID'].values[0]
    YardLine = game_df[(game_df.PlayID==PlayID) & (game_df.GSISID==player1)]['YardLine'].iloc[0]
    
    traces=[]   
    for player in pd.unique(game_df.GSISID):
        player = int(player)
        trace = go.Scatter(
            x = game_df[game_df.GSISID==player].x,
            y = game_df[game_df.GSISID==player].y,
            name='GSISID '+str(player),
            mode='markers'
        )
        traces.append(trace)
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
plot_play_all_players(game_df=game280, PlayID=2918)
pprd.head()

# inputs
custom_layout = False
game_df=game280
PlayID=2918


# Function code
game_df = game_df[game_df.PlayID==PlayID]

GameKey=str(pd.unique(game_df.GameKey)[0])
HomeTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[0]
VisitingTeam = pd.unique(game_df.Home_Team_Visit_Team)[0].split("-")[1]
player1 = game_df[(game_df.PlayID==PlayID)]['GSISID'].values[0]
YardLine = game_df[(game_df.PlayID==PlayID) & (game_df.GSISID==player1)]['YardLine'].iloc[0]

traces=[]   
for player in pd.unique(game_df.GSISID):
    player = int(player)
    trace = go.Scatter(
        x = game_df[game_df.GSISID==player].x,
        y = game_df[game_df.GSISID==player].y,
        name='GSISID '+str(player),
        mode='markers'
    )
    traces.append(trace)
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
pprd.shape
pprd.head()
