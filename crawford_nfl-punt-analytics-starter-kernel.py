import pandas as pd
import glob
from plotly import offline
import plotly.graph_objs as go


pd.set_option('max.columns', None)
offline.init_notebook_mode()
config = dict(showLink=False)

# EDA of concussion plays
video_review = pd.read_csv('../input/video_review.csv')
video_review.head()
def plot_count_category(df, column):
    x = df[column].value_counts().index
    y = df[column].value_counts()
    trace = go.Bar(
        x=x,
        y=y
    )
    data = [trace]
    offline.iplot(data, config=config)
# Players involved in Tackling seem to have the brunt of concussions. I'm shocked.
plot_count_category(video_review, 'Player_Activity_Derived')
# Helmet-to-helmet and heltmet-to-body impacts result in the most coccussions
plot_count_category(video_review, 'Primary_Impact_Type')
# Which games have multiple plays with concussions?
video_review[video_review.duplicated(['GameKey'], keep=False)]
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
    

# Load the movements of players in GameKey 280. 
game280 = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=280)
# Plot a single play, with two players
plot_play(game_df=game280, PlayID=2918, player1=32120, player2=32725)
plot_play(game280,PlayID=3746, player1=27654, player2=33127)