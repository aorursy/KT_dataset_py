import pandas as pd
import glob
from plotly import offline
import plotly.graph_objs as go


pd.set_option('max.columns', None)
offline.init_notebook_mode()
config = dict(showLink=False)

video_review = pd.read_csv('../input/video_review.csv')
video_review.head()
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
                text = game_df[game_df.GSISID==player].Time,
                hoverinfo = 'text',
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
    
time_data = ['2016-08-12 02:07:47:600','2016-08-20 21:04:15:800','2016-08-19 23:53:46.400','2016-08-28 00:28:47.600','2016-09-02 00:33:39.100',
            '2016-09-02 00.13.24.200','2016-10-16 18:46:54.300','2016-10-16 19:38:01.400','2016-11-06 21:00:52.800','2016-11-20 20:41:57.200',
            '2016-11-27 19:21:55.800','2016-11-27 20:32:16.000','2016-12-11 20:30:37.200','2016-12-16 04:19:02.700','2016-12-18 20:20:51.500',
            '2016-12-18 20.55.49.100','2016-12-18 19:16:08.700','2016-12-20 03:19:55.100','2016-12-24 19:59:49.300','2017-08-18 01:39:15.000',
            '2017-08-20 01:29:19.200','2017-08-20 01:42:36.800','2017-08-31 23:10:42.500','2017-09-01 01:24:07.800','2017-09-01 03:10:23.800',
            '2017-09-08 03:10:57.300','','2017-10-01 15:39:58.200','2017-10-08 21:48:30.500','','2017-11-19 19:16:21.400','2017-11-26 18:59:23.700',
            '2017-12-03 19:51:17.000','','2017-12-10 21:27:21.400','2017-12-15 02:03:15.800','2017-12-17 23:13:36.300']
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][0]) #2016-08-12 02:07:47:600
plot_play(game_df=game, PlayID=video_review['PlayID'][0], player1=float(video_review['GSISID'][0]), player2=float(video_review['Primary_Partner_GSISID'][0]))
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][1]) #2016-08-20 21:04:15:800
plot_play(game_df=game, PlayID=video_review['PlayID'][1], player1=float(video_review['GSISID'][1]), player2=float(video_review['Primary_Partner_GSISID'][1]))
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][2]) #2016-08-19 23:53:46.400
plot_play(game_df=game, PlayID=video_review['PlayID'][2], player1=float(video_review['GSISID'][2]), player2=float(video_review['Primary_Partner_GSISID'][2]))
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][3]) #2016-08-28 00:28:47.600
plot_play(game_df=game, PlayID=video_review['PlayID'][3], player1=float(video_review['GSISID'][3]), player2=float(video_review['Primary_Partner_GSISID'][3]))
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][4]) #2016-09-02 00:33:39.100
plot_play(game_df=game, PlayID=video_review['PlayID'][4], player1=float(video_review['GSISID'][4]), player2=float(video_review['Primary_Partner_GSISID'][4]))
game = load_game_and_ngs('../input/NGS-2016-pre.csv',GameKey=video_review['GameKey'][5]) #2016-09-02 00.13.24.200
plot_play(game_df=game, PlayID=video_review['PlayID'][5], player1=float(video_review['GSISID'][5]), player2=float(video_review['Primary_Partner_GSISID'][5]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk1-6.csv',GameKey=video_review['GameKey'][6]) #2016-10-16 18:46:54.300
plot_play(game_df=game, PlayID=video_review['PlayID'][6], player1=float(video_review['GSISID'][6]), player2=float(video_review['Primary_Partner_GSISID'][6]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk1-6.csv',GameKey=video_review['GameKey'][7]) #2016-10-16 19:38:01.400
plot_play(game_df=game, PlayID=video_review['PlayID'][7], player1=float(video_review['GSISID'][7]), player2=float(video_review['Primary_Partner_GSISID'][7]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk7-12.csv',GameKey=video_review['GameKey'][8]) #2016-11-06 21:00:52.800
plot_play(game_df=game, PlayID=video_review['PlayID'][8], player1=float(video_review['GSISID'][8]), player2=float(video_review['Primary_Partner_GSISID'][8]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk7-12.csv',GameKey=video_review['GameKey'][9]) #2016-11-20 20:41:57.200
plot_play(game_df=game, PlayID=video_review['PlayID'][9], player1=float(video_review['GSISID'][9]), player2=float(video_review['Primary_Partner_GSISID'][9]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk7-12.csv',GameKey=video_review['GameKey'][10]) #2016-11-27 19:21:55.800
plot_play(game_df=game, PlayID=video_review['PlayID'][10], player1=float(video_review['GSISID'][10]), player2=float(video_review['Primary_Partner_GSISID'][10]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk7-12.csv',GameKey=video_review['GameKey'][11]) #2016-11-27 20:32:16.000
plot_play(game_df=game, PlayID=video_review['PlayID'][11], player1=float(video_review['GSISID'][11]), player2=float(video_review['Primary_Partner_GSISID'][11]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][12]) #2016-12-11 20:30:37.200
plot_play(game_df=game, PlayID=video_review['PlayID'][12], player1=float(video_review['GSISID'][12]), player2=float(video_review['Primary_Partner_GSISID'][12]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][13]) #2016-12-16 04:19:02.700
plot_play(game_df=game, PlayID=video_review['PlayID'][13], player1=float(video_review['GSISID'][13]), player2=float(video_review['Primary_Partner_GSISID'][13]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][14]) #2016-12-18 20:20:51.500
plot_play(game_df=game, PlayID=video_review['PlayID'][14], player1=float(video_review['GSISID'][14]), player2=float(video_review['Primary_Partner_GSISID'][14]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][15]) #2016-12-18 20.55.49.100
plot_play(game_df=game, PlayID=video_review['PlayID'][15], player1=float(video_review['GSISID'][15]), player2=float(video_review['Primary_Partner_GSISID'][15]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][16])#2016-12-18 19:16:08.700
plot_play(game_df=game, PlayID=video_review['PlayID'][16], player1=float(video_review['GSISID'][16]), player2=float(video_review['Primary_Partner_GSISID'][16]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][17])#2016-12-20 03:19:55.100
plot_play(game_df=game, PlayID=video_review['PlayID'][17], player1=float(video_review['GSISID'][17]), player2=float(video_review['Primary_Partner_GSISID'][17]))
game = load_game_and_ngs('../input/NGS-2016-reg-wk13-17.csv',GameKey=video_review['GameKey'][18]) #2016-12-24 19:59:49.300
plot_play(game_df=game, PlayID=video_review['PlayID'][18], player1=float(video_review['GSISID'][18]), player2=float(video_review['Primary_Partner_GSISID'][18]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][19]) #2017-08-18 01:39:15.000
plot_play(game_df=game, PlayID=video_review['PlayID'][19], player1=float(video_review['GSISID'][19]), player2=float(video_review['Primary_Partner_GSISID'][19]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][20])# 2017-08-20 01:29:19.200
plot_play(game_df=game, PlayID=video_review['PlayID'][20], player1=float(video_review['GSISID'][20]), player2=float(video_review['Primary_Partner_GSISID'][20]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][21]) #2017-08-20 01:42:36.800
plot_play(game_df=game, PlayID=video_review['PlayID'][21], player1=float(video_review['GSISID'][21]), player2=float(video_review['Primary_Partner_GSISID'][21]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][22]) #2017-08-31 23:10:42.500
plot_play(game_df=game, PlayID=video_review['PlayID'][22], player1=float(video_review['GSISID'][22]), player2=float(video_review['Primary_Partner_GSISID'][22]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][23]) #2017-09-01 01:24:07.800
plot_play(game_df=game, PlayID=video_review['PlayID'][23], player1=float(video_review['GSISID'][23]), player2=float(video_review['Primary_Partner_GSISID'][23]))
game = load_game_and_ngs('../input/NGS-2017-pre.csv',GameKey=video_review['GameKey'][24]) #2017-09-01 03:10:23.800
plot_play(game_df=game, PlayID=video_review['PlayID'][24], player1=float(video_review['GSISID'][24]), player2=float(video_review['Primary_Partner_GSISID'][24]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk1-6.csv',GameKey=video_review['GameKey'][25]) #2017-09-08 03:10:57.300
plot_play(game_df=game, PlayID=video_review['PlayID'][25], player1=float(video_review['GSISID'][25]), player2=float(video_review['Primary_Partner_GSISID'][25]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk1-6.csv',GameKey=video_review['GameKey'][26]) 
plot_play(game_df=game, PlayID=video_review['PlayID'][26], player1=float(video_review['GSISID'][26]), player2=float(video_review['Primary_Partner_GSISID'][26]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk1-6.csv',GameKey=video_review['GameKey'][27]) #2017-10-01 15:39:58.200
plot_play(game_df=game, PlayID=video_review['PlayID'][27], player1=float(video_review['GSISID'][27]), player2=float(video_review['Primary_Partner_GSISID'][27]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk1-6.csv',GameKey=video_review['GameKey'][28])  #2017-10-08 21:48:30.500
plot_play(game_df=game, PlayID=video_review['PlayID'][28], player1=float(video_review['GSISID'][28]), player2=float(video_review['Primary_Partner_GSISID'][28]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk7-12.csv',GameKey=video_review['GameKey'][29])  
plot_play(game_df=game, PlayID=video_review['PlayID'][29], player1=float(video_review['GSISID'][29]), player2=float(video_review['Primary_Partner_GSISID'][29]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk7-12.csv',GameKey=video_review['GameKey'][30])  #2017-11-19 19:16:21.400
plot_play(game_df=game, PlayID=video_review['PlayID'][30], player1=float(video_review['GSISID'][30]), player2=float(video_review['Primary_Partner_GSISID'][30]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk7-12.csv',GameKey=video_review['GameKey'][31])  #2017-11-26 18:59:23.700
plot_play(game_df=game, PlayID=video_review['PlayID'][31], player1=float(video_review['GSISID'][31]), player2=float(video_review['Primary_Partner_GSISID'][31]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk13-17.csv',GameKey=video_review['GameKey'][32])  #2017-12-03 19:51:17.000
plot_play(game_df=game, PlayID=video_review['PlayID'][32], player1=float(video_review['GSISID'][32]), player2=float(video_review['Primary_Partner_GSISID'][32]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk13-17.csv',GameKey=video_review['GameKey'][33])  
plot_play(game_df=game, PlayID=video_review['PlayID'][33], player1=float(video_review['GSISID'][33]), player2=0)
game = load_game_and_ngs('../input/NGS-2017-reg-wk13-17.csv',GameKey=video_review['GameKey'][34])  #2017-12-10 21:27:21.400
plot_play(game_df=game, PlayID=video_review['PlayID'][34], player1=float(video_review['GSISID'][34]), player2=float(video_review['Primary_Partner_GSISID'][34]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk13-17.csv',GameKey=video_review['GameKey'][35])  #2017-12-15 02:03:15.800
plot_play(game_df=game, PlayID=video_review['PlayID'][35], player1=float(video_review['GSISID'][35]), player2=float(video_review['Primary_Partner_GSISID'][35]))
game = load_game_and_ngs('../input/NGS-2017-reg-wk13-17.csv',GameKey=video_review['GameKey'][36])   #2017-12-17 23:13:36.300
plot_play(game_df=game, PlayID=video_review['PlayID'][36], player1=float(video_review['GSISID'][36]), player2=float(video_review['Primary_Partner_GSISID'][36]))