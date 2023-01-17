# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
import gc
import feather


video_review = pd.read_csv('../input/video_review.csv')

# Any results you write to the current directory are saved as output.
game_review = pd.read_csv('../input/game_data.csv')
player_role = pd.read_csv('../input/play_player_role_data.csv')
# Player activity during primary injury causing event
temp = video_review["Player_Activity_Derived"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Player-Activity during primary injury causing event",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Player activity",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')

temp = video_review["Primary_Impact_Type"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Impacting source that caused the concussion",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Primary Impact type",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = video_review["Primary_Partner_Activity_Derived"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Primary Partner Activity that caused the concussion",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Primary Partner Activity type",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
fig = plt.figure(figsize = (20,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
df = pd.DataFrame(data=video_review)
sns.heatmap(pd.crosstab(df.Player_Activity_Derived, df.Primary_Partner_Activity_Derived), annot=True, square=True, ax=ax1)
sns.heatmap(pd.crosstab(df.Player_Activity_Derived, df.Primary_Impact_Type), annot=True, square=True, ax=ax2)
sns.heatmap(pd.crosstab(df.Player_Activity_Derived, df.Friendly_Fire), annot=True, square=True, ax=ax3)
sns.heatmap(pd.crosstab(df.Primary_Impact_Type, df.Friendly_Fire), annot=True, square=True, ax=ax4)
merged_csv = pd.merge (video_review,player_role)
player_punt_data = pd.read_csv('../input/player_punt_data.csv')
player_punt_data = player_punt_data.sort_values('GSISID', ascending=False).drop_duplicates('GSISID').sort_index()
merged_csv = pd.merge (merged_csv ,player_punt_data[['GSISID','Position']])
del video_review,player_role,player_punt_data
gc.collect()
temp = merged_csv["Role"].value_counts()
tempe = merged_csv["Position"].value_counts()
fig = {
    "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "hole": .4,
      "type": "pie"
    },
      {
      "values": tempe.values,
      "labels": tempe.index,
      "domain": {"x": [.52, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    
    ],
  "layout": {
        "title":"Player Roles and Punt Positions",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Punt Role",
                 "x": 0.19,
                "y": 0.5
            },
              {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Position",
                "x": 0.80,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
#temp = video_review["Primary_Partner_Activity_Derived"].value_counts()
temp = merged_csv.groupby(['Role','Position']).size()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Player Punt Role and Position Combination",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Position and Punt Role Combined",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
final_csv = pd.merge (merged_csv,game_review)
play_information = pd.read_csv('../input/play_information.csv')
final_csv = pd.merge (final_csv ,play_information[['GameKey','Quarter','PlayID','Poss_Team','Score_Home_Visiting']],on=['GameKey','PlayID'],how='left')
final_csv['home_poss'] = np.where(final_csv['HomeTeamCode'] == final_csv['Poss_Team'], 'Yes', 'No')  
score_away = []
score_home = []
home_win_loss = []

for item in final_csv['Score_Home_Visiting']:
    scores = item.split('-')
    temp =  int(scores[1])
    temp2 = int(scores[0])
    if(temp<temp2):
        temp3 = "Winning"
    elif(temp==temp2):
        temp3 = "Draw"
    else:
        temp3 = "Losing"
    score_away.append(temp)
    score_home.append(temp2)
    home_win_loss.append(temp3)

final_csv['home_score'] = score_home  
final_csv['visit_score'] = score_away
final_csv['home_win_loss'] = home_win_loss
poss_win_loss = []
for row in final_csv.iterrows():
    items = row[1]
    if items['Poss_Team'] == items['HomeTeamCode']:
        if items['home_score'] < items['visit_score']:
            poss_win_loss.append('Losing')
        if items['home_score'] > items['visit_score']:
            poss_win_loss.append('Winning')
                
    elif items['Poss_Team'] == items['VisitTeamCode']:
        if items['visit_score'] < items['home_score']:
            poss_win_loss.append('Losing')
        if items['visit_score'] > items['home_score']:
            poss_win_loss.append('Winning')
                
    if items['home_score'] == items['visit_score']:
        poss_win_loss.append('Draw')
    
final_csv['Poss_Team_Status'] = poss_win_loss
temp = final_csv["home_win_loss"].value_counts()
tempe = final_csv["Poss_Team_Status"].value_counts()
fig = {
    "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "hole": .4,
      "type": "pie"
    },
      {
      "values": tempe.values,
      "labels": tempe.index,
      "domain": {"x": [.52, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    
    ],
  "layout": {
        "title":"Game Status during the time of Concussion",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Home Team Status",
                 "x": 0.145,
                "y": 0.5
            },
              {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Poss Team Status",
                "x": 0.858,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = final_csv["Season_Type"].value_counts()
tempe = final_csv["Quarter"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "hole": .6,
      "type": "pie"
    },
      {
      "values": tempe.values,
      "labels": tempe.index,
      "domain": {"x": [.52, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Season and Game Quarter for Concussions",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Season",
                 "x": 0.170,
                "y": 0.5
            },
              {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Game Quarter",
                "x": 0.828,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
del merged_csv,game_review,play_information
gc.collect()
# Game Wise Analysis of Injuries 

# Turf Analysis 

temp = final_csv["Turf"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"In Which Turf most injuries occured ? ",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Turf",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
# Game Wise Analysis of Injuries 

# Weather Analysi
temp = final_csv["OutdoorWeather"].value_counts()
tempe = final_csv["GameWeather"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "hole": .4,
      "type": "pie"
    },
      {
      "values": tempe.values,
      "labels": tempe.index,
      "domain": {"x": [.52, 1]},
      "hole": .4,
      "type": "pie"
    },
    
    
    ],
  "layout": {
        "title":"How was the weather during the Match ?",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Outdoor Weather",
                 "x": 0.135,
                "y": 0.5
            },
              {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Game Weather",
                "x": 0.85,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
trace = {"x": final_csv["Stadium"], 
          "y": final_csv["Temperature"], 
          "marker": {"size": 12}, 
          "mode": "markers",  
          "type": "scatter"
}


data = [trace]
layout = {"title": "Games and Temperatures", 
          "xaxis": {"title": "Stadiums", }, 
          "yaxis": {"title": "Temperature (in Farenheit)"}}

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic_dot-plot')
trace0 = go.Scatter(
    x=final_csv.Home_Team,
    y=final_csv.Player_Activity_Derived,
    mode='markers',
    marker = dict(
          color = 'rgb(17, 157, 255)',
          size = 20,
          line = dict(
            color = 'rgb(231, 99, 250)',
            width = 2
          ))
)
layout = dict(title='Home Team activity leading to concussion '
)
fig = go.Figure(data=[trace0], layout=layout)
iplot(fig, filename='bubblechart-color')
trace0 = go.Scatter(
    x=final_csv.Visit_Team,
    y=final_csv.Player_Activity_Derived,
    mode='markers',
    marker = dict(
          color = 'rgb(17, 157, 255)',
          size = 20,
          line = dict(
            color = 'rgb(231, 99, 250)',
            width = 2
          ))
)
layout = dict(
            title='Visit Team activity leading to concussion '
)
fig = go.Figure(data=[trace0], layout=layout)
iplot(fig, filename='bubblechart-color')
x = final_csv.Home_Team
y = final_csv.Visit_Team

data = [
    go.Histogram2d(
        x=x,
        y=y
    )
]
layout = dict(
            title='Home Team vs Away Team and Concussions ',
            xaxis=dict(
                    title='Home Team'
                        ),
            yaxis=dict(
                    title='Away Team',
                        )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
del temp,tempe,x,y
gc.collect()
def calculate_speeds(df, dt=None, SI=False):
    data_selected = df[['Time', 'x','y']]
    if SI==True:
        data_selected.x = data_selected.x / 1.0936132983
        data_selected.y = data_selected.y / 1.0936132983
    # Might have used shift pd function ?
    data_selected_diff = data_selected.diff()
    if dt==None:
        # Time is now a timedelta and need to be converted
        data_selected_diff.Time = data_selected_diff.Time.apply(lambda x: (x.total_seconds()))
        data_selected_diff['Speed'] = (data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64).apply(np.sqrt) / data_selected_diff.Time
    else:
        # Need to be sure about the time step...
        data_selected_diff['Speed'] = (data_selected_diff.x **2 + data_selected_diff.y **2).astype(np.float64).apply(np.sqrt) / dt
    #data_selected_diff.rename(columns={'Time':'TimeDelta'}, inplace=True)
    #return data_selected_diff
    df['TimeDelta'] = data_selected_diff.Time
    df['Speed'] = data_selected_diff.Speed
    return df[1:]

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

df_list = []

buffer = ['NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv',
             'NGS-2017-reg-wk13-17.csv',
             'NGS-2017-post.csv']
ngs_files = ['NGS-2016-pre.csv',
             'NGS-2016-reg-wk1-6.csv',
             'NGS-2016-reg-wk7-12.csv','NGS-2016-reg-wk13-17.csv']

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv(f'../input/'+i, usecols=col_names,dtype=dtypes)
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df.Time = pd.to_datetime(df.Time, format =date_format)
    df.sort_values(sortBy, inplace=True)
    df = calculate_speeds(df, SI=True)
    df_list.append(df)
    del df
    gc.collect()

ngs = pd.concat(df_list)

del df_list
gc.collect()
#Converting everything to meters and speed to KMPH
ngs['x'] = ngs['x']/1.0936
ngs['y'] = ngs['y']/1.0936
ngs['dis'] = ngs['dis']/1.0936
ngs['Speed'] = ngs['Speed']* 3.6
ngs = ngs[ngs.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 
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
cut_speed=44 # World record 9,857232 m/s for NFL
ngs = remove_wrong_values(ngs, cutspeed=cut_speed)
video_review = pd.read_csv('../input/video_review.csv')
final = pd.merge(final_csv,ngs,on=['Season_Year','GameKey','PlayID','GSISID'])

def load_layout():
    """
    Returns a dict for a Football themed Plot.ly layout 
    """
    layout = dict(
        title = "Player Pitch Activity",
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
def plot_play(game_df, PlayID, player1=None, player2=None, custom_layout=False):
    """
    Plots player movements on the field for a given game, play, and two players
    """
    game_df = game_df[game_df.PlayID==PlayID]
    finale = final[final.PlayID==PlayID]

    GameKey=str(pd.unique(game_df.GameKey)[0])
    traces=[]   
    listb = []
    
    list1= list(game_df[game_df.GSISID==player1].Event)
    list2= list(game_df[game_df.GSISID==player1].Speed)
    lista = ["Event: "+str(list1[i]) +" + Speed: "+ str(list2[i]) for i in range(len(list1))]
    if not lista:
        lista.append("None")
    
    list3= list(game_df[game_df.GSISID==player2].Event)
    list4= list(game_df[game_df.GSISID==player2].Speed)
    try:
        listb = ["Event: "+str(list3[i]) +" + Speed: "+ str(list4[i]) for i in range(len(list2))]
    except:
        listb.append("None")
        
    trace0 = go.Scatter(
                x = game_df[game_df.GSISID==player1].x,
                y = game_df[game_df.GSISID==player1].y,
                name='Primary GSISID '+str(player1),
                mode = 'lines+markers',
                text = lista,
                line = dict(width = 6,smoothing=1.1),
                marker=dict(
                size=12,
                line = dict(
                color= 'rgb(0,0,0)',
                width = 1),
                color = game_df[game_df.GSISID==player1].Speed, #set color equal to a variable
                colorscale='Viridis',
                colorbar=dict(
                title='Primary Speed'
                ),
                showscale=True
    )
            )
    trace1 = go.Scatter(
                x = game_df[game_df.GSISID==player2].x,
                y = game_df[game_df.GSISID==player2].y,
                name='Partner GSISID '+str(player2),
                text = listb,
                line = dict(
                width = 5),
                mode = 'lines+markers',
                marker=dict(
                size=10,
                line = dict(
                color= 'rgb(0,0,0)',
                width = 1),
                color = game_df[game_df.GSISID==player2].Speed, #set color equal to a variable
                colorscale='Portland',
                colorbar=dict(title='Partner Speed', x =-0.14),
                showscale= True
                )
            )
    
    layout = load_layout()
    layout['title'] = 'Player Activity (Concussion) in GameKey ' + GameKey + ' : ' + str(pd.unique(finale.Home_Team)[0]) +' v/s ' + str(pd.unique(finale.Visit_Team)[0])
    layout['legend'] = dict(orientation="h")
    data = [trace0,trace1]
    fig = dict(data=data, layout=layout)
    print(" Play Information")
    print(" Date :" + str(pd.unique(finale.Game_Date)[0]))
    print(" Home Team :" + str(pd.unique(finale.Home_Team)[0])+ ", Visiting Team : " + str(pd.unique(finale.Visit_Team)[0]) )
    print(" Player Activity Derived :" + str(pd.unique(finale.Player_Activity_Derived)[0])+ ", Primary Partner Activity Derived : " + str(pd.unique(finale.Primary_Partner_Activity_Derived)[0]) )
    print(" Primary Impact Type :" + str(pd.unique(finale.Primary_Impact_Type)[0])+ ", Punt Play Player Role : " + str(pd.unique(finale.Role)[0]) + ", Player Position : " + str(pd.unique(finale.Position)[0]))
    print(" Quarter of Play :" + str(pd.unique(finale.Quarter)[0])+ ", Pocession Team : " + str(pd.unique(finale.Poss_Team)[0]) + ", Score (Home-Visiting) : " + str(pd.unique(finale.Score_Home_Visiting)[0]))
    print(" Home Team Status :" + str(pd.unique(finale.home_win_loss)[0])+ ", Pocession Team Status : " + str(pd.unique(finale.Poss_Team_Status)[0]) )
    print(" Max Speed :" + str(game_df.Speed.max())+ ", Avg Speed :" + str(game_df.Speed.mean()) )
    print(" Stadium :" + str(pd.unique(finale.Stadium)[0])+ ", Turf : " + str(pd.unique(finale.Turf)[0])+", GameWeather :" + str(pd.unique(finale.GameWeather)[0])+ ", Temperature : " + str(pd.unique(finale.Temperature)[0])  )
    del finale
    #print("\n\n\t",play_description)
    iplot(fig)

plot_play(game_df=ngs, PlayID=3129, player1=31057, player2=32482 )
from IPython.display import HTML
# Youtube
HTML('<iframe width="950" height="600" src="http://a.video.nfl.com//films/vodzilla/153233/Kadeem_Carey_punt_return-Vwgfn5k9-20181119_152809972_5000k.mp4" frameborder="0" allowfullscreen></iframe>')

plot_play(game_df=ngs, PlayID=2587, player1=29343, player2=31059 )
plot_play(game_df=ngs, PlayID=538, player1=31023, player2=31941 )
plot_play(game_df=ngs, PlayID=1212, player1=33121, player2=28249 )
plot_play(game_df=ngs, PlayID=1045, player1=32444, player2=31756 )
plot_play(game_df=ngs, PlayID=2342, player1=32410, player2=23259 )
plot_play(game_df=ngs, PlayID=3663, player1=28128, player2=29629 )
plot_play(game_df=ngs, PlayID=3509, player1=27595, player2=31950 )
plot_play(game_df=ngs, PlayID=3468, player1=28987, player2=31950)
plot_play(game_df=ngs, PlayID=1976, player1=32214, player2=32807 )
HTML('<iframe width="950" height="600" src="https://nfl-vod.cdn.anvato.net/league/5691/18/11/25/284954/284954_75F12432BA90408C92660A696C1A12C8_181125_284954_huber_punt_3200.mp4" frameborder="0" allowfullscreen></iframe>')
finale = pd.merge(final_csv,ngs,on=['GSISID'])
import plotly.plotly as py
import plotly.graph_objs as go
temp = finale["Event"].value_counts()
tempe = final["Event"].value_counts()

trace1 = go.Bar(
    x=temp.index,
    y=temp.values,
    name='All Games'
)
trace2 = go.Bar(
    x=tempe.index,
    y=tempe.values,
    name='Concussion'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

layout = go.Layout(title='Events over the Games')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar')
speed_during_punt = final.loc[final['Event'].isin(['punt'])]
speed_during_puntrec = final.loc[final['Event'].isin(['punt_received'])]
speed_during_tackle = final.loc[final['Event'].isin(['tackle'])]
speed_during_down = final.loc[final['Event'].isin(['punt_downed'])]
speed_during_fumble = final.loc[final['Event'].isin(['fumble'])]
speed_during_catch = final.loc[final['Event'].isin(['fair_catch'])]
trace0 = go.Box(
    y=finale.Speed,
    name = 'During Whole Game',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(165,42,42)',
    ),
)

trace1 = go.Box(
    y=speed_during_punt.Speed,
    name = 'During Punt',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(214,12,140)',
    ),
)

trace2 = go.Box(
    y=speed_during_puntrec.Speed,
    name = 'During Punt Rec',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(238,130,238)',
    ),
)

trace3 = go.Box(
    y=speed_during_tackle.Speed,
    name = 'During Tackle',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(46,139,87)',
    ),
)


trace4 = go.Box(
    y=speed_during_down.Speed,
    name = 'During Punt Downed',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(255,215,0)',
    ),
)

trace5 = go.Box(
    y=speed_during_fumble.Speed,
    name = 'During Fumble',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(0,191,255)',
    ),
)

trace6 = go.Box(
    y=speed_during_catch.Speed,
    name = 'During Fair Catch',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(176,48,96)',
    ),
)

layout = go.Layout(
    width=900,
    height=500,
    yaxis=dict(
        title='Speed of the Concussed Player',
        zeroline=False
    ),
)
data = [trace0,trace1,trace2,trace3,trace4,trace5,trace6]
layout = go.Layout(title='Speed of the Players and Events 2016')
fig= go.Figure(data=data, layout=layout)
iplot(fig, filename='alcohol-box-plot')
density_punt = finale.loc[finale['Event'].isin(['punt'])]
density_punts = final.loc[final['Event'].isin(['punt'])]
density_puntrec = finale.loc[finale['Event'].isin(['punt_received'])]
density_puntrecs = final.loc[final['Event'].isin(['punt_received'])]
density_tackle = final.loc[final['Event'].isin(['tackle'])]
density_tackles = finale.loc[finale['Event'].isin(['tackle'])]
trace = go.Histogram2dContour(
        x = density_punt.x,
        y = density_punt.y
)

trace0 = go.Scatter(
    x = density_punts.x,
    y = density_punts.y,
    mode = 'markers',
    name = 'Position of Players (Concussed) during Punts',
    text = list(density_punts.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(25,25,112)',
          size = 14)
)

layout = load_layout()
layout['legend'] = dict(orientation="h")
layout['plot_bgcolor'] = 'rgb(220,220,220)'
layout['title'] = 'Player Density on field during Punt 2016'
data = [trace,trace0]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Basic Histogram2dContour")
trace = go.Histogram2dContour(
        x = density_puntrec.x,
        y = density_puntrec.y
)


trace0 = go.Scatter(
    x = density_puntrecs.x,
    y = density_puntrecs.y,
    mode = 'markers',
    name = 'Position of Players (Concussed) during Punt Recs',
    text = list(density_puntrecs.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(25,25,112)',
          size = 14)
)
layout = load_layout()
layout['plot_bgcolor'] = 'rgb(220,220,220)'
layout['legend'] = dict(orientation="h")
layout['title'] = 'Player Density on field during Punt Rec 2016'
data = [trace,trace0]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Basic Histogram2dContour")
trace = go.Scatter(
    x = density_tackle.x,
    y = density_tackle.y,
    mode = 'markers',
    name = 'Concussion',
    text = list(density_tackle.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(255, 0, 0)',
          size = 20)
)

trace0 = go.Scatter(
    x = density_tackles.x,
    y = density_tackles.y,
    mode = 'markers',
    name = 'Normal',
    marker = dict(
          color = 'rgb(238,221,130)',
          size = 10)
)

layout = load_layout()
layout['title'] = 'Tackle Points on field during 2016'
data = [trace0,trace]
fig = dict(data=data, layout=layout)
iplot(fig, filename='basic-scatter')
del density_punt,density_puntrec,ngs,final,finale
gc.collect()
del speed_during_punt,speed_during_puntrec,speed_during_tackle,speed_during_down,speed_during_fumble
gc.collect()
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

df_list = []

ngs_files = ['NGS-2017-pre.csv',
             'NGS-2017-reg-wk1-6.csv',
             'NGS-2017-reg-wk7-12.csv','NGS-2017-reg-wk13-17.csv']

for i in tqdm.tqdm(ngs_files):
    df = pd.read_csv(f'../input/'+i, usecols=col_names,dtype=dtypes)
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    sortBy = ['Season_Year', 'GameKey', 'PlayID', 'GSISID', 'Time']
    df.Time = pd.to_datetime(df.Time, format =date_format)
    df.sort_values(sortBy, inplace=True)
    df = calculate_speeds(df, SI=True)
    df_list.append(df)
    del df
    gc.collect()

ngs = pd.concat(df_list)

del df_list
gc.collect()
#Converting everything to meters and speed to KMPH
ngs['x'] = ngs['x']/1.0936
ngs['y'] = ngs['y']/1.0936
ngs['dis'] = ngs['dis']/1.0936
ngs['Speed'] = ngs['Speed']* 3.6
ngs = ngs[ngs.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 
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
cut_speed=44 # World record 9,857232 m/s for NFL
ngs = remove_wrong_values(ngs, cutspeed=cut_speed)
ngs.Speed.hist()
video_review = pd.read_csv('../input/video_review.csv')
final = pd.merge(final_csv,ngs,on=['Season_Year','GameKey','PlayID','GSISID'])
plot_play(game_df=ngs, PlayID=3630, player1=30171, player2=29384 )
HTML('<iframe width="950" height="600" src="http://a.video.nfl.com//films/vodzilla/153250/52_yard_Punt_by_Matt_Haack-ENsIvMyf-20181119_161418429_5000k.mp4" frameborder="0" allowfullscreen></iframe>')
plot_play(game_df=ngs, PlayID=2764, player1=32323, player2=31930 )
plot_play(game_df=ngs, PlayID=183, player1=33813, player2=33841 )
plot_play(game_df=ngs, PlayID=1088, player1=32615, player2=31999 )
plot_play(game_df=ngs, PlayID=1526, player1=32894, player2=31763 )
plot_play(game_df=ngs, PlayID=3312, player1=26035, player2=27442 )
plot_play(game_df=ngs, PlayID=1262, player1=33941, player2=27442 )
plot_play(game_df=ngs, PlayID=2792, player1=33838, player2=31317 )
plot_play(game_df=ngs, PlayID=2072, player1=29492, player2=33445 )
plot_play(game_df=ngs, PlayID=1683, player1=32820, player2=25503 )
HTML('<iframe width="950" height="600" src="http://a.video.nfl.com//films/vodzilla/153280/Wing_37_yard_punt-cPHvctKg-20181119_165941654_5000k.mp4" frameborder="0" allowfullscreen></iframe>')
finale = pd.merge(final_csv,ngs,on=['GSISID'])
import plotly.plotly as py
import plotly.graph_objs as go

temp = finale["Event"].value_counts()
tempe = final["Event"].value_counts()

trace1 = go.Bar(
    x=temp.index,
    y=temp.values,
    name='All Games'
)
trace2 = go.Bar(
    x=tempe.index,
    y=tempe.values,
    name='Concussion'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

layout = go.Layout(title='Events over the Games')
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar')
speed_during_punt = final.loc[final['Event'].isin(['punt'])]
speed_during_puntrec = final.loc[final['Event'].isin(['punt_received'])]
speed_during_tackle = final.loc[final['Event'].isin(['tackle'])]
speed_during_down = final.loc[final['Event'].isin(['punt_downed'])]
speed_during_fumble = final.loc[final['Event'].isin(['fumble'])]
speed_during_catch = final.loc[final['Event'].isin(['fair_catch'])]
trace0 = go.Box(
    y=finale.Speed,
    name = 'During Whole Game',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(165,42,42)',
    ),
)

trace1 = go.Box(
    y=speed_during_punt.Speed,
    name = 'During Punt',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(214,12,140)',
    ),
)

trace2 = go.Box(
    y=speed_during_puntrec.Speed,
    name = 'During Punt Rec',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(138,43,226)',
    ),
)

trace3 = go.Box(
    y=speed_during_tackle.Speed,
    name = 'During Tackle',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(30,144,255)',
    ),
)


trace4 = go.Box(
    y=speed_during_down.Speed,
    name = 'During Punt Downed',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(214,179,140)',
    ),
)

trace5 = go.Box(
    y=speed_during_fumble.Speed,
    name = 'During Fumble',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(254,199,140)',
    ),
)

trace6 = go.Box(
    y=speed_during_catch.Speed,
    name = 'During Fair Catch',
    boxpoints='all',
    jitter=0.3,
    marker = dict(
        color = 'rgb(176,48,96)',
    ),
)

layout = go.Layout(
    width=900,
    height=500,
    title = 'Speed , Concussion and Events',
    yaxis=dict(
        title='Speed of the Concussed Player 2017',
        zeroline=False
    ),
)
data = [trace0,trace1,trace2,trace3,trace4,trace5,trace6]
fig= go.Figure(data=data, layout=layout)
iplot(fig, filename='alcohol-box-plot')
density_punt = finale.loc[finale['Event'].isin(['punt'])]
density_punts = final.loc[final['Event'].isin(['punt'])]
density_puntrec = finale.loc[finale['Event'].isin(['punt_received'])]
density_puntrecs = final.loc[final['Event'].isin(['punt_received'])]
density_tackle = final.loc[final['Event'].isin(['tackle'])]
density_tackles = finale.loc[finale['Event'].isin(['tackle'])]
trace = go.Histogram2dContour(
        x = density_punt.x,
        y = density_punt.y
)

trace0 = go.Scatter(
    x = density_punts.x,
    y = density_punts.y,
    mode = 'markers',
    name = 'Position of Players (Concussed) during Punts',
    text = list(density_punts.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(25,25,112)',
          size = 14)
)

layout = load_layout()
layout['legend'] = dict(orientation="h")
layout['plot_bgcolor'] = 'rgb(220,220,220)'
layout['title'] = 'Player Density on field during Punt 2017'
data = [trace,trace0]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Basic Histogram2dContour")
trace = go.Histogram2dContour(
        x = density_puntrec.x,
        y = density_puntrec.y
)


trace0 = go.Scatter(
    x = density_puntrecs.x,
    y = density_puntrecs.y,
    mode = 'markers',
    name = 'Position of Players (Concussed) during Punt Recs',
    text = list(density_puntrecs.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(25,25,112)',
          size = 14)
)
layout = load_layout()
layout['plot_bgcolor'] = 'rgb(220,220,220)'
layout['legend'] = dict(orientation="h")
layout['title'] = 'Player Density on field during Punt Rec 2017'
data = [trace,trace0]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Basic Histogram2dContour")
trace = go.Scatter(
    x = density_tackle.x,
    y = density_tackle.y,
    mode = 'markers',
    name = 'Concussion',
    text = list(density_tackle.Role),
    marker = dict(
        symbol='x',
          color = 'rgb(255, 0, 0)',
          size = 20)
)

trace0 = go.Scatter(
    x = density_tackles.x,
    y = density_tackles.y,
    mode = 'markers',
    name = 'Normal',
    marker = dict(
          color = 'rgb(238,221,130)',
          size = 10)
)

layout = load_layout()
layout['title'] = 'Tackle Points on field during 2017'
data = [trace0,trace]
fig = dict(data=data, layout=layout)
iplot(fig, filename='basic-scatter')
del density_punt,density_puntrec,ngs,final,finale,final_csv
gc.collect()
