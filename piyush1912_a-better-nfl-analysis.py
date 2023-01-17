# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import matplotlib.patches as patches

import seaborn as sns

sns.set_style("darkgrid")

pd.set_option('display.max_rows', 1000)

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot

init_notebook_mode(connected=True)



pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Plot the Football Field.

# Source: https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12, 6.33)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

            plt.text(x, 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

        ax.plot([x, x], [22.91, 23.57], color='white')

        ax.plot([x, x], [29.73, 30.39], color='white')



    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

                 color='yellow')

    return fig, ax
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df



def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
InjuryRecord = import_data("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

PlayList = import_data("../input/nfl-playing-surface-analytics/PlayList.csv")

PlayerTrackData = import_data("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
# StadiumType corrections

stadium_dict = {"Oudoor" : "Outdoor", "Outdoors" : "Outdoor", "Outdor" : "Outdoor", "Ourdoor" : "Outdoor", 

                "Outdoor Retr Roof-Open" : "Outdoor", "Open" : "Outdoor", 'Outdoor Retr Roof-Open' : "Outdoor",

                "Outddors" : "Outdoor", 'Retr. Roof-Open' : "Outdoor",  "Indoor, Open Roof" : "Outdoor", "Heinz Field" : "Outdoor",

                 "Domed, Open" : "Outdoor", "Domed, open" : "Outdoor", 'Retr. Roof - Open' : "Outdoor", "Outside" : "Outdoor",

                "Closed Dome" : "Indoor", "Domed, closed" : "Indoor", "Dome" : "Indoor", "Domed" : "Indoor", 

                "Indoors" : "Indoor", 'Retr. Roof-Closed' : "Indoor", "Retractable Roof" : "Indoor",  'Indoor, Roof Closed' : "Indoor",

                "Retr. Roof - Closed" : "Indoor", 'Dome, closed' : "Indoor", "Retr. Roof Closed" : "Indoor", "nan" : np.NaN, 'Cloudy' : np.NaN}



# Weather corrections

weather_dict = {"Indoors" : "Indoor", "N/A (Indoors)": "Indoor", "Clear skies" : "Clear", "Clear Skies" : "Clear",

                "Clear and cold" : "Clear", 'Cloudy, light snow accumulating 1-3"' : "Cloudy",

                "Rain shower" : "Rain", "Cloudy, 50% change of rain" : "Cloudy", "Clear and warm" : "Clear", 

                "Cloudy with periods of rain" : "Cloudy", "Light Rain" : "Rain", "Light rain" : "Rain", 'Rain Chance 40%' : "Cloudy",

                "Mostly sunny" : "Sunny", "Mostly Sunny" : "Sunny", "Sun & clouds" : "Sunny", "Partly Cloudy" : "Cloudy",

                "Partly cloudy" : "Cloudy", "Coudy" : "Cloudy", "Party Cloudy" : "Cloudy", "Mostly Cloudy" : "Cloudy",

                "Mostly cloudy" : "Cloudy", "Cloudy, 50% change of rain" : "Cloudy", "Cloudy and Cool" : "Cloudy" , "nan" : np.NaN,

               'Cloudy, fog started developing in 2nd quarter' : "Cloudy", 'N/A Indoor' : "Indoor", 'Rain likely, temps in low 40s.' : "Rain",

               'Mostly Coudy' : "Cloudy", 'Scattered Showers' : "Showers", 'Heavy lake effect snow' : "Snow", 'Sunny Skies' : "Sunny",

               'Partly clear' : "Clear", 'Sunny, Windy' : "Sunny", 'cloudy' : "Cloudy", 'Sunny, highs to upper 80s' : "Sunny",

               'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' : "Cloudy", '10% Chance of Rain' : "Cloudy",

               '30% Chance of Rain' : "Cloudy", 'Mostly Sunny Skies' : "Sunny", 'Rainy' : "Rain", 'Cloudy, chance of rain' : "Cloudy",

               'Partly Clouidy' : "Cloudy", 'Partly Sunny' : "Sunny", 'Partly sunny' : "Sunny", 'Cloudy, Rain' : "Rain",

               'Clear and sunny' : "Clear and Sunny", 'Clear and Cool' : "Clear and Cold", 'Heat Index 95' : "Sunny", 'Clear to Partly Cloudy' : "Clear",

               'Clear and Cold' : 'Clear and cold'}



# Revert Days Missed to the Original

days_missed_dict = {1 : "1+", 2: "7+", 3 : "28+", 4 : "42+"}



PlayList["StadiumType"] = PlayList["StadiumType"].astype(str)

PlayList["StadiumType"].replace(stadium_dict, inplace=True)

PlayList["Weather"] = PlayList["Weather"].astype(str)

PlayList["Weather"].replace(weather_dict, inplace=True)

PlayList["PlayType"] = PlayList["PlayType"].replace('0', np.NaN)

InjuryRecord['DaysMissed'] = InjuryRecord['DM_M1'] + InjuryRecord['DM_M7'] + InjuryRecord['DM_M28'] + InjuryRecord['DM_M42']

InjuryRecord['DaysMissed'] = InjuryRecord["DaysMissed"].map(days_missed_dict)
InjuryRecord.head()
import random

def random_colors(number_of_colors):

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

                 for i in range(number_of_colors)]

    return color
fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type": "xy"}, {"type": "xy"}]],

    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = (' Body Parts', 

                                      ' Turf'))



pitch_count = InjuryRecord['BodyPart'].value_counts()

surface_count = InjuryRecord['Surface'].value_counts()



fig.add_trace(go.Bar(y=pitch_count.values,x=pitch_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=1)



fig.add_trace(go.Bar(y=surface_count.values,x=surface_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=2)







fig.update_layout(height=600,width=800,title=" Injuries by Body parts and Turf",title_x=0.5, showlegend=False)



fig.show()
PlayList.head()
fig = make_subplots(rows=1, cols=6,specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],

    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = (' Roster Positions', 

                                      ' Stadium Type',

                                      'Field Type',

                                      'Weather',

                                      'Play Type',

                                      'Position'))

rp_count = PlayList['RosterPosition'].value_counts()

stadium_count = PlayList['StadiumType'].value_counts()

field_count = PlayList['FieldType'].value_counts()

weather_count = PlayList['Weather'].value_counts()

play_count = PlayList['PlayType'].value_counts()

position_count = PlayList['Position'].value_counts()



fig.add_trace(go.Bar(y=rp_count.values,x=rp_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=1)



fig.add_trace(go.Bar(y=stadium_count.values,x=stadium_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=2)



fig.add_trace(go.Bar(y=field_count.values,x=field_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=3)



fig.add_trace(go.Bar(y=weather_count.values,x=weather_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=4)



fig.add_trace(go.Bar(y=play_count.values,x=play_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=5)



fig.add_trace(go.Bar(y=position_count.values,x=position_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=6)



fig.update_layout(height=600,width=1000,title=" All Non injury play data distribution",title_x=0.5, showlegend=False)



fig.show()
Injury_games_play = InjuryRecord.merge(PlayList,

                  on='PlayKey',

                  how='left')
# information about InjuryGamedata

Injury_games_play.info()
Injury_games_play
fig = make_subplots(

    rows=1, cols=8,

    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"},{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],

        shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = (' BodyPart',

                                      'Surface',

                                      'Roster Positions', 

                                      'Stadium Type',

                                      'Field Type',

                                      'Weather',

                                      'Play Type',

                                      'Position'))

pitch_count = Injury_games_play['BodyPart'].value_counts()

surface_count = Injury_games_play['Surface'].value_counts()

rp_count = Injury_games_play['RosterPosition'].value_counts()

stadium_count = Injury_games_play['StadiumType'].value_counts()

field_count = Injury_games_play['FieldType'].value_counts()

weather_count = Injury_games_play['Weather'].value_counts()

play_count = Injury_games_play['PlayType'].value_counts()

position_count = Injury_games_play['Position'].value_counts()



fig.add_trace(go.Bar(y=pitch_count.values,x=pitch_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=1)



fig.add_trace(go.Bar(y=surface_count.values,x=surface_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=2)



fig.add_trace(go.Bar(y=rp_count.values,x=rp_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=3)



fig.add_trace(go.Bar(y=stadium_count.values,x=stadium_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=4)



fig.add_trace(go.Bar(y=field_count.values,x=field_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=5)



fig.add_trace(go.Bar(y=weather_count.values,x=weather_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=6)



fig.add_trace(go.Bar(y=play_count.values,x=play_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=7)



fig.add_trace(go.Bar(y=position_count.values,x=position_count.index,hoverinfo='y',marker = dict(color = random_colors(25))),

              row=1, col=8)



fig.update_layout(height=600,width=1200, title=" All injury data distribution",title_x=0.5,showlegend=False)



fig.show()
Injury_games_play.groupby('BodyPart')['Surface'].value_counts()
df = Injury_games_play.groupby('BodyPart')['Surface'].value_counts().unstack().reset_index()

df
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = go.Figure(data=[

    go.Bar(x=df.BodyPart, y=df.Natural,text='y',hovertext='Natural'),

    go.Bar(x=df.BodyPart, y=df.Synthetic,text='y',hovertext='Synthetic')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=500,width=800, title=" All Body Part injury due to Turfs",xaxis_title='Turfs',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
Injury_games_play.groupby(['BodyPart','PlayType'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayType'])['Surface'].value_counts().unstack().reset_index()
df.head()
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = go.Figure(data=[

    go.Bar(x=df.BodyPart, y=df.Natural,text='y',hovertext=df['PlayType'],name='Natural'),

    go.Bar(x=df.BodyPart, y=df.Synthetic,text='y',hovertext=df['PlayType'],name='Synthetic')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=500,width=800, title=" All Body Part injury due to Play in different Turfs",xaxis_title='Body Parts',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','PlayType'])['Natural'].sum()
df.groupby(['BodyPart','PlayType'])['Synthetic'].sum()




fig = go.Figure(data=[

    go.Bar(x=df.PlayType, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.PlayType, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injury due to PlayType",xaxis_title='Types of play',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
Injury_games_play.groupby(['BodyPart','DaysMissed'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','DaysMissed'])['Surface'].value_counts().unstack().reset_index()





fig = go.Figure(data=[

    go.Bar(x=df.DaysMissed, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.DaysMissed, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injured for durations",xaxis_title='No of absent days due to injuries ',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','DaysMissed'])['Natural'].sum()
df.groupby(['BodyPart','DaysMissed'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','PlayerDay'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayerDay'])['Surface'].value_counts().unstack().reset_index()

df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Synthetic","Natural"))



fig.add_trace(go.Bar(x=df.PlayerDay, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=1, col=1)

fig.add_trace(go.Bar(x=df.PlayerDay, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=1, col=2)



# Change the bar mode

fig.update_xaxes(title_text="Player Day", row=1, col=1)

fig.update_xaxes(title_text="Player Day",row=1, col=2)

fig.update_yaxes(title_text="Injuries", row=1, col=1)

fig.update_yaxes(title_text="Injuries",row=1, col=2)

fig.update_layout(height=600,width=800, title=" All Body Part injury depending on Players day of the year",title_x=0.5,showlegend=False)

fig.show()
Injury_games_play.groupby(['BodyPart','PlayerGame'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayerGame'])['Surface'].value_counts().unstack().reset_index()
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Synthetic","Natural"))



fig.add_trace(go.Bar(x=df.PlayerGame, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=1, col=1)

fig.add_trace(go.Bar(x=df.PlayerGame, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=1, col=2)

# Change the bar mode

fig.update_xaxes(title_text="Player Game", row=1, col=1)

fig.update_xaxes(title_text="Player Game",row=1, col=2)

fig.update_yaxes(title_text="Injuries", row=1, col=1)

fig.update_yaxes(title_text="Injuries",row=1, col=2)

fig.update_layout(height=600,width=800, title=" All Body Part injury depending on Players Game",title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','PlayerGame'])['Natural'].sum()
df.groupby(['BodyPart','PlayerGame'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','RosterPosition'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','RosterPosition'])['Surface'].value_counts().unstack().reset_index()




fig = go.Figure(data=[

    go.Bar(x=df.RosterPosition, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.RosterPosition, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injured according to Injured Player Positions",xaxis_title='Injured Player Position',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','RosterPosition'])['Natural'].sum()
df.groupby(['BodyPart','RosterPosition'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','Position'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','Position'])['Surface'].value_counts().unstack().reset_index()




fig = go.Figure(data=[

    go.Bar(x=df.Position, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.Position, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injured due to Involving Player Position",xaxis_title='Involving Players Position',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','Position'])['Natural'].sum()
df.groupby(['BodyPart','Position'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','Weather'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','Weather'])['Surface'].value_counts().unstack().reset_index()




fig = go.Figure(data=[

    go.Bar(x=df.Weather, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.Weather, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injured due to weather",xaxis_title='Weather',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','Weather'])['Natural'].sum()
df.groupby(['BodyPart','Weather'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','StadiumType'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','StadiumType'])['Surface'].value_counts().unstack().reset_index()




fig = go.Figure(data=[

    go.Bar(x=df.StadiumType, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

    go.Bar(x=df.StadiumType, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural')

])

# Change the bar mode

fig.update_layout(barmode='stack')

fig.update_layout(height=600,width=800, title=" All Body Part injured due to different Stadium Types",xaxis_title='Stadium Types',yaxis_title='Injuries',title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','StadiumType'])['Natural'].sum()
df.groupby(['BodyPart','StadiumType'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','PlayType','DaysMissed'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayType','DaysMissed'])['Surface'].value_counts().unstack().reset_index()
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("Synthetic","Natural","Natural","Synthetic"))



fig.add_trace(go.Bar(x=df.PlayType, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=1, col=1)

fig.add_trace(go.Bar(x=df.PlayType, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=2, col=1)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=2, col=2)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=1, col=2)

# Change the bar mode

# Update xaxis properties

fig.update_xaxes(title_text="PlayType", row=1, col=1)

fig.update_xaxes(title_text="DaysMissed",row=1, col=2)

fig.update_xaxes(title_text="PlayType", showgrid=False, row=2, col=1)

fig.update_xaxes(title_text="DaysMissed", row=2, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Injuries", row=1, col=1)

fig.update_yaxes(title_text="Injuries", row=1, col=2)

fig.update_yaxes(title_text="Injuries", showgrid=False, row=2, col=1)

fig.update_yaxes(title_text="Injuries", row=2, col=2)

fig.update_layout(height=600,width=800, title=" All Body Part injury due to PlayType and DaysMissed",title_x=0.5,showlegend=False)

fig.show()
df.groupby(['BodyPart','PlayType','DaysMissed'])['Natural'].sum()
df.groupby(['BodyPart','PlayType','DaysMissed'])['Synthetic'].sum()
Injury_games_play.groupby(['BodyPart','PlayType','RosterPosition','DaysMissed'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayType','RosterPosition','DaysMissed'])['Surface'].value_counts().unstack().reset_index()
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = make_subplots(

    rows=3, cols=2,

    subplot_titles=("Synthetic","Natural","Synthetic","Natural","Synthetic","Natural"))



fig.add_trace(go.Bar(x=df.PlayType, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=1, col=1)

fig.add_trace(go.Bar(x=df.RosterPosition, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=2, col=1)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=3, col=1)

fig.add_trace(go.Bar(x=df.PlayType, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=1, col=2)

fig.add_trace(go.Bar(x=df.RosterPosition, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=2, col=2)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=3, col=2)

# Change the bar mode

# Update xaxis properties

fig.update_xaxes(title_text="PlayType", row=1, col=1)

fig.update_xaxes(title_text="PlayType",row=1, col=2)

fig.update_xaxes(title_text="RosterPosition", showgrid=False, row=2, col=1)

fig.update_xaxes(title_text="RosterPosition", row=2, col=2)

fig.update_xaxes(title_text="DaysMissed", showgrid=False, row=3, col=1)

fig.update_xaxes(title_text="DaysMissed", row=3, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Injuries", row=1, col=1)

fig.update_yaxes(title_text="Injuries", row=1, col=2)

fig.update_yaxes(title_text="Injuries", showgrid=False, row=2, col=1)

fig.update_yaxes(title_text="Injuries", row=2, col=2)

fig.update_yaxes(title_text="Injuries", showgrid=False, row=3, col=1)

fig.update_yaxes(title_text="Injuries", row=3, col=2)

fig.update_layout(height=1000,width=800, title=" All Body Part injury due to PlayType, Rosterposition and DaysMissed",title_x=0.5,showlegend=False)

fig.show()
Injury_games_play.groupby(['BodyPart','PlayType','Position','DaysMissed'])['Surface'].value_counts()
df = Injury_games_play.groupby(['BodyPart','PlayType','Position','DaysMissed'])['Surface'].value_counts().unstack().reset_index()
df['Natural'] = df['Natural'].fillna(0)

df['Synthetic'] = df['Synthetic'].fillna(0)



fig = make_subplots(

    rows=3, cols=2,

    subplot_titles=("Synthetic","Natural","Synthetic","Natural","Synthetic","Natural"))



fig.add_trace(go.Bar(x=df.PlayType, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=1, col=1)

fig.add_trace(go.Bar(x=df.Position, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=2, col=1)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Synthetic,text='y',hovertext=df['BodyPart'],name='Synthetic'),

              row=3, col=1)

fig.add_trace(go.Bar(x=df.PlayType, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=1, col=2)

fig.add_trace(go.Bar(x=df.Position, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=2, col=2)

fig.add_trace(go.Bar(x=df.DaysMissed, y=df.Natural,text='y',hovertext=df['BodyPart'],name='Natural'),

              row=3, col=2)

# Change the bar mode

# Update xaxis properties

fig.update_xaxes(title_text="PlayType", row=1, col=1)

fig.update_xaxes(title_text="PlayType",row=1, col=2)

fig.update_xaxes(title_text="Position", showgrid=False, row=2, col=1)

fig.update_xaxes(title_text="Position", row=2, col=2)

fig.update_xaxes(title_text="DaysMissed", showgrid=False, row=3, col=1)

fig.update_xaxes(title_text="DaysMissed", row=3, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Injuries", row=1, col=1)

fig.update_yaxes(title_text="Injuries", row=1, col=2)

fig.update_yaxes(title_text="Injuries", showgrid=False, row=2, col=1)

fig.update_yaxes(title_text="Injuries", row=2, col=2)

fig.update_yaxes(title_text="Injuries", showgrid=False, row=3, col=1)

fig.update_yaxes(title_text="Injuries", row=3, col=2)

fig.update_layout(height=1000,width=800, title=" All Body Part injury due to PlayType, Position and DaysMissed",title_x=0.5,showlegend=False)

fig.show()
def preprocess_samp_play(samp_play):



    play_df = PlayerTrackData[PlayerTrackData.PlayKey==samp_play]

    df = play_df.iloc[np.flatnonzero((play_df.event == 'ball_snap') | (play_df.event == 'kickoff'))[0]:]

    df['event'].ffill(inplace=True)



    # Calculate instantaneous acceleration

    df['a'] = (df.s - df.s.shift(1)) / (df.time - df.time.shift(1))

    df.a.iloc[0] = 0 # At the moment of ball_snap or kickoff, acceleration is likely 0

    

    return df
def plot_injury_play(samp_play, df):



    # Define the basic figure consisting in 6 traces: two in the field subplot,

    # 2 in the speed subplot, and 2 in the accel subplot. These will be updated

    # by frames:



    x = df.x.values

    y = df.y.values

    N = df.shape[0] - 1

    title_string = (samp_play+', '+

                   InjuryRecord[InjuryRecord.PlayKey==samp_play].BodyPart.values[0]+', '+

                   'M1: '+str(InjuryRecord[InjuryRecord.PlayKey==samp_play].DM_M1.values[0])+', '+

                   'M7: '+str(InjuryRecord[InjuryRecord.PlayKey==samp_play].DM_M7.values[0])+', '+

                   'M28: '+str(InjuryRecord[InjuryRecord.PlayKey==samp_play].DM_M28.values[0])+', '+

                   'M42: '+str(InjuryRecord[InjuryRecord.PlayKey==samp_play].DM_M42.values[0])+', '+

                   PlayList[PlayList.PlayKey==samp_play].RosterPosition.values[0]+', '+

                   str(PlayList[PlayList.PlayKey==samp_play].StadiumType.values[0])+', '+

                   PlayList[PlayList.PlayKey==samp_play].FieldType.values[0])



    fig = dict(

        layout = dict(height=400,

            xaxis1 = {'domain': [0.0, 0.75], 'anchor': 'y1', 'range': [0, 120], 'tickmode': 'array',

                      'tickvals': [0, 10, 35, 60, 85, 110, 120],

                      'ticktext': ['End', 'G', '25', '50', '25', 'G', 'End']},

            yaxis1 = {'domain': [0.0, 1], 'anchor': 'x1', 'range': [0, 160/3], 

                      'scaleanchor': 'x1', 'scaleratio': 1, 'tickmode': 'array',

                      'tickvals': [0, 23.583, 29.75, 160/3],

                      'ticktext': ['Side', 'Hash', 'Hash', 'Side']},

            xaxis2 = {'domain': [0.8, 1], 'anchor': 'y2', 'range': [0, N]},

            yaxis2 = {'domain': [0.0, 0.475], 'anchor': 'x2', 'range': [0, 10]},

            xaxis3 = {'domain': [0.8, 1], 'anchor': 'y3', 'range': [0, N],

                      'showticklabels': False},

            yaxis3 = {'domain': [0.525, 1], 'anchor': 'x3', 'range': [-10, 10]},

            title = {'text': title_string, 'y':0.92, 'x':0, 'xanchor': 'left', 'yanchor': 'top',

                     'font': dict(size=12)},

            annotations= [{"x": 0.9, "y": 0.425, "font": {"size": 12}, "text": "Speed",

                           "xref": "paper", "yref": "paper", "xanchor": "center",

                           "yanchor": "bottom", "showarrow": False},

                          {"x": 0.9, "y": 0.95, "font": {"size": 12}, "text": "Accel",

                           "xref": "paper", "yref": "paper", "xanchor": "center",

                           "yanchor": "bottom", "showarrow": False}],

            plot_bgcolor = 'rgba(181, 226, 141, 1)', # https://www.hexcolortool.com/#b5e28d

            margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50},

        ),



        data = [

            {'type': 'scatter', # This trace is identified inside frames as trace 0

             'name': 'f1', 

             'x': x, 

             'y': y, 

             'hoverinfo': 'name+text', 

             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},

             'line': {'color': 'rgba(255,79,38,1.000000)'}, 

             'mode': 'lines', 

             'fillcolor': 'rgba(255,79,38,0.600000)', 

             'legendgroup': 'f1',

             'showlegend': False, 

             'xaxis': 'x1', 'yaxis': 'y1'},

            {'type': 'scatter', # This trace is identified inside frames as trace 1

             'name': 'f12', 

             'x': [x[0]],

             'y': [y[0]],

             'mode': 'markers+text',

             'text': df.event.iloc[0],

             'textposition': 'middle left' if x[0] >= 60 else 'middle right', #'middle right',

             'showlegend': False,

             'marker': {'size': 10, 'color':'black'},

             'xaxis': 'x1', 'yaxis': 'y1'},

            {'type': 'scatter', # # This trace is identified inside frames as trace 2

             'name': 'f2', 

             'x': list(range(N)), 

             'y': df.s, 

             'hoverinfo': 'name+text', 

             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},

             'line': {'color': 'rgba(255,79,38,1.000000)'}, 

             'mode': 'lines', 

             'fillcolor': 'rgba(255,79,38,0.600000)', 

             'legendgroup': 'f2',

             'showlegend': False, 

             'xaxis': 'x2', 'yaxis': 'y2'},

            {'type': 'scatter', # This trace is identified inside frames as trace 3

             'name': 'f22', 

             'x': [0],

             'y': [df.s.iloc[0]],

             'mode': 'markers',

             'showlegend': False,

             'marker': {'size': 7, 'color':'black'},

             'xaxis': 'x2', 'yaxis': 'y2'},

            {'type': 'scatter', # # This trace is identified inside frames as trace 4

             'name': 'f3', 

             'x': list(range(N)), 

             'y': df.a, 

             'hoverinfo': 'name+text', 

             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},

             'line': {'color': 'rgba(255,79,38,1.000000)'}, 

             'mode': 'lines', 

             'fillcolor': 'rgba(255,79,38,0.600000)', 

             'legendgroup': 'f2',

             'showlegend': False, 

             'xaxis': 'x3', 'yaxis': 'y3'},

            {'type': 'scatter', # This trace is identified inside frames as trace 5

             'name': 'f33', 

             'x': [0],

             'y': [df.a.iloc[0]],

             'mode': 'markers',

             'showlegend': False,

             'marker': {'size': 7, 'color':'black'},

             'xaxis': 'x3', 'yaxis': 'y3'},

        ]





    )





    frames = [dict(name=k,

                   data=[dict(x=x, y=y),

                         dict(x=[x[k]], y=[y[k]], text=df.event.iloc[k]),

                         dict(x=list(range(N)), y=df.s),

                         dict(x=[k], y=[df.s.iloc[k]]),

                         dict(x=list(range(N)), y=df.a),

                         dict(x=[k], y=[df.a.iloc[k]]),

                       ],

                   traces=[0,1,2,3,4,5]) for k in range(N)]







    updatemenus = [dict(type='buttons',

                        buttons=[dict(label='Play',

                                      method='animate',

                                      args=[[f'{k}' for k in range(N)], 

                                             dict(frame=dict(duration=25, redraw=False), 

                                                  transition=dict(duration=0),

                                                  easing='linear',

                                                  fromcurrent=True,

                                                  mode='immediate'

                                                                     )]),

                                 dict(label='Pause',

                                      method='animate',

                                      args=[[None],

                                            dict(frame=dict(duration=0, redraw=False), 

                                                 transition=dict(duration=0),

                                                 mode='immediate' )])],

                        direction= 'left', 

                        pad=dict(r= 10, t=85), 

                        showactive =True, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')

                ]







    sliders = [{'yanchor': 'top',

                'xanchor': 'left', 

                'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},

                'transition': {'duration': 25.0, 'easing': 'linear'},

                'pad': {'b': 10, 't': 50}, 

                'len': 0.9, 'x': 0.1, 'y': 0, 

                'steps': [{'args': [[k], {'frame': {'duration': 25.0, 'easing': 'linear', 'redraw': False},

                                          'transition': {'duration': 0, 'easing': 'linear'}}], 

                           'label': k, 'method': 'animate'} for k in range(N)       

                        ]}]







    fig.update(frames=frames),

    fig['layout'].update(updatemenus=updatemenus,

              sliders=sliders)







    iplot(fig)
df_syn = Injury_games_play[Injury_games_play["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Ankle"]

df_syn_list = df_syn['PlayKey'].tolist()
samp_play = np.random.choice(InjuryRecord.PlayKey[~InjuryRecord.PlayKey.isna()])

dataset = preprocess_samp_play(samp_play)
# Calculate acceleration

PlayerTrackData['a'] = (PlayerTrackData.s - PlayerTrackData.s.shift(1)) / (PlayerTrackData.time - PlayerTrackData.time.shift(1))



# Calculate instantaneous jerk

PlayerTrackData['j'] = (PlayerTrackData.a - PlayerTrackData.a.shift(1)) / (PlayerTrackData.time - PlayerTrackData.time.shift(1))

dataset['j'] = (dataset.a - dataset.a.shift(1)) / (dataset.time - dataset.time.shift(1))
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['s'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['s'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player Speed Frequency Distribution', # title of plot

    xaxis_title_text='Speed of the player', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['o'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['o'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player Orientation Frequency Distribution', # title of plot

    xaxis_title_text='Angle Facing', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['a'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['a'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player accelaration Frequency Distribution', # title of plot

    xaxis_title_text='Acceleration of the player', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['j'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['j'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player jerk Frequency Distribution', # title of plot

    xaxis_title_text='Jerk of the player movement', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['x'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['x'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player X Position Frequency Distribution', # title of plot

    xaxis_title_text='X Position of the player', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Histogram(

    x=PlayerTrackData['y'].sample(10000),

    histnorm='percent',

    name='Non Injured'

))



fig.add_trace(go.Histogram(

    x=dataset['y'],

    histnorm='percent',

    name='Injured'

))



# The two histograms are drawn on top of another

fig.update_layout(

    barmode='stack',

    title_text='Injured vs Non Injured Player Y Position Frequency Distribution', # title of plot

    xaxis_title_text='Y position of the player', # xaxis label

    yaxis_title_text='Count', # yaxis label

    title_x=0.5,

)

fig.show()
fig = go.Figure()



fig.add_trace(go.Box(

    x=Injury_games_play['Temperature'],

    name='Injured play Temperature'

))



fig.add_trace(go.Box(

    x=PlayList['Temperature'],

    name='Non Injured play Temperature'

))



# The two histograms are drawn on top of another

fig.update_layout(

    title_text='Injured vs Non Injured Player Temperature Distribution', # title of plot

    xaxis_title_text='Temperature', # xaxis label

    title_x=0.5,

)

fig.update_xaxes(range=[0, 110])

fig.show()
print("Acceleration of player is" +" "+ str(dataset['a'].values[0]) + "  " + "Speed of player is" + " "+ str(dataset['s'].values[0]) )

example_play_id = dataset['PlayKey'].values[0]

fig, ax = create_football_field()

PlayerTrackData.query('PlayKey == @example_play_id').plot(kind='scatter', x='x', y='y', ax=ax, color='orange')

plt.show()


plot_injury_play(samp_play, dataset)