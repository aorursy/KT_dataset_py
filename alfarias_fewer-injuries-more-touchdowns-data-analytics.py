# Modules

# Core

import numpy as np

import pandas as pd

# Visualizations

import matplotlib.pylab as plt

import matplotlib.patches as patches

import seaborn as sns

sns.set_style("darkgrid")

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

from plotly.subplots import make_subplots

import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Tools

from IPython.display import HTML, display

import os

import io

import pprint

import warnings

warnings.filterwarnings("ignore")



%config InlineBackend.figure_format = "retina"



DATA_PATH = '../input/nfl-playing-surface-analytics/'



print(os.listdir(DATA_PATH))
# Helper Functions

# For reduce dataframe memory size, there's a big dataframe here

def reduce_mem_usage(df):

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

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

    return df



# Import data and reduce RAM usage

def import_data(file):

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df



# Get dataframe info like columns, missing values, data type, etc.

def get_df_info(df):

    display(df.head(3).style.hide_index().

            set_properties(**{'text-align': 'center'}))

    buf = io.StringIO()

    df.info(buf=buf)

    info = buf.getvalue().split('\n')[-2]

    df_samples = pd.DataFrame({'Number of Samples': df.shape[0], 

                               'Number of Features' : df.shape[1]},

                             index=[0])

    display(df_samples.style.hide_index()

            .set_properties(**{'text-align': 'center'}))

    df_types = df.dtypes

    df_types = pd.DataFrame({'Column':df_types.index, 'Type':df_types.values})

    display(df_types.style.hide_index()

            .set_properties(**{'text-align': 'center'})) 

    missing = df.isnull().sum().sort_values(ascending=False)

    if missing.values.sum() == 0:

        missing = pd.DataFrame({'Missing Values' : "No Missing Values"})

    else:

        missing = missing[missing > 0]

        missing = pd.DataFrame({'Feature' : missing.index, 'Missing Values' : missing.values})

        display(missing.style.hide_index()

                .set_properties(**{'text-align': 'center'}))

  



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
injury_record = import_data(DATA_PATH+"InjuryRecord.csv")

play_list = import_data(DATA_PATH+"PlayList.csv")

player_track = pd.read_csv(DATA_PATH+"PlayerTrackData.csv")
print('injury_record Data Information')

get_df_info(injury_record)

print('play_list Data Information')

get_df_info(play_list)

print('player_track Data Information')

get_df_info(player_track)
# Clean Data

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



play_list["StadiumType"] = play_list["StadiumType"].astype(str)

play_list["StadiumType"].replace(stadium_dict, inplace=True)

play_list["Weather"] = play_list["Weather"].astype(str)

play_list["Weather"].replace(weather_dict, inplace=True)

play_list["PlayType"] = play_list["PlayType"].replace('0', np.NaN)

injury_record['DaysMissed'] = injury_record['DM_M1'] + injury_record['DM_M7'] + injury_record['DM_M28'] + injury_record['DM_M42']

injury_record['DaysMissed'] = injury_record["DaysMissed"].map(days_missed_dict)

injury_record.drop(["DM_M1", "DM_M7", "DM_M28", "DM_M42"], axis=1, inplace=True)
display(injury_record[injury_record.duplicated(["PlayKey"], keep=False)]

        .dropna().style.hide_index())
gameid_list = injury_record["GameID"]



df_aux = pd.DataFrame().reindex_like(play_list)

df_aux = df_aux.dropna().reset_index()

for id in gameid_list:

    df_aux_b = play_list[play_list["GameID"] == id]

    df_aux = df_aux.append(df_aux_b.tail(1))

df_aux = df_aux.drop("index", 1).reset_index(drop=True)

display(df_aux.head(15).style.hide_index())

display(injury_record.head(15).style.hide_index())
display(df_aux.tail(15).style.hide_index())

display(injury_record.tail(15).style.hide_index())
injury_record["PlayKey"] = df_aux["PlayKey"]

aux = injury_record.isnull().sum().sum()

print(f"There are {aux} Features with missing values on Injury Record.")
inj_playlist = play_list[["PlayKey", "RosterPosition", "StadiumType", "Temperature", "Weather",

                        "PlayType", "PlayerGamePlay", "Position", "PositionGroup"]].merge(

                        injury_record, on="PlayKey", how="left")



inj_playlist.drop(inj_playlist.columns[9], axis=1)

del play_list

inj_playlist["StadiumType"] = inj_playlist["StadiumType"].astype('category')

inj_playlist["DaysMissed"] = inj_playlist["DaysMissed"].astype('category')
value = inj_playlist['BodyPart'].value_counts().sort_values(ascending=False)

label = value.index

trace = go.Bar(y=value, 

               x=label,

               marker={'color': value,

                   'colorscale': 'hsv'})

layout = go.Layout(title='Figure 1 - Injured Body Parts', 

                   xaxis={'title' : 'Body Part'},

                  yaxis={'title' : 'Total Count'},

                  title_x=0.5)

fig = go.Figure(data=trace, layout=layout)

iplot(fig)
df_aux = inj_playlist.loc[inj_playlist['BodyPart'].notnull()]

df_aux = inj_playlist.drop_duplicates(subset="PlayKey", keep="first", inplace=False)



fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Surface Distribution', 

                                      '(b) Minimum Days Missed Distribution'))



features = ["Surface", "DaysMissed"]

col_num = 1

for i in features:

    value = df_aux[i].value_counts()

    label = value.index

    fig.add_trace(go.Pie(labels=label, 

                         values=value,

                         showlegend=False,

                         hoverinfo="value",

                         title_text=" ",

                         title_position = "top center",

                         texttemplate = "%{label} <br>(%{percent})",

                         textposition = "inside"),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, title="Figure 2 - Surface and Days Missed Distribution", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Synthetic)', 

                                      '(b) Natural)'))



feature = "BodyPart"

feature_filter = "Surface"

feature_value = ["Synthetic", "Natural"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 3 - Injuries by Surface Type", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Synthetic)', 

                                      '(b) Natural)'))



feature = "DaysMissed"

feature_filter = "Surface"

feature_value = ["Synthetic", "Natural"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 4 - Days Missed by Surface Type", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature = "DaysMissed"

feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 5 - Injuries by Days Missed", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature = "PlayType"

feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=850, title="Figure 6 - Injuries by Play Type", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature = "RosterPosition"

feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=870, title="Figure 7 - Injuries by Roster Position", 

                  title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature = "Weather"

feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=850, title="Figure 8 - Injuries by Weather", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature = "StadiumType"

feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    df_aux = df_aux[feature].value_counts()

    df_aux = df_aux[df_aux > 0]

    value = df_aux.values

    label = df_aux.index



    fig.add_trace(go.Bar(y=value, 

                         x=label,

                         hoverinfo='y',

                         showlegend=False,

                         marker={'color': value,

                         'colorscale': 'hsv'}),

                         row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 9 - Injuries by Stadium Type", title_x=0.5)

fig.show()
inj_playlist["Temperature"] = inj_playlist["Temperature"].replace(-999, np.NaN)



fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_playlist[inj_playlist[feature_filter] == i]

    value = df_aux.Temperature.values



    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 10 - Injuries by Temperature", title_x=0.5)

fig.show()
days_miss_dict = {"1+" : 1, "7+" : 2, "28+" : 3, "42+" : 4}



inj_playlist['Days_Missed'] = inj_playlist["DaysMissed"].map(days_miss_dict)

inj_playlist['Days_Missed'] = inj_playlist["Days_Missed"].astype(float)



body_part_dict = {"Ankle" : 0, "Knee" : 1, "Toes" : 2, "Foot" : 3, "Heel" : 4}



inj_playlist["Body_Part"] = inj_playlist["BodyPart"].map(body_part_dict)

inj_playlist["Body_Part"] = inj_playlist["Body_Part"].astype(float)



surface_dict = {"Synthetic" : 0, "Natural" : 1}



inj_playlist["Surface_"] = inj_playlist["Surface"].map(surface_dict)

inj_playlist["Surface_"] = inj_playlist["Surface_"].astype(float)





stadium_dict = {"Outdoor" : 0, "Indoor" : 1, "Bowl" : 2, "Heinz Field" : 3, "Outside" : 4}



inj_playlist["Stadium_Type"] = inj_playlist["StadiumType"].map(stadium_dict)

inj_playlist["Stadium_Type"] = inj_playlist["Stadium_Type"].astype(float)



playtype_dict = {'Pass' : 0,

 'Rush' : 1,

 'Kickoff' : 2,

 'Kickoff Not Returned' : 3,

 'Kickoff Returned' : 4,               

 'Field Goal' : 5,

 'Punt' : 6,

 'Punt Not Returned' : 7,

 'Punt Returned' : 8,

 'Extra Point' : 9}



inj_playlist["Play_Type"] = inj_playlist["PlayType"].map(playtype_dict)

inj_playlist["Play_Type"] = inj_playlist["Play_Type"].astype(float)



f,ax = plt.subplots(figsize=(12, 9))

sns.heatmap(inj_playlist.corr("pearson"), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.title("Figure 11(a) - Correlation of some Features (Pearson)", fontsize=14)

plt.show()



f,ax = plt.subplots(figsize=(12, 9))

sns.heatmap(inj_playlist.corr("spearman"), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.title("Figure 11(b) - Correlation of some Features (Spearman)", fontsize=14)

plt.show()
df_syn = inj_playlist[inj_playlist["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Ankle"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 12(a) - Routes for Ankle Injuries (Synthetic)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Knee"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 12(b) - Routes for Knee Injuries (Synthetic)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Toes"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 12(c) - Routes for Toes Injuries (Synthetic)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Foot"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 12(d) - Routes for Foot Injuries (Synthetic)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Synthetic"]

df_syn = df_syn[df_syn["BodyPart"] == "Ankle"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 12(e) - Routes for Heel Injuries (Synthetic)", fontsize=14)

plt.show()
df_syn = inj_playlist[inj_playlist["Surface"] == "Natural"]

df_syn = df_syn[df_syn["BodyPart"] == "Ankle"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 13(a) - Routes for Ankle Injuries (Natural)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Natural"]

df_syn = df_syn[df_syn["BodyPart"] == "Knee"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 13(b) - Routes for Knee Injuries (Natural)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Natural"]

df_syn = df_syn[df_syn["BodyPart"] == "Toes"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 13(c) - Routes for Toes Injuries (Natural)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Natural"]

df_syn = df_syn[df_syn["BodyPart"] == "Foot"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 13(d) - Routes for Foot Injuries (Natural)", fontsize=14)

plt.show()



df_syn = inj_playlist[inj_playlist["Surface"] == "Natural"]

df_syn = df_syn[df_syn["BodyPart"] == "Ankle"]

df_syn_list = df_syn['PlayKey'].tolist()

fig, ax = create_football_field()

for playkey, inj_play in player_track.query('PlayKey in @df_syn_list').groupby('PlayKey'):

    inj_play.plot(kind='scatter', x='x', y='y', ax=ax, color='red', alpha=0.2)



plt.title("Figure 13(e) - Routes for Heel Injuries (Natural)", fontsize=14)

plt.show()
df_data = {"delta_time": [0.0], "delta_x": [0.0], "delta_y": [0.0], 

                               "delta_dir": [0.0], "delta_dis": [0.0], "delta_o": [0.0], "delta_s": [0.0]}





df_aux_c = pd.DataFrame(columns=["delta_time", "delta_x", "delta_y", "delta_dir", "delta_dis", "delta_o", "delta_s"])

df_aux = pd.DataFrame(data=df_data)

df_aux_b = inj_playlist[inj_playlist.BodyPart.notnull()]

for i in range(len(inj_playlist[inj_playlist.BodyPart.notnull()])):

    aux = df_aux_b["PlayKey"].values[i]

    aux = player_track[player_track["PlayKey"] == aux]

    aux_a = aux.head(1)

    aux_b = aux.tail(1)

    df_aux_c[["delta_time", "delta_x", "delta_y", "delta_dir", "delta_dis", "delta_o", "delta_s"]] = aux_b[["time",

                                                                                                          "x", "y", "dir", "dis", 

                                                              "o", "s"]] - aux_a[["time", "x", "y", "dir", "dis", "o", "s"]].values

    df_aux = df_aux.append(df_aux_c)

    df_aux_c.reset_index(drop=True, inplace=True)

    df_aux_c = df_aux_c.drop(df_aux_c.index[0])

    df_aux.reset_index(drop=True, inplace=True)

df_aux = df_aux.drop(df_aux.index[0])
inj_track = pd.concat([injury_record, df_aux], axis=1)
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_time.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 14 - Injuries by Delta Time", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_x.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 15 - Injuries by Delta X", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_y.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 16 - Injuries by Delta Y", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_dir.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 17 - Injuries by Delta Direction", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_dis.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 18 - Injuries by Delta Distance", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_o.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 19 - Injuries by Delta Orientation", title_x=0.5)

fig.show()
fig = make_subplots(rows=1, cols=5, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"},

                                          {"type": "xy"}, {"type": "xy"}]], 

                    shared_xaxes=True, shared_yaxes=True,

                    vertical_spacing=1,

                    subplot_titles = ('(a) Ankle', 

                                      '(b) Knee',

                                      '(c) Toes',

                                      '(d) Foot',

                                      '(e) Heel'))



feature_filter = "BodyPart"

feature_value = ["Ankle", "Knee", "Toes", "Foot", "Heel"]

col_num = 1



for i in feature_value:

    df_aux = inj_track[inj_track[feature_filter] == i]

    value = df_aux.delta_s.values

    fig.add_trace(go.Histogram(x=value, showlegend=False),

                 row=1, col=col_num)

    col_num = col_num+1



fig.update_layout(height=500, width=800, title="Figure 20 - Injuries by Delta Speed", title_x=0.5)

fig.show()
inj_track = inj_track.drop(columns=['PlayerKey'])



inj_track["Body_Part"] = inj_track["BodyPart"].map(body_part_dict)

inj_track["Body_Part"] = inj_track["Body_Part"].astype(float)



f,ax = plt.subplots(figsize=(12, 9))

sns.heatmap(inj_track.corr("pearson"), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.title("Figure 21(a) - Correlation of Delta Features with Body Part (Pearson)", fontsize=14)

plt.show()



f,ax = plt.subplots(figsize=(12, 9))

sns.heatmap(inj_track.corr("spearman"), annot=True, linewidths=.5, fmt= '.4f',ax=ax)

plt.title("Figure 21(b) - Correlation of Delta Features with Body Part (Spearman)", fontsize=14)

plt.show()