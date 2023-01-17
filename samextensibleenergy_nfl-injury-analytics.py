# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import timeit          #library to check the time to run a code

import json            #library to read the json formats

import pandas as pd    #library to execute Dataframe

import numpy as np     #library to execute Numerical/Statistical Calculations

from pandas.io.json import json_normalize # library to normalize other JSON formats

import numpy as np

from functools import reduce

from math import radians, cos, sin, asin, sqrt

from datetime import datetime 

import datetime as dt

from collections import Counter

import gc



import warnings

warnings.filterwarnings('ignore')



# Data Visualization Tools

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.tools as tls

import plotly.express as px

import plotly.figure_factory as ff





#Libraries for Modeling

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm, tree

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# p_list_df_raw = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

# p_trk_df_raw = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')

injury_df_raw = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
# injury_df = injury_df_raw.copy()

injury_df_raw.shape
## Checking Repeated injuries in same players in the tournament

injury_df = injury_df_raw

Repeat_injury = injury_df[injury_df['PlayerKey'].duplicated()]

print(Repeat_injury['PlayerKey'])
Repeat_injury = injury_df.loc[injury_df['PlayerKey'].isin([43540,45950,44449,33337,47307])].sort_values(by=['PlayerKey'])

Repeat_injury[['PlayerKey','GameID','BodyPart','Surface']]
# Categorizing the Injury Impact [1+ : Low(1) , 7+ : Moderate(2) , 28+ : High(3) , 42+: Extreme(4)]

injury_df = injury_df_raw

injury_df['Injury_Impact']  = 0



for i in range(len(injury_df)):

    if (injury_df['DM_M42'][i] == 1):

        injury_df['Injury_Impact'][i] = 4

        

    elif (injury_df['DM_M28'][i] == 1) and (injury_df['DM_M42'][i] == 0):

        injury_df['Injury_Impact'][i] = 3

        

    elif (injury_df['DM_M7'][i] == 1) and (injury_df['DM_M28'][i] == 0):

        injury_df['Injury_Impact'][i] = 2

    

    else:

        injury_df['Injury_Impact'][i] = 1

        

injury_df_Final = injury_df.drop(columns=['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'])





injury_df_Final['Injured'] = 1 #also adding the tag as injured players

injury_df_Final.tail()
for i in range(len(injury_df)):

    if (injury_df['DM_M42'][i] == 1):

        injury_df['Injury_Impact'][i] = '6+_Weeks'

        

    elif (injury_df['DM_M28'][i] == 1) and (injury_df['DM_M42'][i] == 0):

        injury_df['Injury_Impact'][i] = '4+_Weeks'

        

    elif (injury_df['DM_M7'][i] == 1) and (injury_df['DM_M28'][i] == 0):

        injury_df['Injury_Impact'][i] = '1+_Week'

    

    else:

        injury_df['Injury_Impact'][i] = '1+_Day'

        

injury_df_Final = injury_df.drop(columns=['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'])





############################################################################################################################

#                                        Injury Impact DISTRIBUTION

############################################################################################################################





import plotly.express as px



data = px.histogram(injury_df_Final, x="Injury_Impact", title='<b>Histogram of Injury Impact: Based on recovery Days</b>',

                      opacity=0.7, color_discrete_sequence=['indianred'])



layout = go.Layout(

)



fig = go.Figure(data=data, layout=layout)



fig.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Recovery Days (Injury Impact)</b>",

    yaxis_title="<b>Count of Players</b>",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='LightPink')

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



############################################################################################################################

#                                        Injury Type DISTRIBUTION

############################################################################################################################



import plotly.express as px



data = px.histogram(injury_df_Final, x="BodyPart", histnorm='percent', title='<b>Histogram of Body Part Distribution</b>',

                      opacity=0.7, color_discrete_sequence=['darkkhaki'])

layout = go.Layout()

fig0 = go.Figure(data=data, layout=layout)





fig0.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig0.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Injured Body part</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig0.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig0.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig0.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='khaki')

fig0.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



fig.show()

fig0.show()



del fig

del fig0

del data

gc.collect()
injury_df = injury_df_raw.copy()

injury_df['Injury_Impact']  = 0



for i in range(len(injury_df)):

    if (injury_df['DM_M42'][i] == 1):

        injury_df['Injury_Impact'][i] = 4

        

    elif (injury_df['DM_M28'][i] == 1) and (injury_df['DM_M42'][i] == 0):

        injury_df['Injury_Impact'][i] = 3

        

    elif (injury_df['DM_M7'][i] == 1) and (injury_df['DM_M28'][i] == 0):

        injury_df['Injury_Impact'][i] = 2

    

    else:

        injury_df['Injury_Impact'][i] = 1

        

injury_df_Final = injury_df.drop(columns=['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'])





injury_df_Final['Injured'] = 1 #also adding the tag as injured players





# injury_df_Final.tail()

del injury_df_raw

del injury_df

del Repeat_injury

#collect residual garbage

gc.collect()
p_list_df_raw = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

p_list_df = p_list_df_raw

#delete when no longer needed

del p_list_df_raw

#collect residual garbage

gc.collect()



#########################################################################################################################

#                                              Weather Tagging

#########################################################################################################################



Cloudy = ['Cloudy','Hazy','Cloudy and Cool', 'Overcast', 'Rain Chance 40%', 'cloudy', 'Cloudy, fog started developing in 2nd quarter', 

          'Cloudy with periods of rain, thunder possible. Winds shifting to WNW',

          'Mostly Cloudy', 'Mostly cloudy', 'Partly Sunny', 'Partly sunny','Mostly Coudy', 'Mostly_Cloudy',

          'Partly Cloudy','Partly cloudy', 'Party Cloudy','Partly Clouidy', 'Partly_Cloudy']  



Sunny = ['Sunny', 'Sunny and warm', 'Mostly Sunny', 'Sunny and clear', 'Sunny Skies','Heat Index 95', 'Sunny, highs to upper 80s', 'Sun & clouds', 'Mostly sunny', 'Sunny, Windy', 'Mostly Sunny Skies']

Clear = ['Clear and warm', 'Fair', 'Clear', 'Clear and Cool', 'Clear Skies', 'Clear skies', 'Partly clear', '10% Chance of Rain', 'Clear and sunny','Clear to Partly Cloudy']          

Rain = ['Rain', 'Showers', 'Scattered Showers', 'Light Rain', 'Cloudy, Rain', 'Rainy','30% Chance of Rain', 'Rain shower']          

Indoor = ['Controlled Climate', 'Indoor', 'Indoors', 'N/A (Indoors)','N/A Indoor']

Snow_Cold = ['Snow', 'Heavy lake effect snow', 'Cloudy, light snow accumulating 1-3"', 'Cloudy and cold',

        'Clear and cold', 'Sunny and cold', 'Cold', 'Rain likely, temps in low 40s.']

    

p_list_df['Weather'] = p_list_df['Weather'].astype(str)   

          

def assign_feature_problem(data):

    

    if any(word in data for word in Cloudy):

          data = "Cloudy"

          return data

    elif any(word in data for word in Sunny):

          data = "Sunny"

          return data

    elif any(word in data for word in Clear):

          data = "Clear"

          return data

    elif any(word in data for word in Rain):

          data = "Rain"

          return data

    elif any(word in data for word in Indoor):

          data = "Indoor"

          return data

    elif any(word in data for word in Snow_Cold):

          data = "Snow_Cold"

          return data

 

p_list_df['Weather'] = p_list_df['Weather'].apply(lambda x : assign_feature_problem(x))





#########################################################################################################################

#                                              Stadium Type Tagging

#########################################################################################################################



Outdoor = ['Outdoor', 'Outdoors', 'Open', 'Domed, open', 'Oudoor', 'Domed, Open', 'Ourdoor', 'Outdoor Retr Roof-Open', 'Outddors',

           'Retr. Roof-Open', 'Retr. Roof - Open', 'Indoor, Open Roof', 'Outdor', 'Outside', 'Cloudy', 'Heinz Field',

           'Retractable Roof']

Indoor = ['Indoors', 'Dome', 'Indoor', 'Domed, closed', 'Dome, closed', 'Closed Dome', 'Domed', 'Indoor, Roof Closed', 

          'Retr. Roof Closed', 'Retr. Roof - Closed', 'Retr. Roof-Closed']





p_list_df['StadiumType'] = p_list_df['StadiumType'].astype(str)   

          

def assign_feature_problem(data):

    

    if any(word in data for word in Outdoor):

          data = "Outdoor"

          return data

    elif any(word in data for word in Indoor):

          data = "Indoor"

          return data

 

p_list_df['StadiumType'] = p_list_df['StadiumType'].apply(lambda x : assign_feature_problem(x))



# Hence adjusting the Weather Column as well

p_list_df['Weather'] = np.where(p_list_df['StadiumType']=='Indoor', 'Indoor', p_list_df['Weather'])

    

#########################################################################################################################

#                                              Roster Position Tagging

#########################################################################################################################



QB = 'Quarterback'

WR = 'Wide Receiver'

LB = 'Linebacker'

RB = 'Running Back'

DL = 'Defensive Lineman'

TE = 'Tight End'

DB = ['Safety', 'Cornerback']

OL = 'Offensive Lineman'

SPEC = 'Kicker' 





p_list_df['RosterPosition'] = p_list_df['RosterPosition'].astype(str)   

          

def assign_feature_problem(data):

    

    if any(word in data for word in QB):

          data = "QB"

          return data

    elif any(word in data for word in WR):

          data = "WR"

          return data

    elif any(word in data for word in LB):

          data = "LB"

          return data

    elif any(word in data for word in RB):

          data = "RB"

          return data

    elif any(word in data for word in DL):

          data = "DL"

          return data

    elif any(word in data for word in TE):

          data = "TE"

          return data

    elif any(word in data for word in DB):

          data = "DB"

          return data

    elif any(word in data for word in OL):

          data = "OL"

          return data

    elif any(word in data for word in SPEC):

          data = "SPEC"

          return data  

 

p_list_df['RosterPosition'] = p_list_df['RosterPosition'].apply(lambda x : assign_feature_problem(x))



p_list_df.tail()



#########################################################################################################################

#                                              Roster_Position_Retained Tagging

#########################################################################################################################



p_list_df['PositionRetained'] = 'N'





for i in range(len(p_list_df)):   

    if p_list_df['RosterPosition'][i] == p_list_df['PositionGroup'][i]:

        p_list_df['PositionRetained'][i] = 'Y'

    else:

        continue

        

  

#########################################################################################################################

#                                              Final DataFrame

#########################################################################################################################



p_list_df_Final = p_list_df.drop(columns=['Temperature', 'Position'])

p_list_df_Final.tail()

#delete when no longer needed

del p_list_df

#collect residual garbage

gc.collect()
############################################################################################################################

#                                         Stadium TYPE DISTRIBUTION

############################################################################################################################



import plotly.express as px

    

data = px.histogram(p_list_df_Final, x="StadiumType", histnorm='percent',

                   title='<b>Distribution of Stadium Type</b>', opacity=0.7)



fig1 = go.Figure(data=data, layout=layout)





fig1.update_layout(

    autosize=False,

    width=500,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig1.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Stadium Type</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig1.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig1.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig1.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='LightPink')

fig1.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



# ############################################################################################################################

# #                                         FIELD TYPE DISTRIBUTION

# ############################################################################################################################



data = px.histogram(p_list_df_Final, x="FieldType", histnorm='percent',

                title='<b>Distribution of Field Type</b>', opacity=0.6, color_discrete_sequence=['olivedrab'])



fig2 = go.Figure(data=data, layout=layout)



fig2.update_layout(

   autosize=False,

    width=500,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig2.update_layout(

#     title="'<b>Distribution of Stadium Type v/s Field Type</b>'",

    title_x=0.5,

    xaxis_title=" <b>Field Type</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig2.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig2.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig2.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='LightPink')

fig2.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



############################################################################################################################

#                                         Weather DISTRIBUTION

############################################################################################################################



import plotly.express as px

    

data = px.histogram(p_list_df_Final, x="Weather", histnorm='percent',

                   title='<b>Distribution of Weather</b>', opacity=0.8, color_discrete_sequence=['crimson'])



fig3 = go.Figure(data=data, layout=layout)





fig3.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig3.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Weather Type</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig3.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig3.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig3.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='LightPink')

fig3.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)





fig1.show()

fig2.show()

fig3.show()





del fig1

del fig2

del fig3

del data

#collect residual garbage

gc.collect()

p_list_df = p_list_df_Final.merge(injury_df_Final, how = 'left', left_on='GameID', right_on='GameID')

p_list_df = p_list_df.drop(columns=['PlayerKey_y', 'PlayKey_y', 'Surface'])

p_list_df = p_list_df.fillna(0)

p_list_df = p_list_df.rename(columns={"PlayerKey_x": "PlayerKey", "PlayKey_x": "PlayKey"})

p_list_df = p_list_df.drop_duplicates(subset=['PlayKey']).reset_index().drop(columns=['index'])

p_list_df['Injury_Impact'] = p_list_df['Injury_Impact'].astype(int)

p_list_df['Injured'] = p_list_df['Injured'].astype(int)

p_list_df.to_csv('Merged_Injury_PlayerList.csv', index=False)

p_list_df





del p_list_df_Final

del injury_df_Final

#collect residual garbage

gc.collect()
p_trk_df_raw = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')

p_trk_df = p_trk_df_raw.drop(columns=['x', 'y', 's'])

#delete when no longer needed

del p_trk_df_raw

#collect residual garbage

gc.collect()
# Grouping By the individual Column

p_trk_df['Non_Alignment_Score'] = np.absolute((p_trk_df['dir'] - p_trk_df['o'])/45).apply(np.ceil)

p_trk_df['Non_Alignment_Score'] = p_trk_df['Non_Alignment_Score'].replace([5, 6, 7, 8], [3, 2, 1,0])

print(p_trk_df['Non_Alignment_Score'].unique())



p_trk_df = p_trk_df.drop(columns=['dir','o'])

Distance = p_trk_df.groupby(['PlayKey']).sum()[['dis']].reset_index().rename({'dis': 'Distance'}, axis=1)

p_trk_df = p_trk_df.drop(columns=['dis'])

Event = p_trk_df.dropna(subset=['event']).groupby(['PlayKey']).first()[['event']].reset_index().rename({'event':'Event'}, axis=1)

p_trk_df = p_trk_df.drop(columns=['event'])

Count = round((p_trk_df.groupby(['PlayKey']).size())/10).reset_index().rename({0: 'Seconds'}, axis=1)

Non_Alignment_Score = round(p_trk_df.groupby(['PlayKey']).mean()[['Non_Alignment_Score']]).reset_index()

p_trk_df = p_trk_df.drop(columns=['Non_Alignment_Score'])





data_frames = [Distance, Count, Event, Non_Alignment_Score]

plr_pos_Final = reduce(lambda  left,right: pd.merge(left,right,on=['PlayKey'],how='inner'), data_frames)



del Distance

del Event

del Count

del Non_Alignment_Score

gc.collect()



plr_pos_Final['Avg_Spd'] = round(plr_pos_Final['Distance']/plr_pos_Final['Seconds'])

plr_pos_Final
plr_pos_Final.to_csv('Downsized_PlrPos.csv', index=False)

data_frames = [p_list_df, plr_pos_Final]

Ftest_Df = reduce(lambda left,right: pd.merge(left,right,on=['PlayKey'], how='inner'), data_frames)

Ftest_Df.to_csv('Merged_Injury_PlayerList_PlayerPos.csv', index=False)

Ftest_Df.tail()



# del p_list_df

# del plr_pos_Final

# gc.collect()
# Ftest_Df = pd.read_csv('Merged_Injury_PlayerList_PlayerPos.csv')

Ftest_Df['Event'].nunique()
Huddle = ['huddle_start_offense', 'huddle_break_offense']

Pass = ['pass_forward', 'pass_arrived', 'pass_outcome_incomplete', 'pass_outcome_caught', 

        'pass_tipped', 'pass_shovel', 'pass_outcome_touchdown', 'pass_outcome_interception', 'pass_tipped']          

Punt = ['punt_play', 'punt', 'punt_land', 'punt_downed', 'punt_received', 'punt_fake','punt_muffed', 'punt_blocked']          

Kickoff = ['kickoff', 'kickoff_land', 'kickoff_play']

line_set = 'line_set'

ball_snap = 'ball_snap'

man_in_motion = 'man_in_motion'

shift = 'shift'

point_play = ['two_point_conversion', 'extra_point_attempt', 'extra_point']

handoff = 'handoff'

timeout = ['timeout_tv', 'timeout', 'timeout_quarter', 'timeout_home']

kick = ['onside_kick', 'drop_kick', 'free_kick']

Others = ['penalty_flag','field_goal_play', 'play_action', 'free_kick_play', 'two_point_play', 'qb_kneel', 'qb_sack',

               'snap_direct', 'run', 'two_minute_warning']





Ftest_Df['Event'] = Ftest_Df['Event'].astype(str)   

          

def assign_feature_problem(data):

    

    if any(word in data for word in Huddle):

          data = "Huddle"

          return data

    elif any(word in data for word in Pass):

          data = "Pass"

          return data

    elif any(word in data for word in Punt):

          data = "Punt"

          return data

    elif any(word in data for word in Kickoff):

          data = "Kickoff"

          return data

    elif any(word in data for word in line_set):

          data = "line_set"

          return data

    elif any(word in data for word in ball_snap):

          data = "ball_snap"

          return data

    elif any(word in data for word in man_in_motion):

          data = "man_in_motion"

          return data    

    elif any(word in data for word in shift):

          data = "shift"

          return data

    elif any(word in data for word in point_play):

          data = "point_play"

          return data

    elif any(word in data for word in handoff):

          data = "handoff"

          return data

    elif any(word in data for word in timeout):

          data = "timeout"

          return data

    elif any(word in data for word in kick):

          data = "kick"

          return data

    elif any(word in data for word in Others):

          data = "Others"

          return data  



Ftest_Df['Event'] = Ftest_Df['Event'].apply(lambda x : assign_feature_problem(x))

Ftest_Df['Event'].unique()
df = Ftest_Df.copy()

# df['Non_Alignment_Score'] = df['Non_Alignment_Score']

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '0.0', 'Aligned (0 Deg)', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '0.5',  '0-45 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '1.5', '45-90 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '2.5',  '90-135 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '1.0',  '0-45 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '2.0', '45-90 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '3.0',  '90-135 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '3.5', '135-180 deg', df['Non_Alignment_Score'])

df['Non_Alignment_Score'] = np.where(df['Non_Alignment_Score'] == '4.0', 'Opposite Aligned (180 deg)', df['Non_Alignment_Score'])



################################################################################################################################

                                                # Non-Alignment Score : For Injured Players Only

################################################################################################################################



import plotly.express as px

    

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="Non_Alignment_Score", histnorm='percent',

                   title='<b>Angular differences b/w Player Direction and Player Orientation at Injury Minute</b>', 

                opacity=0.8, color_discrete_sequence=['paleturquoise'])



fig4b = go.Figure(data=data, layout=layout)





fig4b.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig4b.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Non-Alignment Angle</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig4b.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig4b.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig4b.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Blue')

fig4b.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)





################################################################################################################################

                                                # Average Speed

################################################################################################################################



import plotly.express as px

    

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="Avg_Spd", histnorm='percent',

                   title='<b>Distribution of Average_Speed at Injury Minute : For Injured players</b>', 

                opacity=0.8, color_discrete_sequence=['rosybrown'])



fig5 = go.Figure(data=data, layout=layout)





fig5.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig5.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Avg_Speed (Yards/Second)</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig5.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig5.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig5.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Brown')

fig5.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



################################################################################################################################

                                                # Event When Injured

################################################################################################################################



import plotly.express as px

    

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="Event", histnorm='percent',

                   title='<b>Distribution of Event at Injury Minute : For Injured players</b>', 

                opacity=0.8, color_discrete_sequence=['orchid'])



fig6 = go.Figure(data=data, layout=layout)





fig6.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig6.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Event Type</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig6.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig6.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig6.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Red')

fig6.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)





################################################################################################################################

                                                # player position When Injured

################################################################################################################################

import plotly.express as px

    

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="PositionGroup", histnorm='percent',

                   title='<b>Distribution of Player Position at Injury Minute : For Injured players</b>', 

                opacity=0.8, color_discrete_sequence=['orangered'])



fig7 = go.Figure(data=data, layout=layout)





fig7.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig7.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Player Position</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig7.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig7.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig7.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Brown')

fig7.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)







################################################################################################################################

                                                # player position Retained when Injured

################################################################################################################################

import plotly.express as px

    

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="PositionRetained", histnorm='percent',

                   title='<b>Roster Position retained at Injury Time</b>', 

                opacity=0.8, color_discrete_sequence=['crimson'])



fig8 = go.Figure(data=data, layout=layout)





fig8.update_layout(

    autosize=False,

    width=500,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig8.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Y: Yes, N:No</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig8.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig8.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig8.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Black')

fig8.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)





fig4b.show()

fig5.show()

fig6.show()

fig7.show()

fig8.show()



F_Modeling_Df = Ftest_Df.copy()

F_Modeling_Df = F_Modeling_Df.drop(['RosterPosition', 'PlayKey',

                                   'Seconds', 'PlayerGamePlay', 'Distance'], axis=1)

F_Modeling_Df = Ftest_Df.copy()

F_Modeling_Df = F_Modeling_Df.drop(['RosterPosition', 'PlayKey',

                                   'Seconds', 'PlayerGamePlay', 'Distance'], axis=1)



F_Modeling_Df = F_Modeling_Df.sort_values(['GameID'], ascending=[True])



# The min value in player day is -62. Hence shifting the timeline to make it timeline in positve

F_Modeling_Df['PlayerDay'] = F_Modeling_Df['PlayerDay'] + 63



# Also finding the start day of the second season ad 336th day in the new Timeline.  

F_Modeling_Df['PlayerDay'] = np.where(F_Modeling_Df['PlayerDay'] > 336, F_Modeling_Df['PlayerDay']-336, F_Modeling_Df['PlayerDay'])





# Setting up a new feature to calculate the Resting period between the next game

F_Modeling_Df['Rest_Days'] = 0



for i in range(len(F_Modeling_Df)-1):

        if F_Modeling_Df['GameID'][i+1] != F_Modeling_Df['GameID'][i]:



            F_Modeling_Df['Rest_Days'][i+1] = F_Modeling_Df['PlayerDay'][i+1] - F_Modeling_Df['PlayerDay'][i]



        else:

            F_Modeling_Df['Rest_Days'][i+1] = F_Modeling_Df['Rest_Days'][i] 



# Hence adjusting the season 2 timeline

F_Modeling_Df['Rest_Days'] = np.where(F_Modeling_Df['Rest_Days'] < 0, 0, F_Modeling_Df['Rest_Days'])



# Creating the Player Stress Factor Column

F_Modeling_Df['D1'] = np.where((F_Modeling_Df['Rest_Days']== 0) , 'Fresh','')

F_Modeling_Df['D2'] = np.where((F_Modeling_Df['Rest_Days']> 0)& (F_Modeling_Df['Rest_Days']<4.5) , 'High','')

F_Modeling_Df['D3'] = np.where((F_Modeling_Df['Rest_Days']> 4.5)& (F_Modeling_Df['Rest_Days']<9.5) , 'Medium','')

F_Modeling_Df['D4'] = np.where((F_Modeling_Df['Rest_Days']> 9.5)& (F_Modeling_Df['Rest_Days']<16.6) , 'Normal','')

F_Modeling_Df['D5'] = np.where((F_Modeling_Df['Rest_Days']> 16.5)& (F_Modeling_Df['Rest_Days']<30.5) , 'Low','')

F_Modeling_Df['D6'] = np.where((F_Modeling_Df['Rest_Days']> 30.5) , 'Fresh','')





F_Modeling_Df['Player_Stress'] =  (F_Modeling_Df['D1'] + F_Modeling_Df['D3'] + F_Modeling_Df['D4'] + F_Modeling_Df['D5'] +

                           F_Modeling_Df['D6']  + F_Modeling_Df['D2'] )



F_Modeling_Df = F_Modeling_Df.drop(columns = ['D1','D2', 'D3', 'D4', 'D5', 'D6','PlayerDay','Rest_Days', 'PlayerKey', 'GameID', 'PlayerGame'])

F_Modeling_Df
df = F_Modeling_Df.copy()

################################################################################################################################

                                                # Player Body_Stress at time of Injury

################################################################################################################################

import plotly.express as px

    

df = df[df['Injured']== 1]



data= px.histogram(df, x="Player_Stress", histnorm='percent',

                   title='<b>Distribution of Player Stress at Injury Game : For Injured players</b>', 

                opacity=0.8, color_discrete_sequence=['mediumspringgreen'])



fig9 = go.Figure(data=data, layout=layout)





fig9.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4),

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)'

)



fig9.update_layout(

#     title="Plot Title",

    title_x=0.5,

    xaxis_title=" <b>Player Stress</b>",

    yaxis_title="<b>Distribution Percentage</b>",

    yaxis_range=[0,100],

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig9.update_xaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig9.update_yaxes(ticks="outside", tickwidth=2, tickcolor='crimson', ticklen=5)

fig9.update_xaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=True, gridcolor='Green')

fig9.update_yaxes(showline=True, linewidth=2, linecolor='black', zeroline=True, showgrid=False)



F_Modeling_Df_Injured = F_Modeling_Df[F_Modeling_Df['Injured']==1]

F_Modeling_Df_non_Injured = F_Modeling_Df[F_Modeling_Df['Injured']==0].sample(n = 5000)

U_Sample_df = pd.concat([F_Modeling_Df_Injured, F_Modeling_Df_non_Injured], ignore_index=True)

U_Sample_df



X = U_Sample_df.drop(columns=['Injured','Injury_Impact', 'BodyPart', 'PlayType', 'StadiumType', 'PositionRetained']).copy()

y = U_Sample_df['Injured']

categorical_feature_mask = X.dtypes==object

categorical_cols = X.columns[categorical_feature_mask].tolist()

le = LabelEncoder()

X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype('str')))

X.tail()
X = F_Modeling_Df.drop(columns=['Injured','Injury_Impact', 'BodyPart', 'PlayType', 'StadiumType', 'PositionRetained']).copy()

y = F_Modeling_Df['Injured']

# Categorical boolean mask

categorical_feature_mask = X.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = X.columns[categorical_feature_mask].tolist()

# instantiate labelencoder object

le = LabelEncoder()

# apply le on categorical feature columns

X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype('str')))





from sklearn.linear_model.base import MultiOutputMixin

from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN)

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.

sm = SMOTE(sampling_strategy='minority', random_state=7)

# Fit the model to generate the data.

oversampled_trainX, oversampled_trainY = sm.fit_sample(X, y)

O_Sample_df = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)

O_Sample_df
### For UnderSampling



## MULTI LABEL

# instantiate OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

Multilabel = OneHotEncoder(categories='auto', sparse=False)

# apply OneHotEncoder on categorical feature columns

U_X_Multilabel = Multilabel.fit_transform(X) # It returns an numpy array





## BINARY LABELING

# instantiate OneHotEncoder

BinaryLabel = OneHotEncoder(categories='auto', sparse=True)

# apply OneHotEncoder on categorical feature columns

U_X_Binarylabel = BinaryLabel.fit_transform(X) # It returns an numpy array

U_X_Binarylabel
import seaborn as sns

sns.set(style="white")



# Compute the correlation matrix

corr = X.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
fig = plt.figure(figsize = (15,20))

ax = fig.gca()

X.hist(ax = ax)
x = U_X_Multilabel



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, shuffle=True)



classifiers=[]

model1 = xgboost.XGBClassifier()

classifiers.append(model1)

# model2 = svm.SVC()

# classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()

classifiers.append(model3)

model4 = RandomForestClassifier()

classifiers.append(model4)





for clf in classifiers:

    clf.fit(X_train, y_train)

    y_pred= clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy of %s is %s"%(clf, acc))

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix of %s is %s"%(clf, cm))
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import f1_score



x = U_X_Binarylabel



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, shuffle=True)



classifiers=[]

model1 = xgboost.XGBClassifier()

classifiers.append(model1)

# model2 = svm.SVC()

# classifiers.append(model2)

model3 = tree.DecisionTreeClassifier()

classifiers.append(model3)

model4 = RandomForestClassifier()

classifiers.append(model4)





for clf in classifiers:

    clf.fit(X_train, y_train)

    y_pred= clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy of %s is %s"%(clf, acc))

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix of %s is %s"%(clf, cm))

    f1 = f1_score(y_test, y_pred, average='micro')

    print("F1 Score of %s is %s"%(clf, f1))
X=X

y=y



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)



rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs= -1)

rf_model = rf.fit(X_train,y_train)

sorted(zip((rf.feature_importances_)*100,X_train.columns), reverse = True)[0:7]