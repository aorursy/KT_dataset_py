import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
import base64
init_notebook_mode()

df = pd.read_csv('../input/world-cup-penalty-shootouts/WorldCupShootouts.csv')
def show_shots(df, x, y, size, size_max, hover_name, hover_data, color, title):
    init_notebook_mode()
    fig = px.scatter(df, 
                 x=x,
                 y=y,  
                 size= size,
                 size_max = size_max,
                 color = color,
                 hover_name = hover_name,
                 hover_data = hover_data,
                 range_x = (0,900),
                 range_y = (581,0),
                 width = 900,
                 height = 581,
                 labels = {x:'', y:''})
    image_filename = "../input/goal-image/goal.png"
    plotly_logo = base64.b64encode(open(image_filename, 'rb').read())
    fig.update_layout(xaxis_showgrid=False, 
                    yaxis_showgrid=False,
                    xaxis_showticklabels=False,
                    yaxis_showticklabels=False,
                    title= title,
                    images= [dict(
                    source='data:image/png;base64,{}'.format(plotly_logo.decode()),
                    xref="paper", yref="paper",
                    x=0, y=1,
                    sizex=1, sizey=1,
                    xanchor="left",
                    yanchor="top",
                    sizing = 'stretch',
                    layer="below")])
    iplot(fig)
shot_coords = {
    1:[216,150],
    2:[448,150],
    3:[680,150],
    4:[216,250],
    5:[448,250],
    6:[680,250],
    7:[216,350],
    8:[448,350],
    9:[680,350]
}

df_target = df[df.OnTarget == 1]

df_target['Zone_x'] = df_target['Zone'].apply(lambda x: shot_coords[int(x)][0])
df_target['Zone_y'] = df_target['Zone'].apply(lambda x: shot_coords[int(x)][1])

df_zone = pd.DataFrame(df_target.groupby(['Zone','Zone_x', 'Zone_y']).size()).reset_index()
df_zone.rename(columns = {0:'Number of Shots'}, inplace= True)

show_shots(df_zone, 'Zone_x', 'Zone_y', 'Number of Shots', 70, 'Zone', ['Zone', 'Number of Shots'], 'Number of Shots', 'Shot Location (On Target Shots)')
df_Offtarget = df[df.OnTarget == 0]

df_Offtarget['Zone_x'] = df_Offtarget['Zone'].apply(lambda x: shot_coords[int(x)][0])
df_Offtarget['Zone_y'] = df_Offtarget['Zone'].apply(lambda x: shot_coords[int(x)][1])

df_zone = pd.DataFrame(df_Offtarget.groupby(['Zone','Zone_x', 'Zone_y']).size()).reset_index()
df_zone.rename(columns = {0:'Number of Shots'}, inplace= True)

show_shots(df_zone, 'Zone_x', 'Zone_y', 'Number of Shots', 70, 'Zone', ['Zone', 'Number of Shots'], 'Number of Shots', 'Intended Shot Location (Off Target Shots)')


df_zone = pd.DataFrame(df_target.groupby(['Zone','Zone_x', 'Zone_y', 'Goal']).size()).reset_index()
df_zone.rename(columns = {0:'Number of Shots'}, inplace= True)

show_shots(df_zone, 'Zone_x', 'Zone_y', 'Number of Shots', 70, 'Zone', ['Zone', 'Number of Shots'], 'Goal', 'Shot Success by Zone (On Target Shots)')

for i in range(df_zone.shape[0]):
    zone = df_zone.loc[i, 'Zone']
    df_goal = df_zone[df_zone.Zone == zone]
    tot = df_goal['Number of Shots'].sum()
    goal = df_goal[df_goal.Goal == 1.0]['Number of Shots'].sum()
    df_zone.loc[i, 'Success Percentage'] = goal/tot

df_zone = df_zone[df_zone.Goal == 1.0]
show_shots(df_zone, 'Zone_x', 'Zone_y', 'Number of Shots', 70, 'Zone', ['Zone', 'Number of Shots', 'Success Percentage'], 'Success Percentage', 'Shot Success by Zone (On Target Shots)')

keeper_coords = {
    'L':[216,250],
    'C':[448,250],
    'R':[680,250],
}

df.dropna(inplace=True)

df.replace('l', 'L', inplace=True)
df['Keeper_x'] = df['Keeper'].apply(lambda x: keeper_coords[x][0])
df['Keeper_y'] = df['Keeper'].apply(lambda x: keeper_coords[x][1])

df_keeper = pd.DataFrame(df.groupby(['Keeper','Keeper_x', 'Keeper_y']).size()).reset_index()
df_keeper.rename(columns = {0:'Number of Shots'}, inplace= True)

show_shots(df_keeper, 'Keeper_x', 'Keeper_y', 'Number of Shots', 70, 'Keeper', ['Keeper', 'Number of Shots'], 'Number of Shots', 'Keeper Location')
