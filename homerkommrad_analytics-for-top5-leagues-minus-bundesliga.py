import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')



pes = pd.read_csv("../input/efootball-pes-2020-all-players-csv/deets-updated.csv")

pd.set_option('display.max_rows',20000, 'display.max_columns',20000)

pes=pes[pes['league'].isin(["Ligue 1 Conforama","Serie A TIM","English League","Spanish League","Liga NOS"])]



pes.head()



print("number of players:",len(pes.index))
pes_bl_or_gold=pes[pes['ball_color'].isin(['black','gold'])]

pes_player_detail= pes_bl_or_gold.groupby(['league','ball_color'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['league','ball_color'],values='Player_Count')

fig.update_layout(

    title='Black Ball and Gold Ball Quality Players distribution by Leagues')

fig.show()


pes_player_detail= pes.groupby(['league','registered_position'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['league','registered_position'],values='Player_Count')

fig.update_layout(

    title='League distribution of Playing Positions')

fig.show()
pes_league_playing_style= pes.groupby(['league','playing_style'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_league_playing_style,path=['league','playing_style'],values='Player_Count')

fig.update_layout(

    title='League distribution of Playing Styles')

fig.show()
pes_cfs=pes[pes["registered_position"] =="CF"]

pes_cfs_style= pes_cfs.groupby(['league','playing_style'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_cfs_style,path=['league','playing_style'],values='Player_Count')

fig.update_layout(

    title='League distribution of CF Playing Styles')

fig.show()
pes_midfields=pes[pes["registered_position"].isin(["DMF","CMF","LMF","RMF","AMF"])]

pes_midfields_style= pes_midfields.groupby(['league','playing_style'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_midfields_style,path=['league','playing_style'],values='Player_Count')

fig.update_layout(

    title='League distribution of Midfields Playing Styles')

fig.show()
pes_younglings=pes[pes['age'] < 24]

#pes_younglings=pes_younglings[pes_younglings['ball_color'] != "white"]



pes_player_detail= pes_younglings.groupby(['league','ball_color'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['league','ball_color'],values='Player_Count')

fig.update_layout(

    title='U23 players Ball Color distribution by Leagues')

fig.show()