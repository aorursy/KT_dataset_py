import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')



pes = pd.read_csv("../input/efootball-pes-2019-all-players/pes2019-all-players.v2.csv")

pd.set_option('display.max_rows',20000, 'display.max_columns',20000)

pes.head()





pes.isnull().sum()



pes_sum_or = pes.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['overall_rating']=pes_count_or['Total Players']

Worlds_Best_Team=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Worlds_Best_Team['team_name'][:40], y=Worlds_Best_Team['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,0,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.update_layout(title_text='Top 40 Worlds Best Football Team')

fig.show()

fig = px.pie(pes, names='ball_color',hole=.3)

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=0.9)))

fig.update_layout(

    title='Player Count Divided in from White(1-Star Player) to Black(5-Star Player)')

fig.show()


pes_bl_or_gold=pes[pes['ball_color'].isin(['black','gold'])]

pes_player_detail= pes_bl_or_gold.groupby(['league','ball_color'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['league','ball_color'],values='Player_Count')

fig.update_layout(

    title='Black Ball and Gold Ball Quality Players distribution by Leagues')

fig.show()
pes_europe = pes[pes['region']=='Europe']

pes_player_detail= pes_europe.groupby(['region','nationality','registered_position'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position'],values='Player_Count')

fig.update_layout(

    title='European Player with Country,Playing Position and for the CLUB they play for')

fig.show()
pes_europe = pes[pes['region']=='South America']

pes_player_detail= pes_europe.groupby(['region','nationality','registered_position'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position'],values='Player_Count')

fig.update_layout(

    title='South American Player with Country,Playing Position')

fig.show()
pes_europe = pes[pes['region']=='Africa']

pes_player_detail= pes_europe.groupby(['region','nationality','registered_position'])['league'].count().reset_index(name='Player_Count')

fig = px.sunburst(pes_player_detail,path=['region','nationality','registered_position'],values='Player_Count')

fig.update_layout(

    title='African Player with Country,Playing Position')

fig.show()
fig = px.histogram(pes,x='region')

fig.update_traces(marker_color='rgb(80,10,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(

    title='Continentwise Players in PES2020')

fig.show()
pes_sum_or = pes.groupby('league')['overall_rating'].sum().reset_index()

pes_count_or = pes.groupby('league')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Z=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['league','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Z['league'], y=Z['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(250,10,50)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Best League in the world based on Overall Player Rating')

fig.show()
pes['age'] = pd.to_numeric(pes['age'], errors='coerce')

fig = px.histogram(pes,x='age')

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(size=14))

fig.update_layout(

    title='Agewise Count of Players')

fig.show()
fig = px.box(pes, x=pes['league'], y=pes['age'])

fig.update_traces(marker_color='rgb(255,140,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=12))

fig.update_layout(

    title='Age Distribution of Players with a Club in PES2019')

fig.show()
fig = px.histogram(pes,x='foot')

fig.update_traces(marker_color='rgb(10,255,10)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=0, tickfont=dict(size=14))

fig.update_layout(

    title='Foot Prefferece by the Players in PES2019')

fig.show()
fig = px.pie(pes,names='playing_style', title='Player Playing Style in PES2019')

fig.update_traces(hole=.3, hoverinfo="label+percent+name")

fig.show()
fig = px.histogram(pes,x='registered_position')

fig.update_traces(marker_color='rgb(10,10,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))



fig.update_layout(

    title='Registered Position by the Players in PES2019')

fig.show()
fig = px.box(pes, y=pes['overall_rating'], x=pes['age'])

fig.update_layout(

    title='Agewise Growth and Decline of Player')

fig.update_traces(marker_color='rgb(110,255,110)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=10))

fig.show()
pes_best_players = pes.iloc[pes.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

pes_best_players
English_League = pes[pes['league']=='English League']

English_League_XI = English_League.loc[English_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

English_League_XI
pes_english_league = pes[pes['league']=='English League']

pes_sum_or = pes_english_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_english_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

English_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=English_League['team_name'], y=English_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(0,102,255)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of English League Team')

fig.show()
Spanish_League = pes[pes['league']=='Spanish League']

Spanish_League_XI = Spanish_League.loc[Spanish_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

Spanish_League_XI
pes_spanish_league = pes[pes['league']=='Spanish League']

pes_sum_or = pes_spanish_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_spanish_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Spanish_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Spanish_League['team_name'], y=Spanish_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,64,0)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of Spanish League Team')

fig.show()
Italian_League = pes[pes['league']=='Italian League']

Italian_League_XI = Italian_League.loc[Italian_League.groupby(['registered_position'])['overall_rating'].idxmax()][['name','nationality','team_name','registered_position','overall_rating']]

Italian_League_XI
pes_italian_league = pes[pes['league']=='Italian League']

pes_sum_or = pes_italian_league.groupby('team_name')['overall_rating'].sum().reset_index()

pes_count_or = pes_italian_league.groupby('team_name')['overall_rating'].count().reset_index(name='Total Players')

pes_sum_or['Average overall_rating'] = pes_sum_or['overall_rating']/pes_count_or['Total Players']

pes_sum_or['Total Players']=pes_count_or['Total Players']

Italian_League=pes_sum_or.sort_values('Average overall_rating',ascending=False)[['team_name','Average overall_rating']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=Italian_League['team_name'], y=Italian_League['Average overall_rating'],

                    mode='lines+markers',name='Total Cases'))

fig.update_traces(marker_color='rgb(255,0,191)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_xaxes(tickangle=270, tickfont=dict(size=14))

fig.update_layout(title_text='Overall Rating of Italian League Team')

fig.show()