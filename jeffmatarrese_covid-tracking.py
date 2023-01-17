import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import json, requests
us_stats = requests.get("https://covidtracking.com/api/us/daily")
state_stats = requests.get("https://covidtracking.com/api/states/daily")

df_us = json_normalize(us_stats.json())
df_states = json_normalize(state_stats.json())
def convertDT(df):
    df['DateTime'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    return df

df_us = convertDT(df_us)
df_states = convertDT(df_states)
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
us_plot = px.line(df_us, x=df_us['DateTime'], y=df_us['positive'], title="Total confirmed US COVID cases", height=500, template='plotly_white')

states_plot = px.line(df_states, x=df_states['DateTime'], y=df_states['positive'], color=df_states['state'],
                        title="Total confirmed COVID cases by state", height=800, template='plotly_white')

us_plot.show()
states_plot.show()
us_death_plot = px.line(df_us, x=df_us['DateTime'], y=df_us['death'], title="Total deaths reported in US", template='plotly_white')

states_death_plot = px.line(df_states, x=df_states['DateTime'], y=df_states['death'], color=df_states['state'],
                        title="Total confirmed COVID cases by state", template='plotly_white')

us_death_plot.show()
states_death_plot.show()