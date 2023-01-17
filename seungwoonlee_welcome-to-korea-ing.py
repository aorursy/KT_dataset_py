import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_gender = pd.read_csv('/kaggle/input/south-korea-visitors/Enter_korea_by_gender.csv')

df_purpose = pd.read_csv('/kaggle/input/south-korea-visitors/Enter_korea_by_purpose.csv')

df_age = pd.read_csv('/kaggle/input/south-korea-visitors/Enter_korea_by_age.csv')
df = pd.merge(df_gender, df_purpose, on=['date','nation', 'visitor', 'growth', 'share'])

df = pd.merge(df, df_age, on=['date','nation', 'visitor', 'growth', 'share'])
df.head()
#null data check

df.isnull().sum()
df[['year', 'month']] = df['date'].str.split('-', n=1, expand=True)
df.head()
df_19 = df[df['year'] == '2019']

df_19.groupby(['nation'])['visitor','male', 'female'].sum().reset_index().sort_values(by=['visitor'], ascending=False).head()
map_plot = dict(type = 'choropleth', 

                locations = df_19['nation'],

                locationmode = 'country names',

                z = df_19['visitor'], 

                text = df_19['nation'],

                colorscale = 'rdylgn', reversescale = True)



layout = dict(title = 'Welcome to Korea ',

              geo = dict(showframe = True,

                         projection = {'type': 'equirectangular'}))



fig = go.Figure(data = [map_plot], layout=layout)

fig.show()
fig = px.pie(df_19, values='visitor', names='nation', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
china = df[df['nation'] == 'China']

china.head()
fig = go.Figure()

con = (china['year'] == '2019')



fig.add_trace(go.Scatter(

    x=china.loc[con, 'month'],

    y=china["visitor"],

    mode="lines"))



fig.update_xaxes(tickvals = list(range(1, 13)), ticktext = list(range(1, 13)), title = 'Month')

fig.update_yaxes(title = 'Visitors')

fig.show()