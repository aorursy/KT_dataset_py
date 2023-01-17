import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sb

import matplotlib.pyplot as plt

from matplotlib import rcParams







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/world-happiness/2019.csv')
data.head(3)
data.info()
data.describe()
data.columns
data.rename(columns={"Country or region":"Country"}, inplace=True)
score = data[['Country', 'Score']].groupby('Country').sum().reset_index()

score.sort_values(by='Score', ascending=False).head(20).style.background_gradient(cmap='Purples', subset=['Score'])
fig = px.bar(score.sort_values(by='Score', ascending=False).head(20), 

             'Score', 'Country', 

             color='Country', orientation='h', text='Score')





fig.update_layout(title={

                  'text': "World Happiness Report",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                  height=600,

                  xaxis_title="Score", 

                  yaxis_title="Country",

                  showlegend=False,

                  template='ggplot2')



fig.show()
fig = px.pie(score.sort_values(by='Score', ascending=False).head(5), 

       'Country', 'Score', 

       color_discrete_sequence=px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "World Happiness Report (Top 5)",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                   showlegend=False,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='label+value', pull=[0.2])



fig.data[0].marker.line.width = 2

fig.data[0].marker.line.color = "black"



fig.show()
hle = data[['Healthy life expectancy', 'Country']].groupby('Country').sum().reset_index()

hle.sort_values(by='Healthy life expectancy', ascending=False).head(20).style.background_gradient(cmap='Blues', subset=['Healthy life expectancy'])
fig = px.bar(hle.sort_values(by='Healthy life expectancy', ascending=False).head(20), 

             'Country', 'Healthy life expectancy', 

             color='Country', text='Healthy life expectancy')





fig.update_layout(title={

                  'text': "Healthy life expectancy",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                  height=600,

                  xaxis_title="Score", 

                  yaxis_title="Country",

                  showlegend=False,

                  template='ggplot2')



fig.show()