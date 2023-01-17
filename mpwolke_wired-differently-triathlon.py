#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSJ2Wb3joFI9c3cTrIlff9WmDsQxTlg288HfgXuDaAG8Mu_OVZZ&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTYtoVOKhAKPIr7xJzLa5QXgXa1bmfnKLyLynEq1HW1TGY-kEOz&usqp=CAU',width=400,height=400)
df = pd.read_csv("../input/ironman-703-gdynia-2019/gdynia5.csv")

df.head().style.background_gradient(cmap='prism')
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['Unnamed: 0', 'Swim'], as_index=False).Pos.sum()



fig = px.bar(plot_data, x='Unnamed: 0', y='Pos', color='Swim')

fig.show()
import plotly.express as px



plot_data = df.groupby(['Unnamed: 0'], as_index=False).Pos.sum()



fig = px.line(plot_data, x='Unnamed: 0', y='Pos')

fig.show()
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['Unnamed: 0', 'Pos'], as_index=False).Swim.sum()



fig = px.line(plot_data, x='Unnamed: 0', y='Swim', color='Pos')

fig.show()
cnt_srs = df['Swim'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Swimming',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Swim")
cnt_srs = df['Bike'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Biking',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Bike")
cnt_srs = df['Run'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Running',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Run")
ironman = df.groupby(["Pos"])["Unnamed: 0"].sum().reset_index().sort_values("Pos",ascending=False).reset_index(drop=True)

ironman
fig = go.Figure(data=[go.Bar(

            x=ironman['Pos'][0:10], y=ironman['Unnamed: 0'][0:10],

            text=ironman['Unnamed: 0'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Iron Man',

    xaxis_title="Pos",

    yaxis_title="Unnamed: 0",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=ironman['Pos'][0:10],

    y=ironman['Unnamed: 0'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Ironman Triathlon',

    xaxis_title="Pos",

    yaxis_title="Unnamed: 0",

)

fig.show()
labels = ["Pos","Unnamed: 0"]

values = ironman.loc[0, ["Pos","Unnamed: 0"]]

df = px.data.tips()

fig = px.pie(ironman, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','lightcyan'])

fig.update_layout(

    title='Ironman Triathlon : '+str(ironman["Unnamed: 0"][0]),

)

fig.show()
fig = px.pie(ironman, values=ironman['Unnamed: 0'], names=ironman['Pos'],

             title='Ironman Triathlon',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQGXZC9miK9HUVF-K1cVU7Z5roksX-BtelTLmbEQMn2Q9ZZljuG&usqp=CAU',width=400,height=400)