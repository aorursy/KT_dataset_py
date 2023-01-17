#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQO-7yx1GEjQ2Lt7kx0dmjCi9UdXnJvW4gf65PTy7BBchl517ni',width=400,height=400)
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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19-measures-in-bulgaria/covid-19 Bulgaria measures.csv")

df.head()
df = df.rename(columns={'event info':'event'})
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='Reds')

plt.show()
cnt_srs = df['event'].value_counts().head()

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

    title='Event Info Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="event")
cnt_srs = df['restriction'].value_counts().head()

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

    title='Restriction Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="restriction")
Restrictions = df.groupby(["restriction"])["event"].sum().reset_index().sort_values("restriction",ascending=False).reset_index(drop=True)

Restrictions
fig = go.Figure(data=[go.Bar(

            x=Restrictions['event'][0:10], y=Restrictions['restriction'][0:10],

            text=Restrictions['restriction'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Bulgarian Restrictions',

    xaxis_title="Events",

    yaxis_title="Restrictions",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=Restrictions['event'][0:10],

    y=Restrictions['restriction'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Events Restrictions',

    xaxis_title="Events",

    yaxis_title="Restrictions",

)

fig.show()
labels = ["event","restriction"]

values = Restrictions.loc[0, ["event","restriction"]]

df = px.data.tips()

fig = px.pie(Restrictions, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','lightcyan'])

fig.update_layout(

    title='Event Restrictions : '+str(Restrictions["restriction"][0]),

)

fig.show()
fig = px.pie(Restrictions, values=Restrictions['restriction'], names=Restrictions['event'],

             title='Events Restrictions',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR-w3TCX1YANH7DLF3L9MRDZn1IvanBo6uvho_shBn6xXGY_vi5',width=400,height=400)