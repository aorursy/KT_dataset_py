#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQXPzN9l7RlKE1d3roYp02850vcBoYxEOO7m6-R7UeMRR3JnmRM&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/model-answers/Effectiveness_of_movement_control_strategies.csv")

df.head()
px.histogram(df, x='question', color='answer')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/a94ba5ef-fb0b-4c21-a581-f3f2422359b9/gr1.jpg',width=400,height=400)
px.histogram(df, x='end_score', color='start_score')
seaborn.set(rc={'axes.facecolor':'cyan', 'figure.facecolor':'cyan'})

sns.countplot(df["journal"])

plt.xticks(rotation=90)

plt.show()
fig = px.bar(df, x= "context", y= "answer")

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/7c57b3fc-ed4c-42db-b5bf-566ec5ea9c2c/gr2.jpg',width=400,height=400)
cnt_srs = df['journal'].value_counts().head()

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

    title='Movement Control in Covid-19 time',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="journal")
move = df.groupby(["start_score"])["end_score"].sum().reset_index().sort_values("start_score",ascending=False).reset_index(drop=True)

move
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/8ecce1b6-a3c1-4d8e-b691-3b141b242250/gr3.jpg',width=400,height=400)
labels = ["start_score","end_score"]

values = move.loc[0, ["start_score","end_score"]]

df = px.data.tips()

fig = px.pie(move, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','lightcyan'])

fig.update_layout(

    title='Movement Control in Covid-19 time : '+str(move["end_score"][0]),

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=move['start_score'][0:10], y=move['end_score'][0:10],

            text=move['end_score'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Movement Control in Covid-19 time',

    xaxis_title="start_score",

    yaxis_title="end_score",

)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
fig = go.Figure(data=[go.Scatter(

    x=move['start_score'][0:10],

    y=move['end_score'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Movement Control in Covid-19 time',

    xaxis_title="start_score",

    yaxis_title="end_score",

)

fig.show()
fig = px.pie(move, values=move['end_score'], names=move['start_score'],

             title='Movement Control in Covid-19 time',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTHvFFkH7_TFWFwPDbRvn5xviUtsCB-HNO3jNmu__s-qrRLlu6B&usqp=CAU',width=400,height=400)