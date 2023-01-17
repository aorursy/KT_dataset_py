#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQq3EnNdpTMivXLCfUPaYMYjRCaF-nsiPDSG603Yyr0gAQhSI4s&usqp=CAU',width=400,height=400)
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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/model-answers/Extracorporeal_membrane_oxygenation_(ECMO)_outcomes.csv")

df.head()
px.histogram(df, x='question', color='answer')
px.histogram(df, x='end_score', color='start_score')
sns.countplot(df["journal"])

plt.xticks(rotation=90)

plt.show()
fig = px.bar(df, x= "context", y= "answer")

fig.show()
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

    title='Covid-19 Mortality on ECMO',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="journal")
ecmo = df.groupby(["start_score"])["end_score"].sum().reset_index().sort_values("start_score",ascending=False).reset_index(drop=True)

ecmo
labels = ["start_score","end_score"]

values = ecmo.loc[0, ["start_score","end_score"]]

df = px.data.tips()

fig = px.pie(ecmo, values=values, names=labels, color_discrete_sequence=['royalblue','darkblue','lightcyan'])

fig.update_layout(

    title='Covid-19 Mortality on ECMO : '+str(ecmo["end_score"][0]),

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=ecmo['start_score'][0:10], y=ecmo['end_score'][0:10],

            text=ecmo['end_score'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Covid-19 Mortality on ECMO',

    xaxis_title="start_score",

    yaxis_title="end_score",

)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS6x-hTy34QAXGJwsypfTyilCtOfMd0MuX2fTQ8BdwRhaXa2Nuh&usqp=CAU',width=400,height=400)
fig = go.Figure(data=[go.Scatter(

    x=ecmo['start_score'][0:10],

    y=ecmo['end_score'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Covid-19 Mortality on ECMO',

    xaxis_title="start_score",

    yaxis_title="end_score",

)

fig.show()
fig = px.pie(ecmo, values=ecmo['end_score'], names=ecmo['start_score'],

             title='Covid-19 Mortality on ECMO',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSMTQUgDt4JSQftbHcnLzrKirw61Jxa81MHWsb3pU58CxxkfmqO&usqp=CAU',width=400,height=400)