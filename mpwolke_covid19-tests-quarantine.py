#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR3ew-G2v6y5VV1dicTYu1jCTyJxoTydXtByLh7n2vEyRukr4kv',width=400,height=400)
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

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQMxIOqoz2bUEZUrlGu-f0lrTNI_rUTRp2MzlxMY7ZBEdjL_z8j',width=400,height=400)
df = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

df.head()
cnt_srs = df['medianage'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'purples',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 Median Age Patients',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="medianage")
cnt_srs = df['schools'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'purples',

        reversescale = True

    ),

)



layout = dict(

    title='Schools affected by Covid-19',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="schools")
cnt_srs = df['quarantine'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'purples',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 Quarantines',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="quarantines")
cnt_srs = df['restrictions'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'purples',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 Restrictions',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="restrictions")
cnt_srs = df['urbanpop'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'purples',

        reversescale = True

    ),

)



layout = dict(

    title='Urban Population',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="urbanpop")
fig = px.pie( values=df.groupby(['testpop']).size().values,names=df.groupby(['testpop']).size().index)

fig.update_layout(

    title = "Covid-19 Populational Tests",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df[df.testpop.notna()],x="testpop",marginal="box",nbins=10)

fig.update_layout(

    title = "Covid-19 Populational tests",

    xaxis_title="tests",

    yaxis_title="Number of tests",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)
sns.countplot(df["quarantine"])

plt.xticks(rotation=90)

plt.show()
sns.countplot(df["schools"])

plt.xticks(rotation=90)

plt.show()
Quarantine = df.groupby(["quarantine"])["urbanpop"].sum().reset_index().sort_values("quarantine",ascending=False).reset_index(drop=True)

Quarantine
fig = go.Figure(data=[go.Bar(

            x=Quarantine['urbanpop'][0:10], y=Quarantine['quarantine'][0:10],

            text=Quarantine['quarantine'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Urban Population Quarantine ',

    xaxis_title="Urban Population",

    yaxis_title="Quarantine",

)

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=Quarantine['urbanpop'][0:10],

    y=Quarantine['quarantine'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Urban Population Quarantine',

    xaxis_title="Urban Population",

    yaxis_title="Quarantine",

)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRFVbfwKON5v7sbjp5E9CifMKw09B89cPFWmpRRTsPOdthGQSE-',width=400,height=400)
df_aux = df[df.urbanpop.notna()]

df_aux=df_aux[df_aux.density.notna()]

#df_patients_aux=df_patients_aux.Description.notna()

fig = px.histogram(df_aux,x="urbanpop",color="density",marginal="box",opacity=1,nbins=10)

fig.update_layout(

    title = "Density of Urban Population",

    xaxis_title="Urban Population",

    yaxis_title="Density",

    barmode="group",

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    ))

py.iplot(fig)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTpjmhkDxVwR9hkSwqvUmdRNqNGvEoyNISqP3sKj8LDZw1InVj4',width=400,height=400)