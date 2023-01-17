# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

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
df = pd.read_excel('/kaggle/input/corona-virus/route.xls')

df.head()
df.dtypes
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set3',

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(df['city'])
fig = px.pie( values=df.groupby(['visit']).size().values,names=df.groupby(['visit']).size().index)

fig.update_layout(

    title = "Visit",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
df1 = pd.read_excel('/kaggle/input/corona-virus/patient.xls')

df1.head()
df1.dtypes
fig = px.pie( values=df1.groupby(['infection_reason']).size().values,names=df1.groupby(['infection_reason']).size().index)

fig.update_layout(

    title = "Infection Reason",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
cnt_srs = df1['birth_year'].value_counts().head()

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

    title='',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="birth_year")
fig = px.histogram(df1[df1.infection_reason.notna()],x="infection_reason",marginal="box",nbins=10)

fig.update_layout(

    title = "Infection Reason",

    xaxis_title="infection-reason",

    yaxis_title="disease",

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
df2 = pd.read_excel('/kaggle/input/corona-virus/trend.xls')

df2.head()
df2.dtypes
fig = px.pie( values=df2.groupby(['pneumonia']).size().values,names=df2.groupby(['pneumonia']).size().index)

fig.update_layout(

    title = "Pneumonia",

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
fig = px.histogram(df2[df2.coronavirus.notna()],x="coronavirus",marginal="box",nbins=10)

fig.update_layout(

    title = "Coronavirus",

    xaxis_title="coronavirus",

    yaxis_title="date",

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