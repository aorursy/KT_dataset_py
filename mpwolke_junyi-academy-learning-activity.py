#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/learning-activity-public-dataset-by-junyi-academy/Info_Content.csv")

df.head()
df['content_pretty_name'].value_counts()
fig = px.bar(df,

             y='difficulty',

             x='content_pretty_name',

             orientation='h',

             color='subject',

             title='Junyi Academy learning activity',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.line(df, x="content_pretty_name", y="difficulty", color_discrete_sequence=['green'], 

              title="Junyi Academy learning activity")

fig.show()
fig = px.bar(df, x= "learning_stage", y= "difficulty", color_discrete_sequence=['crimson'],)

fig.show()
cnt_srs = df['difficulty'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'viridis',

        reversescale = True

    ),

)



layout = dict(

    title='Junyi Academy learning activity',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="difficulty")
fig = px.bar(df[['content_pretty_name','difficulty']].sort_values('difficulty', ascending=False), 

                        y = "difficulty", x= "content_pretty_name", color='difficulty', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Junyi Academy learning activity")



fig.show()
plt.figure(figsize=(10,6))

plt.bar(df.difficulty, df.learning_stage,label="learning_stage")

plt.xlabel('difficulty')

plt.ylabel("learning_stage")

plt.legend(frameon=True, fontsize=25)

plt.title('Junyi Academy Learning Activity',fontsize=30)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('difficulty').size()/df['learning_stage'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
fig = go.Figure(data=[go.Scatter(

    x=df['content_pretty_name'][0:10],

    y=df['difficulty'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Junyi Academy learning activity',

    xaxis_title="Content Pretty Name",

    yaxis_title="Difficulty",

)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)