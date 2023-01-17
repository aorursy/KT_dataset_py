#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT7iabmPgXIYgiKjc4bNgLwzVzCCDCrexpP7kqoL1OPoMYRBGfW&usqp=CAU',width=400,height=400)
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
df = pd.read_csv('../input/cov19questions/CORD-19-research-challenge-tasks - Question_Task2.tsv', sep='\t', error_bad_lines=False)

df.head()
df = df.rename(columns={'Unnamed: 8':'unnamed', 'Main Question': 'mainQ', 'Cohort Size ': 'cohort', 'Study type': 'study', 'Answer': 'answer', 'Question': 'question'})
df_question = pd.DataFrame({

    'answer': df.answer,

    'question': df.question

})
fig = px.line(df_question, x="question", y="answer", 

              title="Covid-19 Questions and Answers ")

fig.show()
fig = px.bar(df, 

             x='question', y='answer', color_discrete_sequence=['#D63230'],

             title='Covid-19 Questions & Answers', text='answer')

fig.show()
fig = px.line(df, 

             x='question', y='answer', color_discrete_sequence=['#D63230'],

             title='Covid-19 Questions & Answers', text='answer')

fig.show()
import networkx as nx

df1 = pd.DataFrame(df['answer']).groupby(['answer']).size().reset_index()



G = nx.from_pandas_edgelist(df1, 'answer', 'answer', [0])

colors = []

for node in G:

    if node in df["answer"].unique():

        colors.append("blue")

    else:

        colors.append("lightgreen")

        

nx.draw(nx.from_pandas_edgelist(df1, 'answer', 'answer', [0]), with_labels=True, node_color=colors)
labels = df['answer'].value_counts().index

size = df['answer'].value_counts()

colors=['#3FBF3F','#7F3FBF']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Covid-19 Questions & Answers', fontsize = 20)

plt.legend()

plt.show()
cnt_srs = df['study'].value_counts().head()

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

    title='Covid-19 Questions & Answers',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="study")
cnt_srs = df['Country'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'magenta',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 Questions & Answers',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Country")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ48_H8JX5iLTOJKPmczR5p-qUm-Dj6sC2N6f6N2Ife0q143cGW&usqp=CAU',width=400,height=400)