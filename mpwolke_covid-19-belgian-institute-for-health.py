#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://thumbs.dreamstime.com/b/belgium-flag-futuristic-digital-abstract-composition-coronavirus-inscription-covid-outbreak-concept-belgium-flag-173661510.jpg',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/uncover/regional_sources/the_belgian_institute_for_health/dataset-of-confirmed-cases-by-date-and-municipality.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df[['cases', 'nis5']].sort_values('nis5', ascending=False), 

             y="nis5", x="cases", color='cases', 

             log_y=True, template='ggplot2', title='Covid-19 Belgium')

fig.show()
fig = px.bar(df, 

             x='date', y='tx_descr_nl', color_discrete_sequence=['#D63230'],

             title='Covid-19 Belgium', text='tx_descr_nl')

fig.show()
#seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["tx_rgn_descr_nl"])

plt.xticks(rotation=90)

plt.show()
#seaborn.set(rc={'axes.facecolor':'#27F1E7', 'figure.facecolor':'#27F1E7'})

sns.countplot(df["tx_adm_dstr_descr_fr"])

plt.xticks(rotation=90)

plt.show()
px.histogram(df, x='nis5', color='cases')
fig = px.bar(df, x= "date", y= "nis5")

fig.show()
cnt_srs = df['tx_descr_nl'].value_counts().head()

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

    title='Covid-19 Belgium',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="tx_descr_nl")
cnt_srs = df['tx_rgn_descr_fr'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'ylorrd',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 Belgium',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="tx_rgn_descr_fr")
fig = px.pie(df, values=df['nis5'], names=df['date'],

             title='Covid-19 Belgium',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.line(df, x="date", y="nis5", 

              title="Covid-19 Belgium")

fig.show()
import networkx as nx

df1 = pd.DataFrame(df['cases']).groupby(['cases']).size().reset_index()



G = nx.from_pandas_edgelist(df1, 'cases', 'cases', [0])

colors = []

for node in G:

    if node in df["cases"].unique():

        colors.append("red")

    else:

        colors.append("lightgreen")

        

nx.draw(nx.from_pandas_edgelist(df1, 'cases', 'cases', [0]), with_labels=True, node_color=colors)
labels = df['cases'].value_counts().index

size = df['cases'].value_counts()

colors=['#BF3F3F','#B8BF3F']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Covid-19 Belgium', fontsize = 20)

plt.legend()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQpz7FXIA05tgX8PtlJKnjbaG2g80xxsrFA-1l56LCYocl6JQHl&usqp=CAU',width=400,height=400)