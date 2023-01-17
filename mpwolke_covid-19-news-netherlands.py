#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTFQKZhKbk1LZBZcHIbLZuwadUK38gUG8kw1ldV0N9uK7odr0a0&usqp=CAU',width=400,height=400)
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
df = pd.read_csv('../input/covid19newsnetherlands/covid-19_news.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df, 

             x='category', y='time', color_discrete_sequence=['Orange'],

             title='Covid-19 News Netherland', text='category')

fig.show()
fig = px.bar(df, x= "category", y= "link")

fig.show()
cnt_srs = df['category'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Oranges',

        reversescale = True

    ),

)



layout = dict(

    title='Covid-19 News Netherlands ',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="category")
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.category)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
fig = px.line(df, x="category", y="link", 

              title="Covid-19 News Netherlands")

fig.show()
import networkx as nx

df1 = pd.DataFrame(df['category']).groupby(['category']).size().reset_index()



G = nx.from_pandas_edgelist(df1, 'category', 'category', [0])

colors = []

for node in G:

    if node in df["category"].unique():

        colors.append("green")

    else:

        colors.append("lightgreen")

        

nx.draw(nx.from_pandas_edgelist(df1, 'category', 'category', [0]), with_labels=True, node_color=colors)
fig = px.density_contour(df, x="category", y="link", color_discrete_sequence=['purple'])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSm3lVECP1yqIW-rwDX1F1_9zh5YHR49Cv4c7jtAVv7RI4fhqBY&usqp=CAU',width=400,height=400)