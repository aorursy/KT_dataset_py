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
df = pd.read_csv('../input/broadway-weekly-grosses/synopses.csv', encoding='ISO-8859-2')

df.head()
fig = px.bar(df, 

             x='show', y='synopsis', color_discrete_sequence=['#a142f5'],

             title='On Broadway', text='synopsis')

fig.show()
fig = px.line(df, x="show", y="synopsis", color_discrete_sequence=['#a142f5'],

              title="On Broadway")

fig.show()
cnt_srs = df['synopsis'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='On Broadway',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="synopsis")
fig = px.density_contour(df, x="show", y="synopsis", color_discrete_sequence=['purple'])

fig.show()
fig = px.scatter(df.dropna(), x='show',y='synopsis', trendline="show", color_discrete_sequence=['purple'])

fig.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.show)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()