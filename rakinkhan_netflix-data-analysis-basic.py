# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (25,15)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'white', width = 1920,  height = 1080, max_words = 50).generate(' '.join(df.listed_in))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Genre',fontsize = 30)

plt.show()
import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True)
def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace





py.iplot([pie_plot(df['type'].value_counts(), ['#556B2F', '#8B0000'], 'Content Type')])
temp_df = df['rating'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['rating'],

                marker = dict(color = 'rgb(255,165,0)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'Most Rated Rating On NETFLIX' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()



df1 = df[df["type"] == "TV Show"]

df2 = df[df["type"] == "Movie"]



temp_df1 = df1['release_year'].value_counts().reset_index()

temp_df2 = df2['release_year'].value_counts().reset_index()





# create trace1

trace1 = go.Bar(

                x = temp_df1['index'],

                y = temp_df1['release_year'],

                name="TV Shows",

                marker = dict(color = 'rgb(249, 6, 6)'))

# create trace2 

trace2 = go.Bar(

                x = temp_df2['index'],

                y = temp_df2['release_year'],

                name = "Movies",

                marker = dict(color = 'rgb(26, 118, 255)'))





layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR BY CONTENT TYPE' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
temp_df = df['country'].value_counts().reset_index()[0:20]





# create trace1

trace1 = go.Bar(

                x = temp_df['index'],

                y = temp_df['country'],

                marker = dict(color = 'rgb(153,255,153)',

                              line=dict(color='rgb(0,0,0)',width=1.5)))

layout = go.Layout(template= "plotly_dark",title = 'TOP 20 COUNTIES WITH MOST CONTENT' , xaxis = dict(title = 'Countries'), yaxis = dict(title = 'Count'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()