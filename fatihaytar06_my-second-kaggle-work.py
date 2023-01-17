# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
top_5000_movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

top_5000_movies=top_5000_movies[['title','overview','popularity','release_date','budget','revenue','original_language','runtime','tagline','vote_average','vote_count']]
top_5000_movies.head()
top_5000_movies = top_5000_movies[top_5000_movies.vote_count>=3000]
top_5000_movies=top_5000_movies.sort_values(by=['vote_average'],ascending=False)
trace1=go.Scatter(

    x=top_5000_movies.budget,

    y=top_5000_movies.revenue,

    mode='markers',

    marker=dict(color='rgb(89,66,156)')

)

data=[trace1]

layout=dict(xaxis=dict(title='Budget of Movies'),yaxis=dict(title='Revenue of Movies'))

fig=dict(data=data,layout=layout)

iplot(fig)
top_5000_movies['rank']=np.arange(1,249)
top_5000_movies.vote_average.value_counts()
top_5000_movies.head()
top_5000_movies.head(10)
top_5000_movies.info()
top_100_movies=top_5000_movies.head(100)

trace1=go.Scatter3d(

    x=top_100_movies.budget,

    y=top_100_movies.revenue,

    z=top_100_movies.vote_average,

    mode='markers',

    marker=dict(size=10,color='red')



)

data=[trace1]

layout=go.Layout(margin=dict(l=0,r=0,t=0,b=0))

fig=go.Figure(data=data,layout=layout)

iplot(fig)
import matplotlib.pyplot as plt

plt.figure(figsize=(18,10))

sns.barplot(x=top_100_movies.title,y=top_100_movies.vote_average)

plt.xticks(rotation=90)

plt.xlabel('Movies')

plt.show()
top_5000_movies.reset_index()
top_100_movies['budget']=top_100_movies['budget']/max(top_100_movies['budget'])

top_100_movies.revenue=top_100_movies.revenue/max(top_100_movies.revenue)

f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x=top_100_movies.title,y=top_100_movies.revenue,color='red',alpha=0.8)

sns.pointplot(x=top_100_movies.title,y=top_100_movies.budget,color='blue',alpha=0.8)

plt.text(2,0.9,'Budget',color='blue',fontsize=18,style='italic')

plt.text(2,0.85,'Revenue',color='red',fontsize=18,style='italic')

plt.xlabel('Movies',fontsize=15,color='green')

plt.ylabel('Values',fontsize=15,color='cyan')

plt.xticks(rotation=90)

plt.title('Revenue vs Budget',fontsize=25)

plt.grid()

plt.show()
df_x = top_5000_movies.iloc[:4,:]

trace1=go.Bar(

    x=df_x.title,

    y=df_x.budget,

    name='Budget',

    marker=dict(color='rgb(89,66,179)',line=dict(color='white',width=1.5)),



)

trace2=go.Bar(

    x=df_x.title,

    y=df_x.revenue,

    name='Revenue',

    marker=dict(color='rgb(255,1,10)',line=dict(color='white',width=1.5)),



)

data=[trace1,trace2]

layout=go.Layout(barmode='group')

fig=go.Figure(data=data,layout=layout)

iplot(fig)
top_5000_movies['popularity']=(top_5000_movies['popularity']/max(top_5000_movies['popularity']))*80

international_color = [float(each) for each in top_5000_movies.revenue]

data=go.Scatter(

    x=top_5000_movies.vote_average,

    y=top_5000_movies.budget,

    mode='markers',

    marker=dict(color=international_color,size=top_5000_movies['popularity'],showscale=True),

    text=top_5000_movies.title

)

data=[data]

layout=go.Layout(xaxis=dict(title='IMDB Score'),yaxis=dict(title='Budget'))

fig=go.Figure(data=data,layout=layout)

plt.savefig('Scat.png')

iplot(fig)

plt.show()
from wordcloud import WordCloud
data=top_100_movies.title

plt.subplots(figsize=(8,8))

worlcloud=WordCloud(

        background_color='white',

        width=512,

        height=384,).generate(' '.join(data))

plt.imshow(worlcloud)

plt.axis('off')

plt.savefig('movies.png')

plt.show()

trace1=go.Box(

    y=top_5000_movies.budget,

    name='Budgets of Movies',

    marker=dict(color='rgb(53,5,96)')

)

trace2=go.Box(

    y=top_5000_movies.revenue,

    name='Revenues of Movies',

    marker=dict(color='rgb(99,5,89)')

)

data=[trace1,trace2]

iplot(data)