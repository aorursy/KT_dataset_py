# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#importing plotly

import plotly.graph_objects as go

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix.shape
netflix.head(10)
import plotly.express as px

import plotly.graph_objects as po
netflix_group = netflix['type'].value_counts()



# Trace used to represent the actual data

trace = go.Pie(labels=netflix_group.index,values=netflix_group.values,pull=[0.10])

# Layout for the data

layout = go.Layout(title='Netflix Shows vs Movies',height=400)



#Figure

fig = go.Figure(data=trace,layout=layout)

fig.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.figure(figsize=(10,10))

#WordCloud Parenthesis includes deets about the format & its parameters, stopwords won't be included

#max_words is maximum words represented in the word cloud

#' '.join because title contains spaces. 

#generate as in word cloud to be genearated on which text

wordcloud = WordCloud(stopwords=STOPWORDS, max_words=100).generate(' '.join(netflix['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 35)

plt.show()
netflix_country = netflix['country'].value_counts()

netflix_country = netflix_country[:10][::-1]



trace=go.Bar(x=netflix_country.values,y=netflix_country.index,orientation='h',text=netflix_country.values,textposition='auto')

layout = go.Layout(title="Countries with most content", height=700)

fig = go.Figure(data=[trace], layout=layout)

fig.show()
netflix_shows=netflix[netflix['type']=='TV Show']



durations= netflix_shows[['title','duration']]

durations['no_of_seasons']=durations['duration'].str.replace(' Season','')



durations['no_of_seasons']=durations['no_of_seasons'].str.replace('s','')



durations['no_of_seasons']=durations['no_of_seasons'].astype(str).astype(int)

to_be_plot = durations.sort_values('no_of_seasons',ascending=False)[:30]



trace = go.Table(header=dict(values=['Title', 'No of seasons']),cells=dict(values=[to_be_plot['title'],to_be_plot['no_of_seasons']],fill_color='lightgreen'))



fig = go.Figure(data=[trace])

fig.show()
netflix['release_year'].astype(int)

netflix_movies = netflix[netflix['type']=='Movie']

netflix_movies = netflix_movies[netflix_movies['release_year']>=2000]

to_be_plot = netflix_movies['release_year'].value_counts().sort_index()
data = [go.Scatter(x=to_be_plot.index,y=to_be_plot.values,mode='markers')]

layout = go.Layout(title='Scatter Plot')

fig = go.Figure(data, layout)

fig.show()
netflix_movies = netflix[netflix['type']=='Movie']

netflix_movies['duration']=netflix['duration'].str.replace(' min','')

netflix_movies['duration']=netflix_movies['duration'].astype('int')
trace = go.Histogram(x= netflix_movies['duration'],xbins=dict(size=0.5))

layout = go.Layout(template= "plotly_dark", title = 'Distribution of Movies Duration in Minutes')

fig = go.Figure(data = [trace], layout = layout)

fig.show()
netflix_movies = netflix[netflix['type']=='Movie']

netflix_movies = netflix_movies[netflix_movies['country']=='India']

to_be_plot=netflix_movies['director'].value_counts()[:10]
trace = go.Bar(y=to_be_plot.index,x=to_be_plot.values,orientation='h',marker=dict(color="orange"))

layout = go.Layout(template='plotly_dark',title="Movie Directors from India with most content")

fig = go.Figure(data=[trace], layout=layout)

fig.show()
netflix['rating'].value_counts().plot.pie(figsize=(10,10)) #distribution according to the rating

plt.show()