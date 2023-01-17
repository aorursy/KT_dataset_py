# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go



import plotly.express as px

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
netflix=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix['country'].unique()

netflix_india=netflix.loc[netflix['country'] == 'India']

netflix_United_States=netflix.loc[netflix['country'] == 'United States']

netflix_Germany=netflix.loc[netflix['country'] == 'Germany']

netflix_United_Kingdom=netflix.loc[netflix['country'] == 'United Kingdom']

netflix_Japan=netflix.loc[netflix['country'] == 'Japan']

netflix_Spain=netflix.loc[netflix['country'] == 'Spain']

netflix_Canada=netflix.loc[netflix['country'] == 'Canada']

netflix_Australia=netflix.loc[netflix['country'] == 'Australia']
netflix_country_list=[netflix_india,netflix_United_States,netflix_Germany,netflix_United_Kingdom,netflix_Japan,netflix_Spain,netflix_Canada,netflix_Australia]
group_netflix=netflix.type.value_counts()

trace=go.Pie(labels=group_netflix.index,values=group_netflix.values,pull=[0.05])

layout = go.Layout(title="TV Shows VS Movies", height=400, legend=dict(x=1.1, y=1.3))

fig = go.Figure(data=[trace],layout=layout)

fig.update_layout(height=500,width=700)

fig.show()
netflix['date_added']
netflix["date_added"] = pd.to_datetime(netflix['date_added'])

netflix['year_added'] = netflix['date_added'].dt.year
netflix
tv_show=[]

movie_show=[]

for i in range(2010,2019):

    h=netflix.loc[(netflix['type']=='Movie') &  (netflix['year_added']==float(i))]

    g=netflix.loc[(netflix['type']=='TV Show') &  (netflix['year_added']==float(i))]

    tv_show.append(len(h))

    movie_show.append(len(g))

trace1 = go.Scatter(x=[i for i in range(2010,2019)],y=movie_show,mode='lines+markers',name='Movies')

trace2=go.Scatter(x=[i for i in range(2010,2019)],y=tv_show,mode='lines+markers',name='TV Shows')

data=[trace1,trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
top_countries=netflix.country.value_counts()

top_countries=top_countries[:15][::-1]

trace=go.Bar(x=top_countries.values,y=top_countries[:15].index,orientation='h',name='',marker=dict(color='#6ad49b'))

data=[trace]

layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data, layout=layout)

fig.show()
netflix.head().T
def netflix_data(country):

    country_netflix=country.reset_index()

    country_netflix.rename(columns={'description':'categories'},inplace=True)

    country_netflix.date_added[1:]=pd.to_datetime(country_netflix.date_added[1:])

    trace1=go.Scatter(x=country.date_added,y=country_netflix.categories[1:],mode='lines+markers',name='Movies')

    data=[trace1]

    layout = go.Layout(title="Netflix Popularity in "+str(country)+" from LAST 5 Years", height=400,width=1500, legend=dict(x=0.1, y=1.1))

    fig = go.Figure(data,layout=layout)

    fig.show()

for country in netflix_country_list:

    netflix_data(country)
def content_over_years(country):

    movie_per_year=[]



    tv_shows_per_year=[]

    for i in range(2008,2020):

        h=netflix.loc[(netflix['type']=='Movie') & (netflix.year_added==i) & (netflix.country==str(country))] 

        g=netflix.loc[(netflix['type']=='TV Show') & (netflix.year_added==i) &(netflix.country==str(country))] 

        movie_per_year.append(len(h))

        tv_shows_per_year.append(len(g))

    trace1 = go.Scatter(x=[i for i in range(2008,2020)],y=movie_per_year,mode='lines+markers',name='Movies')

    trace2 = go.Scatter(x=[i for i in range(2008,2020)],y=tv_shows_per_year,mode='lines+markers',name='TV Show')

    data=[trace1,trace2]

    layout = go.Layout(title="Content added over the years in "+str(country), legend=dict(x=0.1, y=1.1, orientation="h"))

    fig = go.Figure(data, layout=layout)

    fig.show()

countries=['United States','Australia','Turkey','Hong Kong','Thailand',"United Kingdom",'Taiwan',"Egypt",'France','Spain'

          ,'Mexico','Japan','South Korea','India','Canada']



for country in countries:

    content_over_years(str(country))
small = netflix.sort_values("release_year", ascending = True)

small = small[small['duration'] != ""]

small[['title', "release_year"]][:15]
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator 

import matplotlib.pyplot as plt

from PIL import Image

char_mask = np.array(Image.open("../input/netflixanalysis/istockphoto-543052538-612x612.jpg"))

#char_mask = np.array(Image.open("../input/picture/images.jpg"))

image_colors = ImageColorGenerator(char_mask)

#alice_mask = np.array(Image.open("../input/netflix/abc.jpg"))

plt.rcParams['figure.figsize'] = (10, 10)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'white', width = 1000,  height = 1000, max_words =300,mask=char_mask).generate(' '.join(netflix['title']))

plt.imshow(wordcloud.recolor(color_func=image_colors))

#plt.imshow(wordcloud)

wordcloud.to_file('/kaggle/working/show.jpg')

plt.axis('off')

plt.title('Most Popular Words in Title',fontsize = 30)

plt.show()