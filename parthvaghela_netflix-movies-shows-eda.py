import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import plotly.graph_objects as go

import matplotlib.pyplot as plt

import missingno as msno

import os
netflix_data = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv") # load netflix movies and TV Series data into Pandas Dataframe.

netflix_data.shape
msno.matrix(netflix_data)

plt.show()
for column in netflix_data.columns:

    null_count = netflix_data[column].isna().sum()

    if null_count > 0:

        print(f"{column}'s null count: {null_count}")
netflix_data.head()
netflix_data.describe(include='all').head(4)
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(14, 14), facecolor=None)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=1000, height=1000, max_words=150).generate(' '.join(netflix_data['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("Most Popular words in Title", fontsize=25)

plt.show()
plt.subplots(figsize=(9,7))

netflix_data['type'].value_counts().plot(kind='bar', fontsize=12,color='blue')

plt.xlabel('Type',fontsize=12)

plt.ylabel('Count',fontsize=12)

plt.title('Type Count',fontsize=12)

plt.ioff()
netflix_data["date_added"] = pd.to_datetime(netflix_data['date_added'])

netflix_data['year_added'] = netflix_data['date_added'].dt.year

netflix_data[netflix_data['year_added']> 2009]['year_added']



plt.figure(figsize=(10,8))

sns.countplot(x='year_added', hue='type', data=netflix_data[netflix_data['year_added']>2000])

# Show the plot

plt.show()
import squarify

from collections import Counter

country_data = netflix_data['country'].dropna()

country_count_dict = dict(Counter([country_name.strip() for country_name in (','.join(country_data).split(','))]))

country_data_count = pd.Series(country_count_dict).sort_values(ascending=False)

y = country_data_count[:25]

plt.rcParams['figure.figsize'] = (20, 16)

squarify.plot(sizes = y.values, label = y.index, color=sns.color_palette("RdGy", n_colors=25))

plt.title('Top 25 producing countries', fontsize = 25, fontweight="bold")

plt.axis('off')

plt.show()
start_year = 2010

end_year = 2020

def content_over_years(country):

    movie_per_year=[]

    tv_shows_per_year=[]



    for i in range(start_year,end_year):

        h=netflix_data.loc[(netflix_data['type']=='Movie') & (netflix_data.year_added==i) & (netflix_data.country==str(country))] 

        g=netflix_data.loc[(netflix_data['type']=='TV Show') & (netflix_data.year_added==i) &(netflix_data.country==str(country))] 

        movie_per_year.append(len(h))

        tv_shows_per_year.append(len(g))







    trace1 = go.Scatter(x=[i for i in range(start_year, end_year)], y=movie_per_year, mode='lines+markers', name='Movies')



    trace2=go.Scatter(x=[i for i in range(start_year, end_year)], y=tv_shows_per_year, mode='lines+markers', name='TV Shows')



    data=[trace1, trace2]



    layout = go.Layout(title="Content added over the years in "+str(country), legend=dict(x=0.1, y=1.1, orientation="h"))



    fig = go.Figure(data, layout=layout)



    fig.show()



countries=['United States', 'India', "United Kingdom", "France", 'Australia','Turkey','Hong Kong','Thailand', 'Taiwan',"Egypt", 'Spain'

          ,'Mexico','Japan','South Korea','Canada']



for country in countries:

    content_over_years(str(country))
rating_df = netflix_data['rating'].dropna()

rating_df.value_counts()

fig, ax = plt.subplots(figsize=(10,7))

rating_df.value_counts().plot(kind='bar', fontsize=12,color='blue')

plt.xlabel('Rating Type',fontsize=12)

plt.ylabel('Count',fontsize=12)

plt.title('Rating Type Count',fontsize=12)

plt.ioff()