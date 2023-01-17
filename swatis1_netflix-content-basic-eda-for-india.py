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

import seaborn as sns
data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
data.head()
data.shape
data.columns
data.isnull().sum()
data[data==0].count()
data.info()
data.duplicated().sum()
data.drop('show_id', axis=1, inplace= True)
sns.countplot(data.type)
Count_movie = len(data[data.type=='Movie'])

Count_tv = len(data[data.type=='TV Show'])



print("Percentage of movies: {:.2f}%".format((Count_movie / (len(data.type))*100)))

print("Percentage of TV Shows: {:.2f}%".format((Count_tv / (len(data.type))*100)))
data['director'].value_counts().head(15).plot.bar()
data['country'].value_counts().head(15).plot.bar()
data.dtypes
data['release_year']= pd.to_numeric(data['release_year'])
pd.crosstab(data.release_year, data.type).plot(kind = 'bar', figsize=(20,6))

plt.xlabel('Movies Released Year')

plt.ylabel('Frequency')

plt.title('Type of Movies Released in a Year')

plt.show()
pd.crosstab(data.rating, data.type).plot(kind='bar', figsize=(20,10))

plt.xlabel('Movies Rating')

plt.ylabel('Frequency')

plt.title('Rating of various contents on Netflix')

plt.show()
#slicing the data specific to India

#contents that have been released only for Indian viewers and not in other countries

data_india= data[data['country']== 'India']
data_india.head()
data_india.type.value_counts()
data_india.isnull().sum()
#dropping the null values

data_india= (data_india.dropna())
data_india.shape
#top 15 directors in India

data_india.director.value_counts().head(15).plot.bar()
#top 15 cast of India

data_india['cast'].value_counts().head(15).plot.bar()
#word cloud for the popular actors of India

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  

                      height = 1000, max_words = 121).generate(' '.join(data_india['cast']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Actors',fontsize = 30)

plt.show();
#Word Cloud for Keywords in Title

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  

                      height = 1000, max_words = 121).generate(' '.join(data_india['title']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Keywords in Movies or Shows Titles',fontsize = 30)

plt.show()
#word cloud for description of shows

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  

                      height = 1000, max_words = 121).generate(' '.join(data_india['description']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Key description of Netflix Shows in India',fontsize = 30)

plt.show()
#word cloud for the netflix shows in listed category

plt.rcParams['figure.figsize'] = (13, 13)

wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = 1000,  

                      height = 500, max_words = 200).generate(' '.join(data_india['listed_in']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Listed Category Word Cloud',fontsize = 30)

plt.show()
#number of netflix contents in India has increased after 2005

plt.figure(figsize=(20,8))

sns.countplot(data_india.release_year)

plt.xticks(rotation=90)
#netflix movies are released at a very high speed at compared to TV shows on netflix in India

plt.figure(figsize=(20,8))

sns.countplot(data_india.release_year, hue= data_india.type)

plt.xticks(rotation=90)
#created this list by referrimg to: https://www.kaggle.com/shivamsharma22/netflix-bollywood-movies-analysis

movie_list = {'TV-Y7':'Child Movies',

              'TV-G':'Family Movies',

              'TV-PG':'Family Movies-Parental Guidance',

              'TV-14':'Family Movies-Parental Guidance',

              'TV-MA':'Adult Movies',

              'TV-Y7-FV':'Child Movies',

              'PG-13':'Family Movies-Parental Guidance',

              'PG':'Family Movies-Parental Guidance',

              'R':'Adult Movies',

              'NR':'Unrated Movies',

              'UR':'Unrated Movies'}
#adding a new column 'movie_type' to the existing dataset

data_india['movie_type'] = data_india['rating'].map(movie_list)
data_india.head()
sns.countplot(data_india.movie_type, hue= data_india.type)

plt.xticks(rotation=90)
plt.figure(figsize=(20,8))

sns.countplot(data_india.duration, hue= data_india.type)

plt.xticks(rotation=90)