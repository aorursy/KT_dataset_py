# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, ImageColorGenerator
df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df.head()
df.info()
df.isnull().sum()
df.drop(['Rotten Tomatoes', 'Unnamed: 0' ,'ID'], axis=1, inplace=True)
df = df.dropna(how='any')
df.describe()
text = ",".join(review for review in df.Title)
wordcloud = WordCloud(max_words=200,collocations=False,background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()
text = ",".join(review for review in df.Directors)
wordcloud = WordCloud(max_words=200,collocations=False,background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()
df.head()
netflix = df.loc[df['Netflix'] == 1]
primevideo = df.loc[df['Prime Video'] == 1]
hulu = df.loc[df['Hulu'] == 1]
disney = df.loc[df['Disney+'] == 1]
df.columns
netflix.drop(['Hulu', 'Prime Video',
       'Disney+'], axis=1, inplace=True)
hulu.drop(['Netflix', 'Prime Video',
       'Disney+'], axis=1, inplace=True)
disney.drop(['Netflix', 'Prime Video',
       'Hulu'], axis=1, inplace=True)
primevideo.drop(['Netflix', 'Hulu',
       'Disney+'], axis=1, inplace=True)
netflix.head()
total_netflix = len(netflix.index)
total_hulu = len(hulu.index)
total_disney = len(disney.index)
total_primevideo = len(primevideo.index)
print ('Netflix has: ', total_netflix, ' movies')
plt.figure(figsize=(7,7))
labels = 'Netflix' , 'Hulu', 'Prime Video', 'Disney+'
sizes = [total_netflix,total_hulu,total_primevideo,total_disney]
explode = (0.1, 0.1, 0.1, 0.1 )

fig1 , ax1 = plt.subplots()

ax1.pie(sizes,
        explode = explode,
        labels = labels,
        autopct = '%1.1f%%',
        shadow = True,
        startangle = 100)

ax1.axis ('equal')
plt.show()
df['Runtime'].head()
top15_runtime_netflix = netflix.sort_values(by='Runtime', ascending=False).head(20)
plt.figure(figsize=(10,8))
sns.barplot(x='Runtime', y='Title', data=top15_runtime_netflix, hue='Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Minutes')
plt.ylabel('Movie')
plt.title('Top 20 movies by Run Time on Netflix')

plt.show()
top15_runtime_prime = primevideo.sort_values(by='Runtime', ascending=False).head(20)
plt.figure(figsize=(10,10))
sns.barplot(x='Runtime', y='Title', data=top15_runtime_prime, hue='Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Minutes')
plt.ylabel('Movie')
plt.title('Top 20 movies by Run Time on Amazon Prime Videp')

plt.show()
top15_runtime_disney = disney.sort_values(by='Runtime', ascending=False).head(20)
plt.figure(figsize=(10,10))
sns.barplot(x='Runtime', y='Title', data=top15_runtime_disney, hue='Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Minutes')
plt.ylabel('Movie')
plt.title('Top 20 movies by Run Time on Disney+')

plt.show()
top15_runtime_hulu = hulu.sort_values(by='Runtime', ascending=False).head(20)
plt.figure(figsize=(10,10))
sns.barplot(x='Runtime', y='Title', data=top15_runtime_hulu, hue='Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Minutes')
plt.ylabel('Movie')
plt.title('Top 20 movies by Run Time on Hulu')

plt.show()
top_rated = [(netflix['IMDb'] > 8).sum(), (primevideo['IMDb'] > 8).sum(), (hulu['IMDb'] > 8).sum(), (disney['IMDb'] > 8).sum()]
platforms = ['Netflix', 'Amazon prime', 'HULU', 'Disney']
data_top_rated = pd.DataFrame({'platforms' : platforms,
                               'Count' : top_rated})
sns.barplot(x= 'platforms', y = 'Count', data=data_top_rated)
top_rated = [(netflix['IMDb'] > 8).sum()/(len(netflix))*1000, 
            (primevideo['IMDb'] > 8).sum()/len(primevideo)*1000, 
            (hulu['IMDb'] > 8).sum()/len(hulu)*1000,
            (disney['IMDb'] > 8).sum()/len(disney)*1000]
platforms = ['Netflix', 'Amazon prime', 'HULU', 'Disney']
data_top_rated_per = pd.DataFrame({'platforms' : platforms,
                               'Count' : top_rated})
sns.barplot(x= 'platforms', y = 'Count', data=data_top_rated_per)
top_rated_netflix = netflix.sort_values('IMDb',ascending = False).head(15)
plt.figure(figsize=(10,8))
sns.barplot(x='Title', y='IMDb', data=top_rated_netflix)
plt.xticks(rotation='vertical')
plt.title('Netflix')
plt.xlabel('Movie Titles')
plt.show()
top_rated_disney = disney.sort_values('IMDb',ascending = False).head(15)
plt.figure(figsize=(10,8))
sns.barplot(x='Title', y='IMDb', data=top_rated_disney)
plt.xticks(rotation='vertical')
plt.title('disney+')
plt.xlabel('Movie Title')
plt.show()
top_rated_hulu = hulu.sort_values('IMDb',ascending = False).head(15)
plt.figure(figsize=(10,8))
sns.barplot(x='Title', y='IMDb', data=top_rated_hulu)
plt.xticks(rotation='vertical')
plt.title('Hulu')
plt.xlabel('Movie Title')
plt.show()
top_rated_prime = primevideo.sort_values('IMDb',ascending = False).head(15)
plt.figure(figsize=(10,8))
sns.barplot(x='Title', y='IMDb', data=top_rated_prime)
plt.xticks(rotation='vertical')
plt.title('Amazon Prime')
plt.xlabel('Movie Title')
plt.show()
top_rated = df.sort_values('IMDb',ascending = False).head(10)
plt.figure(figsize=(10,8))
sns.barplot(x='Title', y='IMDb', data=top_rated)
plt.xticks(rotation='vertical')
plt.title('Top rated movies of all times in dataset')
plt.xlabel('Movie Title')
plt.show()
count_by_lang = df.groupby('Language')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})
plt.figure(figsize=(10,12))
sns.barplot(x='Language', y = 'Movie Count', data=count_by_lang)
plt.title('Count by Languages')
plt.xticks(rotation='vertical')
plt.show()
yearly_count = df.groupby('Year')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'})
plt.figure(figsize=(20,10))
sns.barplot(x='Year', y = 'Movie Count', data=yearly_count)
plt.xticks(rotation='vertical')
plt.show()
country_count = df.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending = False).head(10).rename(columns = {'Title':'Movie Count'})
country_count
plt.figure(figsize=(10,12))
sns.barplot(x='Country', y = 'Movie Count', data=country_count)
plt.xticks(rotation='vertical')
plt.show()
directors_count = df.groupby('Directors')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)
directors_count
plt.figure(figsize=(10,12))
sns.barplot(x='Directors', y = 'Movie Count', data=directors_count)
plt.xticks(rotation='vertical')
plt.show()
imdb = df[df['IMDb'] > 8]
directors_above_8 = imdb.groupby('Directors')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(15)
plt.figure(figsize=(10,12))
sns.barplot(x='Directors', y = 'Movie Count', data=directors_above_8)
plt.xticks(rotation='vertical')
plt.title('Directors with the most 8+ rated movies')
plt.show()
countries_above_8 = imdb.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)
countries_above_8
plt.figure(figsize=(10,12))
sns.barplot(x='Country', y = 'Movie Count', data=countries_above_8)
plt.xticks(rotation='vertical')
plt.title('Countries with the most 8+ rated movies')
plt.show()
top_genres = df.groupby('Genres')['Title'].count().reset_index().rename(columns = {'Title':'Movie Count'}).sort_values('Movie Count',ascending = False).head(10)
top_genres
plt.figure(figsize=(10,12))
sns.barplot(x='Genres', y = 'Movie Count', data=top_genres)
plt.xticks(rotation='vertical')
plt.show()
