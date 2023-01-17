import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
imdbdata=pd.read_csv('../input/IMDB-Movie-Data.csv')
imdbdata.head()
imdbdata.info()
imdbdata.dtypes
imdbdata.describe()
from os import path

from scipy.misc import imread

import matplotlib.pyplot as plt

import random

from wordcloud import WordCloud, STOPWORDS

text = (str(imdbdata['Title']))

plt.subplots(figsize=(20,15))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1500,

                          height=1200

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('Title')

plt.axis('off')

plt.show()
imdbdata=imdbdata.rename(columns = {'Revenue (Millions)':'Revenue_Millions'})
imdbdata=imdbdata.rename(columns = {'Runtime (Minutes)':'Runtime_Minutes'})
imdbdata["Genre"].value_counts()
seperate_genre='Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western'

for genre in seperate_genre:

    df = imdbdata['Genre'].str.contains(genre).fillna(False)

    print('The total number of movies with ',genre,'=',len(imdbdata[df]))

    f, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(x='Year', data=imdbdata[df], palette="Greens_d");

    plt.title(genre)

    compare_movies_rating = ['Runtime_Minutes', 'Votes','Revenue_Millions', 'Metascore']

    for compare in compare_movies_rating:

        sns.jointplot(x='Rating', y=compare, data=imdbdata[df], alpha=0.7, color='b', size=8)

        plt.title(genre)
from os import path

from scipy.misc import imread

import matplotlib.pyplot as plt

import random

from wordcloud import WordCloud, STOPWORDS

text = (str(imdbdata['Description']))

plt.subplots(figsize=(20,15))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1500,

                          height=1200

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('DESCRIPTION')

plt.axis('off')

plt.show()
imdbdata.Director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 10 DIRECTORS OF MOVIES')
imdbdata.Actors.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 10 ActorS OF MOVIES')
ctors=imdbdata["Actors"]

actors=set(ctors)

from wordcloud import WordCloud, STOPWORDS

plt.subplots(figsize=(10,10))

text = (str(actors))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('ACTORS')

plt.axis('off')

plt.show()
imdbdata["Year"].value_counts()
sns.stripplot(x="Year", y="Rating", data=imdbdata, jitter=True);

print(' RATING BASED ON YEAR')
sns.swarmplot(x="Year", y="Votes", data=imdbdata);

print(' VOTES BASED ON YEAR')
sns.stripplot(x="Year", y="Revenue_Millions", data=imdbdata, jitter=True);

print(' REVENUE BASED ON YEAR')
sns.swarmplot(x="Year", y="Metascore", data=imdbdata);

print(' METASCORE BASED ON YEAR')
imdbdata["Runtime_Minutes"].value_counts()
imdbdata.Runtime_Minutes.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('TOP 10 runtime OF MOVIES')
time=imdbdata.Runtime_Minutes

sns.distplot(time, bins=20, kde=False, rug=True);
sns.distplot(time, hist=False, rug=True);
imdbdata["Rating"].value_counts()
#top 10 rating movies 



Sortedrating= imdbdata.sort_values(['Rating'], ascending=False)

Sortedrating.head(10)
# low rated movies

lowratedmovies= imdbdata.query('(Rating > 0) & (Rating < 3.0)')

lowratedmovies.head()


print('number of low rated movies :')

len(lowratedmovies)
#medium rated movies

mediumratedmovies= imdbdata.query('(Rating > 3.0) & (Rating < 7.0)')

mediumratedmovies.head()
print('number of medium rated movies :')

len(mediumratedmovies)
sns.jointplot(x="Rating", y="Metascore", data=mediumratedmovies);

plt.title('(MOVIES WITH MEDIUM RATING , METASCORE')
sns.jointplot(x="Rating", y="Votes", data=mediumratedmovies);

plt.title('(MOVIES WITH MEDIUM RATING , VOTES')
sns.jointplot(x="Rating", y="Revenue_Millions", data=mediumratedmovies);

plt.title('(MOVIES WITH MEDIUM RATING , REVENUE')
#high rated movies

highratedmovies= imdbdata.query('(Rating > 7.0) & (Rating < 10.0)')

highratedmovies.head()
print('number of high rated movies :')

len(highratedmovies)
sns.jointplot(x="Rating", y="Metascore", data=highratedmovies);

plt.title('(MOVIES WITH HIGH RATING , METASCORE')
sns.jointplot(x="Rating", y="Votes", data=highratedmovies);

plt.title('(MOVIES WITH HIGH RATING ,VOTES')
sns.jointplot(x="Rating", y="Revenue_Millions", data=highratedmovies);

plt.title('(MOVIES WITH HIGH RATING ,REVENUE')
#top voted movies 



Sortedvotes= imdbdata.sort_values(['Votes'], ascending=False)

v= Sortedvotes.query('(Votes > 1000000)')

print('number of movies voted more than 1 million :')

len(v)


print('more than 1 million voted movies title :')

Sortedvotes["Title"].head(6)
print('more than 1 million voted movies rating:')

Sortedvotes["Rating"].head(6)
print('more than 1 million voted movies revenue:')

Sortedvotes["Revenue_Millions"].head(6)
print('more than 1 million voted movies Metascore:')

Sortedvotes["Metascore"].head(6)
#sorting based on revenue

Sortedrevenue= imdbdata.sort_values(['Revenue_Millions'], ascending=False)
#top 5 high revenue movies

Sortedrevenue.head()
m= Sortedrevenue.query('(Revenue_Millions > 500)')

print('number of movies with more than half million revenue:')

len(m)
n= Sortedrevenue.query('(Revenue_Millions < 500)')

print('number of movies with less than half million revenue:')

len(n)
#TOP REVENUE MOVIE

m.head()

from wordcloud import WordCloud, STOPWORDS

plt.subplots(figsize=(10,10))

text = (str(m['Actors']))

wordcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='white',

                          width=1200,

                          height=1000

                         ).generate(text)





plt.imshow(wordcloud)

plt.title('ACTORS IN HIGH REVENUE MOVIES')

plt.axis('off')

plt.show()

imdbdata.Metascore.value_counts()
metascore=imdbdata.Metascore

sns.boxplot(metascore);
sns.kdeplot(metascore);