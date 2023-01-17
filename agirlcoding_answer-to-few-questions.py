###Import all the relevant libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure, show

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
###import the DataSet

tv_shows= pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')

tv_shows.head()

tv_shows.info()
###General Information about the DataSet

BestMovie=tv_shows.loc[tv_shows['IMDb'].idxmax()]

LongestMovie=tv_shows.loc[tv_shows['Runtime'].idxmax()]

Title=BestMovie['Title']

LongestTitle=LongestMovie['Title']

Longest=LongestMovie['Runtime']



print('According to IMDb evaluation, the best movie is '+ Title + ', and the longest one is '+LongestTitle+ ' with ' +str(Longest) + 'min!')
###Divide the DataSet according to the Platform

Netflix=tv_shows[tv_shows['Netflix']>0][['Title','Age','IMDb','Year','Directors','Genres','Country','Language','Runtime']]

PrimeVideo=tv_shows[tv_shows['Prime Video']>0][['Title','Age','IMDb','Year','Directors','Genres','Country','Language','Runtime']]

Hulu=tv_shows[tv_shows['Hulu']>0][['Title','Age','IMDb','Year','Directors','Genres','Country','Language','Runtime']]

Disney=tv_shows[tv_shows['Disney+']>0][['Title','Age','IMDb','Year','Directors','Genres','Country','Language','Runtime']]
index_netflix = Netflix.index

total_netflix_movies = len(index_netflix)



index_hulu = Hulu.index

total_hulu_movies = len(index_hulu)



index_prime = PrimeVideo.index

total_prime_movies = len(index_prime)



index_disney = Disney.index

total_disney_movies = len(index_disney)



figure(figsize=(15,6))

AllPlatforms=[total_netflix_movies,total_hulu_movies,total_prime_movies,total_disney_movies ]

#AllPlatforms.sort()

Labels=['Neflix', 'Hulu', 'Prime Video', 'Disney']

ComparisonChart=sns.barplot(Labels, AllPlatforms, palette='Blues')

plt.xlabel('Platform', fontweight='bold', color = 'darkturquoise', fontsize='14')

plt.ylabel('Number of Streaming Available', fontweight='bold', color = 'darkturquoise', fontsize='14')

#ComparisonChart.set(xlabel='Platform', ylabel='Number of Streaming Available')

ComparisonChart.set_title('Number of Streaming Available per Platform', fontweight='bold', color = 'teal', fontsize='18')
# to seperate Genres column in dataset

seperated_genres = tv_shows['Genres'].str.get_dummies(',')



List_genres=[]



for col in seperated_genres:

    List_genres.append(col)



print(List_genres)

# to seperate Language column in dataset

seperated_languages = tv_shows['Language'].str.get_dummies(',')



List_languages=[]



for col in seperated_languages:

    List_languages.append(col)

## Let's build a search function



def NextMovie(ScoreMovie,GenreMovie,YearMovie,LanguageMovie):

    possiblemovie1=tv_shows.loc[tv_shows['IMDb']>(ScoreMovie)]

    possiblemovie2=possiblemovie1.loc[tv_shows['Genres'].str.contains(GenreMovie, na=False)]

    possiblemovie3=possiblemovie2.loc[tv_shows['Year']>(YearMovie)]

    possiblemovie4=possiblemovie3.loc[tv_shows['Language'].str.contains(LanguageMovie, na=False)]

    try:

      print(possiblemovie4['Title'])

    except:

      print("I\'m so sorry, no Title available, try to change parameters")

    
print(List_languages)

print(List_genres)

### Example



NextMovie(8,'Adventure',2015,'English')
def Availability(title):

    

    NetflixAvailable=Netflix.loc[Netflix['Title']==(title)]

    PrimeVideoAvailable=PrimeVideo.loc[PrimeVideo['Title']==(title)]

    DisneyAvailable=PrimeVideo.loc[PrimeVideo['Title']==(title)]

    HuluAvailable=PrimeVideo.loc[PrimeVideo['Title']==(title)]

   

        

    if (len(NetflixAvailable) >0):

            print('It is available on Netflix')

    else:

            print('It is not Available on Netflix')



    if (len(PrimeVideoAvailable) >0):

            print('It is available on PrimeVideo')

    else:

            print('It is not Available on PrimeVideo')



    if (len(DisneyAvailable) >0):

            print('It is available on Disney+')

    else:

            print('It is not Available on Disney+')



    if (len(HuluAvailable) >0):

            print('It is available on Hulu')

    else:

            print('It is not Available on Hulu')

        

        

###Example

title='Inglourious Basterds'

Availability(title)