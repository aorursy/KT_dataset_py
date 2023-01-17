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
credits=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies=pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
print(movies.shape)
movies.head(1)
print(credits.shape)
credits.head()
credits.isnull().sum()
credits=credits[['movie_id','cast','crew']]
credits.rename(columns={'movie_id':'id'},inplace=True)
credits.head()
df=movies.merge(credits,on='id')
df.sample(1)
# General IMDB wala
# Content based system
# weighted rating (WR) = [(v/(v+m)) × R] + [(m/(v+m)) × C]

# Where:

# R = average for the movie (mean) = (rating)

# v = number of votes for the movie = (votes)

# m = minimum votes required to be listed in the Top Rated list (currently 25,000)

# C = the mean vote across the whole report
m=25000
C=df['vote_average'].mean()
def weighted_rating(row,m=m,C=C):
    
    R=row['vote_average']
    v=row['vote_count']
    
    return ((v/(v+m))*R) + ((m/(v+m))*C)
df['wr']=df.apply(weighted_rating,axis=1)
df[['title','wr']].sort_values('wr',ascending=False)
# Important cols for content based filtering

# 1. Genre
# 2. keywords
# 3. overview
# 4. cast
# 5. crew
df['original_language'].value_counts()
def recommend(movie_name):
    pass
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english')
df['overview'].fillna(' ',inplace=True)
overview_matrix=cv.fit_transform(df['overview']).toarray()
overview_matrix
def recommend(movie_name):
    # find the index of this movie
    index_pos=df[df['title']==movie_name].index[0]
    
    # calulate similarity
    recommended_movie_index=sorted(list(enumerate(similarity_score[index_pos])),reverse=True,key=lambda x:x[1])[1:6]
    
    # movie name from index
    
    for i in recommended_movie_index:
        print(df.iloc[i[0]].title)
    
recommend('Harry Potter and the Prisoner of Azkaban')
from sklearn.metrics.pairwise import cosine_similarity

similarity_score=cosine_similarity(overview_matrix)
similarity_score
sorted(list(enumerate(similarity_score[200])),reverse=True,key=lambda x:x[1])[1:11]
df.iloc[15].title
df.columns
df[['title','cast','crew','genres','keywords']]['cast'][0]
['Action','Adventure','Science Fiction']
# genres
from ast import literal_eval
def clean_genre(x):
    genre=[]
    for i in literal_eval(x):
        genre.append(i['name'])
    return genre
df['genres']=df['genres'].apply(clean_genre)
df[['title','cast','crew','genres','keywords']]
def clean_keyword(x):
    keywords=[]
    for i in literal_eval(x):
        keywords.append(i['name'])
    return keywords
df['keywords']=df['keywords'].apply(clean_keyword)
df['keywords']
def clean_crew(x):
    for i in literal_eval(x):
        if i['job']=='Director':
            return i['name']
        
df['crew']=df['crew'].apply(clean_crew)
df['crew']
def clean_cast(x):
    actor=[]
    count=0
    for i in literal_eval(x):
        if count<4:
            actor.append(i['name'])
            count+=1
        else:
            return actor
    
df['cast']=df['cast'].apply(clean_cast)
df['crew'][0]
def clean(x):
    
    y=[]
    
    if type(x)==list:
        
        for i in x:
            
            i=i.lower()
            i=i.replace(" ","")
            y.append(i)
            
        return y
    if type(x)==str:
        
        x=x.lower()
        x=x.replace(" ","")
        
        return x
features=['cast','crew','genres','keywords']

for i in features:
    
    df[i]=df[i].apply(clean)
    
df[['cast','crew','genres','keywords']]
def merge_cols(x):
    y=""
    
    y = y + x['crew'] + " " + " ".join(x['cast']) + " " + " ".join(x['genres']) + " " + " ".join(x['keywords'])
    
    return y
df.dropna(subset=['cast','crew'],inplace=True)
df['all']=df.apply(merge_cols,axis=1)
cv=CountVectorizer(stop_words='english')
movie_matrix=cv.fit_transform(df['all']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix=cosine_similarity(movie_matrix)
def recommend(movie_name):
    # find the index of this movie
    index_pos=df[df['title']==movie_name].index[0]
    
    # calulate similarity
    recommended_movie_index=sorted(list(enumerate(similarity_matrix[index_pos])),reverse=True,key=lambda x:x[1])[1:11]
    
    # movie name from index
    
    for i in recommended_movie_index:
        print(df.iloc[i[0]].title)
    
recommend('Titanic')
df[['id','title','all']].to_csv('movie.csv')
import pickle
pickle.dump(similarity_matrix,open('similarity_matrix.pkl','wb'))
