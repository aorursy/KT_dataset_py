import numpy as np
import pandas as pd
import nltk
movies_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')
data = movies_data.merge(credits_data, how = 'inner', on = movies_data['id'])
data.info()
data.drop(['homepage', 'key_0', 'title_x', 'title_y', 'movie_id'], axis = 1, inplace = True)
data.info()
from ast import literal_eval
data['genres'] = data['genres'].apply(literal_eval)
data['keywords'] = data['keywords'].apply(literal_eval)
data['cast'] = data['cast'].apply(literal_eval)
data['crew'] = data['crew'].apply(literal_eval)
def getDirectorName(crew) :
    for i in crew :
        if i['job'] == 'Director' :
            return i['name']
def getNameList(words) :
    names = []
    for i in words :
        names.append(i['name'])
    
    if len(names) > 5 :
        return names[0:5]
    
    else :
        return names
data['Director'] = data['crew'].apply(lambda x : getDirectorName(x))
data['keywords'] = data['keywords'].apply( lambda x : getNameList(x) )
data['genres'] = data['genres'].apply( lambda x : getNameList(x) )
data['cast'] = data['cast'].apply( lambda x : getNameList(x) )
data.info()
data['Director'] = data['Director'].fillna(' ')
def cleanDiretor(name) :
    name = name.lower()
    name = name.replace(' ', '')
    return name
def cleanWordList(words) :
    names = []
    
    for word in words :
        word = word.lower()
        word = word.replace(' ', '')
        names.append(word)
    return names
def finalData(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['Director'] + ' ' + ' '.join(x['genres'])

data['Final_Data'] = data.apply(finalData, axis=1)
data.info()
from sklearn.feature_extraction.text import CountVectorizer
countvectorizer = CountVectorizer()
vectormatrix = countvectorizer.fit_transform(data['Final_Data'])
vectormatrix.shape
from sklearn.metrics.pairwise import cosine_similarity
similaritymatrix = cosine_similarity(vectormatrix, vectormatrix)
similaritymatrix.shape
similaritymatrix[0][0:10]
indices = pd.Series(data = data['id'].index, index = data['original_title']).drop_duplicates()
indices
def getSimilarMovies(moviename) :
    index = indices[moviename]
    
    similarmovies = list(enumerate(similaritymatrix[index]))
    similarmovies = sorted(similarmovies, key = lambda x : x[1], reverse = True)
    similarmovies = similarmovies[1:11]
    
    moviesindex = []
    
    for movie in similarmovies :
        moviesindex.append(movie[0])
        
    similarmovies = data['original_title'].iloc[moviesindex]
    
    return similarmovies
getSimilarMovies('The Wolverine')
getSimilarMovies('The Dark Knight Rises')
