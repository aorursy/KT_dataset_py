! pip install rake_nltk



import pandas as pd

import numpy as np

from rake_nltk import Rake



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer



from ast import literal_eval
movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')



# Join datasets

credits.columns = ['id', 'title', 'cast', 'crew']



alldata = movies.merge(credits, on = 'id')

alldata.head()
import warnings

warnings.filterwarnings('ignore')



# Trim dataset to include relevant features

df = alldata[['id', 'original_title', 'genres', 'keywords', 'overview', 'original_language', 'cast', 'crew']]



# Parse stringed list features into python objects

features = ['keywords', 'genres', 'cast', 'crew']

for i in features:

    df[i] = alldata[i].apply(literal_eval)

    

# Extract list of genres

def list_genres(x):

    l = [d['name'] for d in x]

    return(l)

df['genres'] = df['genres'].apply(list_genres)



# Extract top 3 cast members

def list_cast(x):

    l = [d['name'] for d in x]

    if len(l) > 3:

        l = l[:3]

    return(l)

df['cast'] = df['cast'].apply(list_cast)



# Extract top 5 keywords

def list_keywords(x):

    l = [d['name'] for d in x]

    if len(l) > 5:

        l = l[:5]

    return(l)

df['keywords'] = df['keywords'].apply(list_keywords)



# Extract director

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan

df['director'] = df['crew'].apply(get_director)



# Drop the now unnecessary crew feature

df = df.drop('crew', axis = 1)



# Clean features of spaces and lowercase all to ensure uniques

def clean_feat(x):

    if isinstance(x, list):

        return [i.lower().replace(" ","") for i in x]

    else:

        if isinstance(x, str):

            return x.lower().replace(" ", "")

        else:

            return ''



features = ['keywords', 'genres', 'cast', 'director']

for i in features:

    df[i] = df[i].apply(clean_feat)
df.head()
missing = df.columns[df.isnull().any()]

df[missing].isnull().sum().to_frame()
# Replace NaN from overview with an empty string

df['overview'] = df['overview'].fillna('')
# Initialize empty column

df['plotwords'] = ''



# function to get keywords from a text

def get_keywords(x):

    plot = x

    

    # initialize Rake using english stopwords from NLTK, and all punctuation characters

    rake = Rake()

    

    # extract keywords from text

    rake.extract_keywords_from_text(plot)

    

    # get dictionary with keywords and scores

    scores = rake.get_word_degrees()

    

    # return new keywords as list, ignoring scores

    return(list(scores.keys()))



# Apply function to generate keywords

df['plotwords'] = df['overview'].apply(get_keywords)
df_keys = pd.DataFrame() 



df_keys['title'] = df['original_title']

df_keys['keywords'] = ''



def bag_words(x):

    return(' '.join(x['genres']) + ' ' + ' '.join(x['keywords']) + ' ' +  ' '.join(x['cast']) + 

           ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['plotwords']))

df_keys['keywords'] = df.apply(bag_words, axis = 1)



df_keys.head()
# create count matrix

cv = CountVectorizer()

cv_mx = cv.fit_transform(df_keys['keywords'])
# create cosine similarity matrix

cosine_sim = cosine_similarity(cv_mx, cv_mx)

cosine_sim
# create list of indices for later matching

indices = pd.Series(df_keys.index, index = df_keys['title'])
def recommend_movie(title, n = 10, cosine_sim = cosine_sim):

    movies = []

    

    # retrieve matching movie title index

    if title not in indices.index:

        print("Movie not in database.")

        return

    else:

        idx = indices[title]

    

    # cosine similarity scores of movies in descending order

    scores = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    

    # top n most similar movies indexes

    # use 1:n because 0 is the same movie entered

    top_n_idx = list(scores.iloc[1:n].index)

        

    return df_keys['title'].iloc[top_n_idx]
recommend_movie('Toy Story', n = 5)
recommend_movie('The Avengers')
recommend_movie('The Hobbit: An Unexpected Journey')
recommend_movie('Ocean\'s Eleven', n = 7)