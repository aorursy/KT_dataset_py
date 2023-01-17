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
credits = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

credits.shape
movies = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

movies.shape
credits.columns = ['id','movie','cast','crew']
df = movies.merge(credits, on='id')

df.shape
df.head(3)
# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    df[feature] = df[feature].apply(literal_eval)
# Get the director's name from the crew feature. If director is not listed, return NaN

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
# Returns the list top 3 elements or entire list; whichever is more.

def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 5:

            names = names[:5]

        return names



    #Return empty list in case of missing/malformed data

    return []
# Define new director, cast, genres and keywords features that are in a suitable form.

df['director'] = df['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    df[feature] = df[feature].apply(get_list)
# Print the new features of the first 3 films

df[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
# Function to convert all strings to lower case and strip names of spaces

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''
# Apply clean_data function to your features.

features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    df[feature] = df[feature].apply(clean_data)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df['soup'] = df.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim = cosine_similarity(count_matrix, count_matrix)
# Reset index of our main DataFrame and construct reverse mapping as before

df = df.reset_index()

indices = pd.Series(df.index, index=df['title'])
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return df['title'].iloc[movie_indices]
get_recommendations("The Lord of the Rings: The Return of the King", cosine_sim)
get_recommendations("Spectre", cosine_sim)
get_recommendations("A.I. Artificial Intelligence", cosine_sim)