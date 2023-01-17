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
#https://www.datacamp.com/community/tutorials/recommender-systems-python

import pandas as pd

metadata = pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')
metadata = metadata[:20000]

print(metadata.shape)
C = metadata['vote_average'].mean()

print(C)
m = metadata['vote_count'].quantile(0.90)

print(m)
q_movies = metadata.copy().loc[metadata['vote_count']>m]
q_movies.shape
def weighted_rating(data,m=m,c=C):

    v = data['vote_count']

    R = data['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating,axis = 1)
q_movies.sort_values(by = 'score',ascending = False,inplace = True)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)

#This is the simple recommmender
metadata['overview'].head()

#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

metadata['overview'] = metadata['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(metadata['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
#Array mapping from feature integer indices to feature name.

tfidf.get_feature_names()[5000:5010]
# Import linear_kernel

from sklearn.metrics.pairwise import cosine_similarity



# Compute the cosine similarity matrix

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape

cosine_sim[1]
#Construct a reverse map of indices and movie titles

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
indices
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    idx = indices[title]

    print(idx)



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))

    

    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    

    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]

    

    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]

    

    # Return the top 10 most similar movies

    return metadata['title'].iloc[movie_indices]
get_recommendations('The Dark Knight Rises')

list(enumerate(cosine_sim[18252]))
get_recommendations('The Godfather')

# Load keywords and credits

credits = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')

keywords = pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')

credits.head()
keywords.head()
# Remove rows with bad IDs.

metadata = metadata.drop([19730])



# Convert IDs to int. Required for merging

keywords['id'] = keywords['id'].astype('int')

credits['id'] = credits['id'].astype('int')

metadata['id'] = metadata['id'].astype('int')



# Merge keywords and credits into your main metadata dataframe

metadata = metadata.merge(credits, on='id')

metadata = metadata.merge(keywords, on='id')
metadata
# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    metadata[feature] = metadata[feature].apply(literal_eval)
metadata
def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
# Define new director, cast, genres and keywords features that are in a suitable form.

metadata['director'] = metadata['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    metadata[feature] = metadata[feature].apply(get_list)
metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
#Function to convert all strings to lower case and strip names of spaces

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

    metadata[feature] = metadata[feature].apply(clean_data)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Create a new soup feature

metadata['soup'] = metadata.apply(create_soup, axis=1)
metadata[['soup']].head(2)
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(metadata['soup'])
count_matrix.shape
