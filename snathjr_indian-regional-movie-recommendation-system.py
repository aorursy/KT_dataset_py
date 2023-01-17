# Import some basic libraries in python for data preprocessing



import json

import numpy as np

import pandas as pd
# Read movies.json file and load into moviesData



moviesData = []



with open('../input/indian-regional-movie/movies.json') as moviesFile:

    for line in moviesFile:

        moviesData.append(json.loads(line))



# Now take the moviesData and create a DataFrame



moviesDF = pd.DataFrame.from_records(moviesData)



# View the head of the moviesDF DataFrame



moviesDF.head()
# Let's create a dataframe with name and description

plotDF = moviesDF.loc[:, ['name', 'description']]



# Convert both columns to lowercase

plotDF.loc[:, 'name'] = plotDF.loc[:, 'name'].apply(lambda x: x.lower())

plotDF.loc[:, 'description'] = plotDF.loc[:, 'description'].apply(lambda x: x.lower())



# Drop all the rows which are empty

plotDF = plotDF[plotDF['description'] != '']



# Now drop duplicates in the plotDF

plotDF = plotDF.drop_duplicates()



plotDF.head()
# Lets quickly view the info of plotDF

plotDF.info()



# There are 2850 unique movies, we can see that there are no null description
# Import the TfIdfVectorizer from scikit-learn library

from sklearn.feature_extraction.text import TfidfVectorizer



# Define a TF IDF Vectorizer Object

# with the removal of english stopwords turned on

tfidf = TfidfVectorizer(stop_words = 'english')



# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature

tfidf_matrix = tfidf.fit_transform(plotDF['description'])



# View the shape of the TfIdf_matrix

tfidf_matrix.shape
# Import linear_kernel from scikit-learn to compute the dot product

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix by using linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



# Let's view the shape of the cosine similarity matrix

print("Shape of cosine similarity matrix: ", cosine_sim.shape)



# Let's quickly view how the matrix looks like in first few column and rows

cosine_sim[0:5, 0:5]
# Construct a pandas series of movie name as index and index as value

indices = pd.Series(plotDF.index, index = plotDF['name'])

indices.head()
# Function for returning top 10 movies for a movie title as input

def plot_based_recommender(title, df = plotDF, cosine_sim = cosine_sim, indices = indices):

  # Convert title to lower-case

  title = title.lower()



  # Obtain the index of the movie that matched the title 

  try:

    idx = indices[title]

  except KeyError:

    print('Movie does not exist :(')

    return False



  # Get the pairwise similarity score of all the movies with that movie

  # and convert it into a list of tuples (position, similarity score)

  sim_scores = list(enumerate(cosine_sim[idx]))



  # Sort the movies based on the cosine similarity scores

  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)



  # Get the scores of the top 10 most similar movies. Ignore the first movie.

  sim_scores = sim_scores[1:11]



  # get the movie indices

  movie_indices = [sim_score[0] for sim_score in sim_scores]



  # Return the top 10 similar movies

  return df['name'].iloc[movie_indices]
plot_based_recommender('Pyaar Ka Punchnama')
# Prepare the data

metaDF = moviesDF[['name', 'genre', 'language', 'director', 'cast', 'description']]

metaDF.head()
# We want to keep only the first 3 genre and cast (actors) in the list format

# We want to keep only the first director



metaDF.loc[:, 'genre'] = metaDF.loc[:, 'genre'].apply(lambda x: x[:3] if len(x) > 3 else x)

metaDF.loc[:, 'cast'] = metaDF.loc[:, 'cast'].apply(lambda x: x[:3] if len(x) > 3 else x)

metaDF.loc[:, 'director'] = metaDF.loc[:, 'director'].apply(lambda x: x[:1])

metaDF.head()
def sanitize(x):

    if isinstance(x, list):

        # Strip spaces

        return [i.replace(" ", "") for i in x]

    else:

        # if it is empty, return an empty string

        if isinstance(x, str):

            return x.replace(" ", "")

        else: 

            return ''
#Apply the generate_list function to cast, keywords, director and genres 

for feature in ['director', 'cast']:

    metaDF[feature] = metaDF[feature].apply(sanitize)
metaDF.head()
# Function that creates a soup out of the desired metadata

def create_soup(x):

    return ' '.join(x['genre']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['cast'] + [x['name']] + [x['description']])



# Create the new soup feature 

metaDF['soup'] = metaDF.apply(create_soup, axis=1)



#Display the soup of the first movie 

metaDF.iloc[0]['soup']
# Let's create a dataframe with name and soup

soupDF = metaDF.loc[:, ['name', 'soup']]



# Convert both columns to lowercase

soupDF.loc[:, 'name'] = soupDF.loc[:, 'name'].apply(lambda x: x.lower())

soupDF.loc[:, 'soup'] = soupDF.loc[:, 'soup'].apply(lambda x: x.lower())



# Drop all the rows which are empty

soupDF = soupDF[soupDF['soup'] != '']



# Now drop duplicates in the soupDF

soupDF = soupDF.drop_duplicates()



soupDF.head()

print(soupDF.shape)
# Import the TfIdfVectorizer from scikit-learn library

from sklearn.feature_extraction.text import TfidfVectorizer



# Define a TF IDF Vectorizer Object

# with the removal of english stopwords turned on

tfidf = TfidfVectorizer(stop_words = 'english')



# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature

tfidf_matrix = tfidf.fit_transform(soupDF['soup'])



# View the shape of the TfIdf_matrix

tfidf_matrix.shape
# Import linear_kernel from scikit-learn to compute the dot product

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix by using linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



# Let's view the shape of the cosine similarity matrix

print("Shape of cosine similarity matrix: ", cosine_sim.shape)



# Let's quickly view how the matrix looks like in first few column and rows

cosine_sim[0:5, 0:5]
# Construct a pandas series of movie name as index and index as value

indices = pd.Series(soupDF.index, index = soupDF['name'])

indices.head()
# Function for returning top 10 movies for a movie title as input

def plot_based_recommender(title, df = soupDF, cosine_sim = cosine_sim, indices = indices):

  # Convert title to lower-case

  title = title.lower()



  # Obtain the index of the movie that matched the title 

  try:

    idx = indices[title]

  except KeyError:

    print('Movie does not exist :(')

    return False



  # Get the pairwise similarity score of all the movies with that movie

  # and convert it into a list of tuples (position, similarity score)

  sim_scores = list(enumerate(cosine_sim[idx]))



  # Sort the movies based on the cosine similarity scores

  sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)



  # Get the scores of the top 10 most similar movies. Ignore the first movie.

  sim_scores = sim_scores[1:11]



  # get the movie indices

  movie_indices = [sim_score[0] for sim_score in sim_scores]



  # Return the top 10 similar movies

  return df['name'].iloc[movie_indices]
plot_based_recommender('Pyaar Ka Punchnama')
# database 

db = {} 

db['cosine_sim'] = cosine_sim

db['indices'] = indices



import pickle



dbfile = open('examplePickle', 'ab') 

pickle.dump(db, dbfile)                      

dbfile.close() 