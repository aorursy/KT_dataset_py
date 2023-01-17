# Importing some of the required libraries, will use imports as and when ever required

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from keras.layers import Dense

import string

from ast import literal_eval
credits = pd.read_csv('../input/the-movies-dataset/credits.csv')
# type of credits data is str, so we need to change them to python list object so that we can fetch detals inside it easily

type(credits['cast'][0])
# checking whether there are any null values in credits

credits.isnull().sum()
# we shall convert the features cast and crew to python objects

def load_data(data, columns):

    df = data.copy()

    for column in columns:

        df[column] = df[column].apply(literal_eval)

    return df
# loading data using above function which will convert str to list

credits = load_data(credits,['cast', 'crew'])
# peek into the data

credits.head()
# we can see how the features type has changed

type(credits['cast'][0])
# from casting details we shall pick the name identifier

credits['cast'][0]
# here we are picking a specific identifier, combine parameter will remove the space between first and last names because our model should consider the entire name as single character

def pick_a_detail_from_list(obj, identifier,number_of_words = True, combine = False):

    

    df = obj.copy()

    for i in range(len(df)):

        df[i] = [j[identifier] for j in df[i]]

        # if we want to limit the number of details to take we can pass True for this parameter

        if number_of_words:

            if (len(df[i])  > 4):

                df[i] = df[i][:4]  

    # if we want to combine the first and last name of the person we can pass True for this parameter

    if combine:

        df = df.apply(lambda x : [str.lower(name.replace(" ","")) for name in x])

    return df
# listing out few names from cast column

credits['cast_name'] = pick_a_detail_from_list(credits['cast'],'name', True)
# can see how we obtained the list of 4 names of cast for each movie

credits.head(3)
# details of crew

credits['crew'][0]
# similarly we can do for crew column, but for simplicity we will take only director name

def get_diro(x, identifier):

    for i in x:

        if i['job'] == identifier:

            return str.lower(i['name']).replace(" ","")

    return np.nan

credits['director'] = [get_diro(credits['crew'][j],identifier = 'Director') for j in range(len(credits['crew']))]

#credits['music'] = [get_diro(credits['crew'][j],identifier = 'Music') for j in range(len(credits['crew']))]
# Combining list of cast names into single string

credits['cast_name'] = credits['cast_name'].apply(lambda x : ' '.join(x))
# we can take a new dataframe and store only required things

credits_final = credits[['cast_name','director','id']].copy()
# loading keywords data

keywords = pd.read_csv('../input/the-movies-dataset/keywords.csv')



keywords = load_data(keywords,['keywords'])



# peek into the data

keywords.head()
# details of keywords column

keywords['keywords'][0]
# checking for null values

keywords.isnull().sum()
# we shall take all the keywords without limiting to 4 top words as we did in for cast names

keywords['keys'] = pick_a_detail_from_list(keywords['keywords'], 'name', False, False)
# obtained list of keywords 

keywords.head(3)
# combining list of keywords into single string

keywords['keys'] = keywords['keys'].apply(lambda x : ' '.join(x))
# splitting required columns from keywords data

keywords_final = keywords[['keys','id']].copy()
# loading metadata file

metadata = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
# peek into the data

metadata.head()
# Checking for null values

metadata.isnull().sum()
# we shall take genres and overview from metadata, but only genre need to be converted into python object 

metadata = load_data(metadata,['genres'])
# details of genres column

metadata['genres'][0]
# obtainng list of genres for movies

metadata['genre'] = pick_a_detail_from_list(metadata['genres'], 'name', False, True)
# converted list of genres

metadata['genre']
# Joining list of genres into one single string

metadata['genre'] = metadata['genre'].apply(lambda x : ' '.join(x))
# Splitting required data from metadata set

metadata_final = metadata[['genre','overview','id', 'original_title']].copy()

# Merging all three datasets (credits, keywords, metadata)

cbrs_final = credits_final.merge(keywords_final, on= 'id', how = 'inner')

# in metadata we have "id" as int so changing to str

cbrs_final['id'] = cbrs_final['id'].astype('str')
cbrs_final = cbrs_final.merge(metadata_final, on = 'id')
# peek into the combined data

cbrs_final.head(3)
# since we didnt process overview anywhere we have to lower the string in overview and remove punctuations, digits

cbrs_final['overview'] = cbrs_final['overview'].apply(lambda x : str(x).lower().translate(str.maketrans("","",string.punctuation+string.digits)))
cbrs_final.head(3)
# checking for null values in final dataset ready for converting into vectors

cbrs_final.isnull().sum()
len(cbrs_final)
cbrs_final.drop_duplicates(inplace= True)
len(cbrs_final)
# now lets combine all the text columns into one 



column_text = ['overview','cast_name','genre','director','keys']

cbrs_final['movie_info'] = cbrs_final[column_text].apply(lambda x : ' '.join(x.astype(str)), axis =1)
cbrs_final.head(2)
# sample movie info for first movie #Toy Story

cbrs_final['movie_info'][0]
# now we shall see traditional content based RS by using cosine similarity

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

movie_matrix = count.fit_transform(cbrs_final['movie_info'])
# lets map movie to index so that we can fetch recommendations

index_movie = pd.Series(cbrs_final.index, index=cbrs_final['original_title'])
# Functions for obtaining recommendations given title and similarity matrix

def cbrs_recommendations(title, similarity_matrix):

    # Get the index of the movie that matches the title

    idx = index_movie[title]



    # Get the pairwsie similarity scores of all movies with that movie

    similarity = list(enumerate(similarity_matrix[idx]))



    # Sort the movies based on the similarity scores

    similarity = sorted(similarity, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    similarity = similarity[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in similarity]



    # Return the top 10 most similar movies

    return cbrs_final['original_title'].iloc[movie_indices]
movie_matrix.shape
# the matrix shape is so large and using more than allocated memory so lets use the movies present in links_small.csv

links = pd.read_csv('../input/the-movies-dataset/links_small.csv')
# Checking for null values in links data, our concentration is on tmdbId column since in other datasets we have id which is tmdbID in links

links.isnull().sum()
# since we have null in our required column removing null columns

links.dropna(axis =0,inplace=True)
links.dtypes
# in links_small data we have id as float so changing to str, first changing to int and then str because if we directly change to str we get string as '862.0' instead of '862'

links['tmdbId'] = links['tmdbId'].astype(int)

links['tmdbId'] = links['tmdbId'].astype('str')

cbrs_small = cbrs_final.merge(links, left_on = 'id', right_on = 'tmdbId')
# peek into the final dataset

cbrs_small.head(4)
# now we shall get cosine similarity for the small data, obtaining count vectorized matrix



movie_matrix1 = count.fit_transform(cbrs_small['movie_info'])
# using cosine similairty from pairwise library we can compute cosine similarity for all the movies 

from sklearn.metrics.pairwise import cosine_similarity



cos_sim = cosine_similarity(movie_matrix1, movie_matrix1)
# Obtaining recommendations from the function created above 

cbrs_recommendations('The Man from Beyond', cos_sim)
# By using surprise dataset we can achieve SVD

from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate

reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()
# Removing timestamp and loading to surprise dataset(It must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings, in this order.)

ratings_data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# now we can initialize the SVD

svd = SVD()
# now we can evaluate SVD

cross_validate(svd, ratings_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# now we can fit our model

train = ratings_data.build_full_trainset()

svd.fit(train)
# predict (we need to provide userid itemid and  true rating,if we have. if we need for any iteam we can mention it as 0)

svd.predict(1, 1029, 3)