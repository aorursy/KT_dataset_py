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
data1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

data2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
data1.head()
data2.head()
data1.columns=['id','title','cast','crew']

data3=data2.merge(data1,on='id')
data3.head()
# Parse the stringified features into their corresponding python objects

from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    data3[feature] = data3[feature].apply(literal_eval)
#geting the name of the director

def director(n):

    for i in n:

        if i['job']=="Director":

            return i['name']

    return np.nan

    
#create a function to return only 3 elements

def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
#defining new director,cast,crew,keyword features

data3['director']=data3['crew'].apply(director)

features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    data3[feature] = data3[feature].apply(get_list)
#checking the new features

data3[['title_x','director','cast','crew','keywords','genres']].head()
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

    data3[feature] = data3[feature].apply(clean_data)
def meta_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

data3['soup'] = data3.apply(meta_soup, axis=1)
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(data3['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim = cosine_similarity(count_matrix, count_matrix)
data3 = data3.reset_index()

indices = pd.Series(data3.index, index=data3['title_y'])
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

    return data3['title_x'].iloc[movie_indices]
get_recommendations("Spectre")
get_recommendations("Apocalypse Now")
get_recommendations("Before Sunrise")
from surprise import Dataset,SVD,Reader

from surprise.model_selection import cross_validate

reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd=SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 10]
svd.predict(10, 2995, 3)