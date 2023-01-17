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
credits=pd.read_csv(r"/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies=pd.read_csv(r"/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

movies.shape
credits.head()
credits.columns
movies.head()
credits.rename(columns={'movie_id':'id'},inplace=True) 

#I have chanced the 'movie_id' name to 'id' so that it matches the movies dataframes 'id' column





#Now i am going to merge credits and movies dataframe based on id

movies=movies.merge(credits,on='id')
movies.head()
C= movies['vote_average'].mean()  #mean vote across the whole report

C
m= movies['vote_count'].quantile(0.9)

m
q_movies = movies.loc[movies['vote_count'] >= m]

q_movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)



#Print the top 15 movies

q_movies[['original_title', 'vote_count', 'vote_average', 'score']].reset_index(drop=True).head(10)
pop= movies.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))



plt.barh(pop['original_title'].head(6),pop['popularity'].head(6), align='center',

        color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")
movies['overview'].head(5)
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

movies['overview'] = movies['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(movies['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles

indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()
pd.DataFrame(indices).head()
def get_recomendation(title,cosine_sim =cosine_sim ):

    #Get the index of the movie given its title

    idx=indices[title]

    #Get the list of cosine similarity scores

    sim_scores=list(enumerate(cosine_sim[idx]))

    #sort based on sim_score

    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)

    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]

    #Get the  movies indices 

    sim_scores=[i[0] for i in sim_scores]

     

    return movies['original_title'].iloc[sim_scores]

        
get_recomendation('Avatar')
get_recomendation('The Dark Knight Rises')


# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    movies[feature] = movies[feature].apply(literal_eval)
movies[feature].head()
movies.head()
movies['crew'][1]
# Get the director's name from the crew feature. If director is not listed, return NaN

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan
movies['director'] = movies['crew'].apply(get_director)
movies['director'].head()
def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []

features = ['cast', 'keywords', 'genres']

for feature in features:

    movies[feature] = movies[feature].apply(get_list)
movies.head()
movies[['original_title','director','cast', 'keywords','genres']].head()
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

    movies[feature] = movies[feature].apply(clean_data)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies['soup'] = movies.apply(create_soup, axis=1)
pd.DataFrame(movies['soup']).head()
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(movies['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of our main DataFrame and construct reverse mapping as before

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['original_title'])
def get_recomendation(title,cosine_sim =cosine_sim ):

    #Get the index of the movie given its title

    idx=indices[title]

    #Get the list of cosine similarity scores

    sim_scores=list(enumerate(cosine_sim[idx]))

    #sort based on sim_score

    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)

    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]

    #Get the  movies indices 

    sim_scores=[i[0] for i in sim_scores]

     

    return movies['original_title'].iloc[sim_scores]
get_recomendation('Avatar',cosine_sim2)
get_recomendation('The Godfather',cosine_sim2)
from surprise import Reader, Dataset, SVD

reader = Reader()

ratings = pd.read_csv('/kaggle/input/the-movies-dataset/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

data




from surprise.model_selection import cross_validate







# Use the famous SVD algorithm

svd = SVD()



# Run 5-fold cross-validation and then print results

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)