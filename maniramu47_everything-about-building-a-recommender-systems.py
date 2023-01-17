import pandas as pd

import numpy as np

df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

df1.shape
df2.shape
df1.columns = ['id','tittle','cast','crew']
df1.head()
df2=df2.merge(df1,on="id")

df2.head()
c=df2['vote_average'].mean()

print(c)
m=df2['vote_count'].quantile(0.9)

m
qualified_movies=df2.copy().loc[df2['vote_count']>=m]

qualified_movies.shape
def imdbscore(x,m=m,c=c):

    v=x['vote_count']

    r=x['vote_average']

    #now remember the imdb formula 

    return (v/(v+m) * r) + (m/(m+v) * c)
#now apply this to our dataframe qualified_movies through apply() function.

qualified_movies['score']=qualified_movies.apply(imdbscore,axis=1)
#Sort movies based on score calculated above

qualified_movies = qualified_movies.sort_values('score', ascending=False)



#Print the top 15 movies

qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
popular=df2.sort_values(by='popularity',ascending=False)

popular[['title', 'vote_count', 'vote_average']].head()
df2['overview'].head(5)
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'.This is very important because they occurs in all the

#overviews and that will result in getting false similarity scores

tfidf = TfidfVectorizer(stop_words='english')



df2['overview'].isna().sum()
#Replace NaN with an empty string

df2['overview'] = df2['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(df2['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
    # Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

#if any duplicate indexes are present they will get eliminated 
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title(the movie for which we need recommendations)

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores.Here as it is list to sort --sorted() function is used.

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return df2['title'].iloc[movie_indices]

get_recommendations('The Avengers')
# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']#since we are considering actors,directors present in queue and genre and keywords about movie

for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
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

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
# Define new director, cast, genres and keywords features that are in a suitable form.

df2['director'] = df2['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)
df2.head()
# Function to convert all strings to lower case and strip names of spaces

def clean_data(li):

    if isinstance(li, list):#To avoid errors

        return [str.lower(i.replace(" ", "")) for i in li]#for get_list() function.

    else:

        #Check if director exists. If not, return empty string

        if isinstance(li, str):

            return str.lower(li.replace(" ", ""))#for director

        else:

            return ''
# Apply clean_data function to your features.

features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    df2[feature] = df2[feature].apply(clean_data)
def create_solution(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['solution'] = df2.apply(create_solution, axis=1)
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')#not considering the words like 'the','a'etc..,.

count_matrix = count.fit_transform(df2['solution'])
# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)#we can use linear kernel as above said also
# Reset index of our main DataFrame and construct reverse mapping as before

df2 = df2.reset_index()#resetting

indices = pd.Series(df2.index, index=df2['title'])#now again mapping the movies and id's...
df2.head()
get_recommendations('The Dark Knight Rises', cosine_sim2)
from surprise import Reader, Dataset, SVD

reader = Reader()

ratings = pd.read_csv('../input/for-collaborative/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]#actual ratings
svd.predict(1, 31)