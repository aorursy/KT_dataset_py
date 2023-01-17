import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from surprise import SVD, Reader, Dataset 

from surprise.model_selection import cross_validate
df_credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df_credits.head(2)
df_movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df_movies.head(2)
df_credits.columns=['id', 'title', 'cast', 'crew']
df_movies.drop(['title'], axis=1, inplace=True)
df_movielens=pd.merge(df_credits,df_movies,on='id')
df_movielens.head(2)
df_movielens.shape
df_movielens['vote_average'].mean()
df_movielens['vote_count'].quantile(q=0.9)
df_filtered=df_movielens[df_movielens['vote_count']>df_movielens['vote_count'].quantile(q=0.9)]
df_filtered.shape
def movie_score(x):

    v=x['vote_count']

    m=df_movielens['vote_count'].quantile(q=0.9)

    R=x['vote_average']

    C=df_movielens['vote_average'].mean()

    return ((R*v)/(v+m))+((C*m)/(v+m))
df_filtered['score']=df_filtered.apply(movie_score, axis=1)
df_highscore=df_filtered.sort_values(by='score', ascending=False).head(10)
df_highscore[['title', 'vote_count', 'vote_average', 'popularity', 'score']]
df_popular= df_movielens.sort_values('popularity', ascending=False).head(10)
df_popular[['title', 'vote_count', 'vote_average', 'popularity']]
plt.figure(figsize=(8,6))

sns.barplot(y='title',x='popularity', data=df_popular, palette='viridis')

plt.xlabel("Popularity", fontsize=12)

plt.ylabel("Title", fontsize=12)

plt.title("Popular Movies", fontsize=15)

plt.show()
df_movielens['overview'].head()
df_movielens['overview'].isnull().sum()
df_movielens['overview'].fillna(' ', inplace=True)
tfidfv=TfidfVectorizer(analyzer='word', stop_words='english')

tfidfv_matrix=tfidfv.fit_transform(df_movielens['overview'])

print(tfidfv_matrix.todense())

tfidfv_matrix.todense().shape
cosine_sim1 = linear_kernel(tfidfv_matrix, tfidfv_matrix)
cosine_sim1.shape 
indices=pd.Series(data=list(df_movielens.index), index= df_movielens['title'] )
indices.head()
# Function that takes in movie title as input and outputs most similar movies

def content_recommendations(title, cosine_sim):

    

    # Get the index of the movie that matches the title

    idx = indices[title]

    

    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))

    

    # Sort the movies based on the similarity scores

    sim_scores.sort(key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores=sim_scores[1:11]

    

    # Get the movie indices

    ind=[]

    for (x,y) in sim_scores:

        ind.append(x)

        

    # Return the top 10 most similar movies

    tit=[]

    for x in ind:

        tit.append(df_movielens.iloc[x]['title'])

    return pd.Series(data=tit, index=ind)
content_recommendations('The Dark Knight Rises',cosine_sim1)
content_recommendations('The Avengers',cosine_sim1)
type(df_movielens['cast'].iloc[0])
features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    df_movielens[feature] = df_movielens[feature].apply(literal_eval)
type(df_movielens['cast'].iloc[0])
# Get the director's name from the crew feature. If director is not listed, return NaN

def get_director(x):

    for a in x:

        if a['job']=='Director':

            return a['name'] 

    return 'NaN'
# Get the list top 3 elements or entire list; whichever is more in cast, genres and keywords columns.

def get_top3(x):

    new=[]

    for a in x[:3]:

        new.append(a['name']) 

    return new

#Return empty list in case of missing/malformed data

    return []
df_movielens['director']=df_movielens['crew'].apply(lambda x: get_director(x))
df_movielens['actor']=df_movielens['cast'].apply(lambda x:get_top3(x))
df_movielens['genres']=df_movielens['genres'].apply(lambda x:get_top3(x))
df_movielens['keywords']=df_movielens['keywords'].apply(lambda x:get_top3(x))
df_movielens[['title', 'actor', 'director', 'keywords', 'genres']].head(3)
def clean_director(x):

    return x.lower().replace(' ','')
def clean_top3(x):

    new=[]

    for a in x:

        new.append(a.lower().replace(' ',''))

    return new
df_movielens['director']=df_movielens['director'].apply(lambda x: clean_director(x))
df_movielens['actor']=df_movielens['actor'].apply(lambda x:clean_top3(x))
df_movielens['keywords']=df_movielens['keywords'].apply(lambda x:clean_top3(x))
df_movielens['genres']=df_movielens['genres'].apply(lambda x:clean_top3(x))
df_movielens[['title', 'actor', 'director', 'keywords', 'genres']].head(3)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['actor']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df_movielens['soup'] = df_movielens.apply(create_soup, axis=1)
cv = CountVectorizer(stop_words='english')

cv_matrix = cv.fit_transform(df_movielens['soup'])
cosine_sim2 = cosine_similarity(cv_matrix, cv_matrix)
content_recommendations('The Dark Knight Rises', cosine_sim2)
content_recommendations('The Godfather', cosine_sim2)
from surprise import SVD, Reader, Dataset 

from surprise.model_selection import cross_validate
df_rating= pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

df_rating.head()
# We will use the famous SVD algorithm.

svd = SVD()
reader = Reader()
# Load the ratings_small dataset (download it if needed),

data = Dataset.load_from_df(df_rating[['userId', 'movieId', 'rating']], reader)
# Run 5-fold cross-validation and print the results

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#sample full trainset

trainset = data.build_full_trainset()
# Train the algorithm on the trainset

svd.fit(trainset)
df_rating[df_rating['userId'] == 1]
# predict ratings for the testset

svd.predict(uid=1, iid=302, r_ui=None)
# directly grab the estimated ratings for the testset

svd.predict(uid=1, iid=302, r_ui=None).est
df_movielens.columns=['movieId', 'title', 'cast', 'crew', 'budget', 'genres', 'homepage',

       'keywords', 'original_language', 'original_title', 'overview',

       'popularity', 'production_companies', 'production_countries',

       'release_date', 'revenue', 'runtime', 'spoken_languages', 'status',

       'tagline', 'vote_average', 'vote_count', 'director', 'actor', 'soup']
# Function that takes in movie title as input and outputs most similar movies

def hybrid_recommendations(userId, title):

    

    # Get the index of the movie that matches the title

    idx = indices[title]

    

    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim2[idx]))

    

    # Sort the movies based on the similarity scores

    sim_scores.sort(key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores=sim_scores[1:11]

    

    # Get the movie indices

    ind=[]

    for (x,y) in sim_scores:

        ind.append(x)

        

    # Grab the title,movieid,vote_average and vote_count of the top 10 most similar movies

    tit=[]

    movieid=[]

    vote_average=[]

    vote_count=[]

    for x in ind:

        tit.append(df_movielens.iloc[x]['title'])

        movieid.append(df_movielens.iloc[x]['movieId'])

        vote_average.append(df_movielens.iloc[x]['vote_average'])

        vote_count.append(df_movielens.iloc[x]['vote_count'])



        

    # Predict the ratings a user might give to these top 10 most similar movies

    est_rating=[]

    for a in movieid:

        est_rating.append(svd.predict(userId, a, r_ui=None).est)  

        

    return pd.DataFrame({'index': ind, 'title':tit, 'movieId':movieid, 'vote_average':vote_average, 'vote_count':vote_count,'estimated_rating':est_rating}).set_index('index').sort_values(by='estimated_rating', ascending=False)

hybrid_recommendations(1,'Avatar')
hybrid_recommendations(4,'Avatar')
content_recommendations('Avatar', cosine_sim2)