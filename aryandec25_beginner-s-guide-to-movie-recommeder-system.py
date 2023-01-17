import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from ast import literal_eval



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input/tmdb-movie-metadata/"))
movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
credits.head(1)
movies.head(1)
credits.columns=['id','title','cast','crew']
movies=movies.merge(credits,on='id')
movies.shape
movies.info()
movies.describe()
movies['genres']=movies['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x])
movies['title']=movies['title_x']
movies.drop(['title_x','title_y'],axis=1,inplace=True)
vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

C= vote_averages.mean()

C
m = vote_counts.quantile(0.95)

m
movies['year']=movies['release_date'].apply(lambda x: str(x).split('-')[0]  if x != np.nan else np.nan)
movies.head(1)
qualified=movies[(movies['vote_count']>=m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity','genres']]
qualified.shape
def weighted_ratio(x):

    v=x['vote_count']

    R=x['vote_average']

    return (((v/(v+m))*R) + ((m/(v+m))*C))
qualified['wr']=qualified.apply(weighted_ratio,axis=1)
qualified['wr']=np.round(qualified['wr'],2)
qualified=qualified.sort_values(by='wr',ascending=False)
gen=movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

gen.name='genre'

gen_movies=movies.drop('genres',axis=1).join(gen)
gen_movies.head(3)
def build_chart(genre,percentile=0.95):

    df=gen_movies[gen_movies['genre'] == genre]

    vote_counts=df[df['vote_count'].notnull()]['vote_count'].astype(int)

    vote_averages=df[df['vote_average'].notnull()]['vote_average'].astype(int)

    c=vote_averages.mean()

    m=vote_counts.quantile(percentile)

    

    qualified=df[(df['vote_count']>=m) & df['vote_count'].notnull() & df['vote_average'].notnull()][['title','year','vote_count','vote_average','popularity']]

    



    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(250)

    

    return qualified
build_chart('Romance').head(10)
build_chart('Action')
movies['overview'].head()
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf=TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

movies['overview']=movies['overview'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data 

tfidf_matrix=tfidf.fit_transform(movies['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel



cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
def content_based(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]
content_based('Black Swan')
content_based('The Avengers')
from surprise import Reader, Dataset, SVD

reader = Reader()

ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()
data=Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

data
svd=SVD()

trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1,206,5)