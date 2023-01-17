%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



#import libraries specific to recommendation system

from surprise import KNNWithMeans

from surprise import Dataset

from surprise import accuracy

from surprise.model_selection import train_test_split



import warnings

warnings.filterwarnings("ignore")
movies=pd.read_csv('../input/movies-data/movies_metadata.csv')

ratings=pd.read_csv('../input/movies-data/ratings_small.csv')
movies.head(2)
ratings.head()
movies.info()
movies.budget =pd.to_numeric(movies.budget, errors='coerce')
movies.describe()
# Exploring the languages of the movies

pd.unique(movies['original_language'])
movies = movies[['id', 'original_title', 'original_language','vote_average','vote_count','adult','budget','revenue','runtime','status']]

movies = movies.rename(columns={'id':'movieId'})
mean_budget = movies['budget'].mean(skipna=True)

print (mean_budget)
movies['budget']=movies.budget.mask(movies.budget == 0,mean_budget)
mean_revenue = movies['revenue'].mean(skipna=True)

print (mean_revenue)
movies['revenue']=movies.revenue.mask(movies.revenue == 0,mean_revenue)
# Filtering English movie only

movies = movies[movies['original_language']== 'en'] 

movies.head()
movies.dtypes

ratings.dtypes

movies.movieId =pd.to_numeric(movies.movieId, errors='coerce')

ratings.movieId = pd.to_numeric(ratings.movieId, errors= 'coerce')
#creating a single dataframe merging the movie_data and ratings_data

df= pd.merge(ratings, movies, on='movieId', how='inner')
df.info()
df.isnull().sum() # or df.isna.sum()
df['status'].fillna(df['status'].mode()[0], inplace=True)
df['runtime'].fillna(df['runtime'].mode()[0], inplace=True)
df.isnull().sum()
df.describe()
ratings = pd.DataFrame(df.groupby('original_title')['rating'].mean().sort_values(ascending=False))

ratings.head(20)
ratings['number_of_ratings'] = df.groupby('original_title')['rating'].count()

ratings.head()
import matplotlib.pyplot as plt

#%matplotlib inline

ratings['rating'].hist(bins=50)

plt.title('Histogram');

plt.xlabel('Rating')

plt.ylabel('Number of movies')
ratings['number_of_ratings'].hist(bins=100)

plt.title('Histogram');

plt.xlabel('Number of ratings')

plt.ylabel('Number of movies')
import seaborn as sns

sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
from surprise import Dataset, Reader

reader = Reader(rating_scale=(0, 5))

data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
#use user based true/false to switch between user-based or item-based collaborative filters

trainset,testset=train_test_split(data,test_size=.15)
algo=KNNWithMeans(k=50,sim_options={'name':'pearson_baseline','user_based':True})

algo.fit(trainset)
#We can now query for speicific predictions

userId=str(196) #raw user id

movieId=str(302) #raw item id

# get a prediction for specific users and items

pred=algo.predict(userId,movieId,verbose=True) 
#run the trained model against the tesset

test_pred=algo.test(testset)

test_pred
accuracy.rmse(test_pred)
def MovieRecommender(df, MovieName, No_of_recommendation):

    movie_matrix = df.pivot_table(index='userId', columns='original_title', values='rating').fillna(0)

    movie_matrix.head(10)

    movie_user_rating = movie_matrix[MovieName]

    similar_to_movie=movie_matrix.corrwith(movie_user_rating)

    corr = pd.DataFrame(similar_to_movie, columns=['Correlation'])

    corr.dropna(inplace=True)

    corr = corr.join(ratings['number_of_ratings'])

    c=corr[corr['number_of_ratings'] > 50].sort_values(by='Correlation', ascending=False).head(No_of_recommendation)

    print(c)

    return c
c=MovieRecommender(df, MovieName='The Million Dollar Hotel', No_of_recommendation=5)