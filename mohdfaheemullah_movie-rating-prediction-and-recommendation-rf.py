%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")



# machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
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

df.head()
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
features=list(set(df.columns)-set(['rating'])-set(['original_language'])-set(['original_title'])) 

features



X1=df[features]

X= pd.get_dummies(X1)

X.astype(np.float64)

y=df['rating'].astype(np.int64)

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

y_train.shape
# Data scaling

scaler = StandardScaler()



# Fit on training set only.

scaler.fit(X_train)



# Apply transform to both the training set and the test set.

X_train = scaler.transform(X_train)



X_test = scaler.transform(X_test)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
acc_random_forest_test = round(random_forest.score(X_test, y_test) * 100, 2)

acc_random_forest_test
Y_pred
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