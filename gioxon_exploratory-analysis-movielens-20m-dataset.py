# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# movies data
column_names_movies = ['movie_id', 'title', 'genres']
movies = pd.read_csv('../input/movielens-20m-dataset/movie.csv', delimiter = ',', names = column_names_movies,
                     header = 0)
movies.head()
# ratings data
column_names_ratings = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', delimiter = ',', names= column_names_ratings
                      , header = 0)
ratings.head()
# check NaN values
print ("Number of movies Null values: ", movies.isnull().sum())
print ("Number of ratings Null values: ", ratings.isnull().sum())
# drop Null values
movies.dropna(inplace=True)

# and check again for Nan values
print ("Number of movies Null values: ", movies.isnull().sum())
# summary of the movies Dataframe
movies.info()
# summary of the ratings Dataframe
ratings.info()
# users data
column_names_users = ['user_id', 'gender', 'age', 'occupation', 'zipcode']
users = pd.read_csv('../input/movielens-1m-dataset-users-data/users.csv', delimiter = ';', names= column_names_users
                    , header = 0)
users.head(5)
# summary of the users Dataframe
users.info()
# split title and release year in separate columns
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.head()
# check for NaN values
print ("Number of movies Null values: ", movies.isnull().sum())

# and drop them
movies.dropna(inplace=True)

# verify they dropped
print ("Number of movies Null values: ", movies.isnull().sum())
# convert year column to integer 
year_asint = movies['year'].astype(int)
# add the int(year) column  
movies.drop(['year'], inplace = True, axis = 1)
movies['year'] = year_asint
movies.head()
# convert timestamp to date
date = pd.to_datetime(ratings['timestamp']).astype(str)

# add column date to ratings
ratings['date'] = date
ratings.head()
# remove the time from the date
ratings['date'] = [time[:10] for time in ratings['date']]
ratings.drop(['timestamp'], inplace = True, axis = 1)
ratings.head()
#  descriptive statistics of movies Dataframe
movies.describe()
#  descriptive statistics of ratings Dataframe
ratings.describe()
df = pd.merge(movies,ratings,on='movie_id')
df.head()
# summary of the df Dataframe
df.info()
# number of ratings with values equals or greater than 4.5
df[df['rating'] >= 4.5]['movie_id'].count()
# ration of the movies with values equals or greater than 4.5 
df[df['rating'] >= 4.5]['movie_id'].count() / ratings['rating'].count()
# number of ratings with values equals 5.0
df[df['rating'] == 5]['movie_id'].count()
# ration of the movies with values equals or greater than 5.0
df[df['rating'] == 5]['movie_id'].count() / ratings['rating'].count()
# The most rated movies
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
# create Dataframe: ratings count per movie
df_ratingCount = pd.DataFrame(df.groupby('title', as_index = False)['rating'].count()
                              .rename(columns={'rating' : 'ratingCount'}))
df_ratingCount.sort_values('ratingCount',ascending=False).head(5)
# create Dataframe: mean rating per movie
df_ratingMean = pd.DataFrame(df.groupby('title', as_index = False)['rating'].mean()
                              .rename(columns={'rating' : 'ratingMean'}))
df_ratingMean.sort_values('ratingMean',ascending=False).head(5)
# create Dataframe: ratings count and mean rating per movie
df_movies_ext = pd.merge(df_ratingCount,df_ratingMean, on='title')
df_movies_ext.sort_values('ratingCount',ascending=False).head(5)
# create Dataframe: ratings count per value
df_rating_dist= pd.DataFrame(ratings.groupby('rating', as_index = False)['user_id'].count()
                         .rename(columns={'user_id' : 'ratingCount'}))
df_rating_dist.head(10)
# the users with most ratings
ratings_per_user = ratings[['user_id', 'movie_id']].groupby('user_id').count()
ratings_per_user = ratings_per_user.rename(columns={'movie_id' : 'Total'})
ratings_per_user.sort_values('Total',ascending=False).head()
# users per gender 
users_by_gender = users.groupby('gender', as_index = False)['user_id'].count()
users_by_gender.rename(columns={'user_id' : 'Total'}).head()
# users per occupation (top 5)
users_by_occ =users.groupby('occupation', as_index = False)['user_id'].count()
users_by_occ = users_by_occ.rename(columns={'user_id' : 'Total'})
users_by_occ.sort_values('Total',ascending=False).head()
# most popular genres (with most ratings)
most_popular_genre = df.groupby('genres', as_index = False)['user_id'].count()
most_popular_genre = most_popular_genre.rename(columns={'user_id' : 'Total ratings'})
most_popular_genre.sort_values('Total ratings',ascending=False).head(10)
# movies distribution per genres
movies_genres = movies.groupby('genres', as_index = False)['movie_id'].count()
movies_genres = movies_genres.rename(columns={'movie_id' : 'Total ratings'})
movies_genres.sort_values('Total ratings',ascending=False).head(10)
# select plot style
plt.style.use('ggplot')

# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
# the graph indicates that most of the movies have less than 40000 ratings
plt.figure(figsize=(12,6))
df_movies_ext[df_movies_ext['ratingCount'] > 10000]['ratingCount'].hist(bins=100)
plt.title('Number of ratings per movie')
plt.xlabel('Number of ratings')
plt.ylabel('Number of movies')
# most movies have been released after 1990
plt.figure(figsize=(12,8))
movies[movies['year'] > 1960]['year'].hist(bins=150)
plt.title('Movies per year')
plt.xlabel('Year')
plt.ylabel('Number of movies')
# most of the students are students
sns.factorplot("occupation", data=users, aspect=3, kind="count", color="b").set_xticklabels(rotation=90)
# most of the grades are equals or greater than 3
sns.jointplot(x='rating',y='ratingCount',data=df_rating_dist,alpha=0.5)
# the movies with the most ratings tend to have better rankings
sns.jointplot(x='ratingMean',y='ratingCount',data= df_movies_ext,alpha=0.5)
# most users have less than 4000 ratings
plt.figure(figsize=(12,8))
ratings_per_user['Total'].hist(bins=100, edgecolor='black', log=True)
plt.title('Ratings per user')
plt.xlabel('Number of ratings given')
plt.ylabel('Number of users')
plt.xlim(0,)
plt.xticks(np.arange(0,10000,1000))
plt.show()
