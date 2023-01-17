# import libraties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Reading movies file

movies = pd.read_csv('/kaggle/input/movielens-latest-small/movies.csv', encoding='latin-1')
movies.head()
print('Shape:', movies.shape)
print('Movie ids:', movies.movieId.nunique())
print('Titles:', movies.title.nunique())
# Reading ratings file

ratings = pd.read_csv('/kaggle/input/movielens-latest-small/ratings.csv', encoding='latin-1')
ratings.head()
print('Shape:', ratings.shape)
print('Movie ids:', ratings.movieId.nunique())
print('Number of users:', ratings.userId.nunique())
# These are the movies that have been stored with two different ids

movies.title.value_counts().sort_values(ascending=False).head(5)
# getting the ids of a movie
movies[movies['title'] == 'Saturn 3 (1980)']
# checking those ids in 'rating' dataframe and count which id is most watched
ratings[(ratings['movieId'] == 2851) | (ratings['movieId'] == 168358)]['movieId'].value_counts()
# deleting the id who is less watched
movies = movies[movies['movieId'] != 168358]
# getting the ids of a movie
movies[movies['title'] == 'Confessions of a Dangerous Mind (2002)']
# checking those ids in 'rating' dataframe and count which id is most watched
ratings[(ratings['movieId'] == 6003) | (ratings['movieId'] == 144606)]['movieId'].value_counts()
# deleting the id who is less watched
movies = movies[movies['movieId'] != 144606]
# getting the ids of a movie
movies[movies['title'] == 'Emma (1996)']
# checking those ids in 'rating' dataframe and count which id is most watched
ratings[(ratings['movieId'] == 838) | (ratings['movieId'] == 26958)]['movieId'].value_counts()
# deleting the id who is less watched
movies = movies[movies['movieId'] != 26958]
# getting the ids of a movie
movies[movies['title'] == 'War of the Worlds (2005)']
# checking those ids in 'rating' dataframe and count which id is most watched
ratings[(ratings['movieId'] == 34048) | (ratings['movieId'] == 64997)]['movieId'].value_counts()
# deleting the id who is less watched
movies = movies[movies['movieId'] != 64997]
# getting the ids of a movie
movies[movies['title'] == 'Eros (2004)']
# checking those ids in 'rating' dataframe and count which id is most watched
ratings[(ratings['movieId'] == 32600) | (ratings['movieId'] == 147002)]['movieId'].value_counts()
# deleting the id who is less watched
movies = movies[movies['movieId'] != 147002]
movies_ratings = pd.merge(movies, ratings, on='movieId')
movies_ratings.head()
movies_ratings.shape
# dropping 'timestamp' column
movies_ratings = movies_ratings[['userId','movieId', 'title', 'genres', 'rating']]

# sort the dataframe according to 'userId' and then 'movieId'
movies_ratings.sort_values(['userId','movieId'], inplace=True)

# resetting the index
movies_ratings.reset_index(drop=True, inplace=True)

# top 10 rows
movies_ratings.head(10)
# number of customer ids
movies_ratings.userId.nunique()
# number of movie ids
movies_ratings.movieId.nunique()
# number of movie titles
movies_ratings.title.nunique()
# removing the extra whitespaces(if any) from the column 'title' and 'genres'
movies_ratings['title'] = movies_ratings['title'].str.strip()
movies_ratings['genres'] = movies_ratings['genres'].str.strip()

# extracting the 'year'
movies_ratings['year'] = movies_ratings['title'].str[-5:-1]
movies_ratings.year.unique()
movies_ratings['year'] = movies_ratings['year'].replace('irro',2011)
movies_ratings['year'] = movies_ratings['year'].replace('atso',2011)
movies_ratings['year'] = movies_ratings['year'].replace(' Bab',2017)
movies_ratings['year'] = movies_ratings['year'].replace('ron ',2017)
movies_ratings['year'] = movies_ratings['year'].replace('r On',2018)
movies_ratings['year'] = movies_ratings['year'].replace('lon ',1994)
movies_ratings['year'] = movies_ratings['year'].replace('imal',2016)
movies_ratings['year'] = movies_ratings['year'].replace('osmo',2019)
movies_ratings['year'] = movies_ratings['year'].replace('he O',2016)
movies_ratings['year'] = movies_ratings['year'].replace(' Roa',2015)
movies_ratings['year'] = movies_ratings['year'].replace('ligh',2016)
movies_ratings['year'] = movies_ratings['year'].replace('erso',2016)
# movieIds where genre is missing

movies_ratings[movies_ratings['genres']=='(no genres listed)'].drop_duplicates('movieId')['movieId'].values
movies_ratings.loc[movies_ratings['movieId']==122896,"genres"] = 'Adventure|Action|Fantasy'
movies_ratings.loc[movies_ratings['movieId']==114335,"genres"] = 'Fantasy'
movies_ratings.loc[movies_ratings['movieId']==174403,"genres"] = 'Documentary|Biography'
movies_ratings.loc[movies_ratings['movieId']==172591,"genres"] = 'Crime|Drama|Thriller'
movies_ratings.loc[movies_ratings['movieId']==176601,"genres"] = 'Sci-Fi|Fantasy'
movies_ratings.loc[movies_ratings['movieId']==155589,"genres"] = 'Comedy'
movies_ratings.loc[movies_ratings['movieId']==147250,"genres"] = 'Crime|Mystery|Romance'
movies_ratings.loc[movies_ratings['movieId']==171749,"genres"] = 'Animation|Crime|Drama'
movies_ratings.loc[movies_ratings['movieId']==173535,"genres"] = 'Crime|Drama|Mystery'
movies_ratings.loc[movies_ratings['movieId']==134861,"genres"] = 'Comedy'
movies_ratings.loc[movies_ratings['movieId']==159161,"genres"] = 'Comedy'
movies_ratings.loc[movies_ratings['movieId']==171631,"genres"] = 'Documentary|Comedy'
movies_ratings.loc[movies_ratings['movieId']==171891,"genres"] = 'Documentary'
movies_ratings.loc[movies_ratings['movieId']==142456,"genres"] = 'Comedy|Fantasy'
movies_ratings.loc[movies_ratings['movieId']==181413,"genres"] = 'Documentary'
movies_ratings.loc[movies_ratings['movieId']==159779,"genres"] = 'Comedy|Fantasy'
movies_ratings.loc[movies_ratings['movieId']==169034,"genres"] = 'Musical'
movies_ratings.loc[movies_ratings['movieId']==171495,"genres"] = 'Sci-Fi'
movies_ratings.loc[movies_ratings['movieId']==172497,"genres"] = 'Action|Sci-Fi'
movies_ratings.loc[movies_ratings['movieId']==166024,"genres"] = 'Drama|Music'
movies_ratings.loc[movies_ratings['movieId']==167570,"genres"] = 'Drama|Fantasy|Mystery'
movies_ratings.loc[movies_ratings['movieId']==129250,"genres"] = 'Comedy'
movies_ratings.loc[movies_ratings['movieId']==143410,"genres"] = 'Action|Drama|War'
movies_ratings.loc[movies_ratings['movieId']==149330,"genres"] = 'Animation|Sci-Fi'
movies_ratings.loc[movies_ratings['movieId']==182727,"genres"] = 'Musical'
movies_ratings.loc[movies_ratings['movieId']==152037,"genres"] = 'Romance|Musical'
movies_ratings.loc[movies_ratings['movieId']==165489,"genres"] = 'Drama|Animation|History'
movies_ratings.loc[movies_ratings['movieId']==141866,"genres"] = 'Horror|Music|Thriller'
movies_ratings.loc[movies_ratings['movieId']==122888,"genres"] = 'Action|Adventure|Drama'
movies_ratings.loc[movies_ratings['movieId']==156605,"genres"] = 'Comedy|Drama|Romance'
movies_ratings.loc[movies_ratings['movieId']==141131,"genres"] = 'Action|Mystery|Sci-Fi'
movies_ratings.loc[movies_ratings['movieId']==181719,"genres"] = 'Biography|Drama'
movies_ratings.loc[movies_ratings['movieId']==132084,"genres"] = 'Drama|Romance'
movies_ratings.loc[movies_ratings['movieId']==161008,"genres"] = 'Drama|Music|Romance'
# replacing 'musical' with 'music' as both have same meaning
movies_ratings['genres'] = movies_ratings['genres'].str.replace('Musical','Music')
# converting string to int
movies_ratings['year'] = movies_ratings['year'].astype(int)
movies_ratings.info()
movies_ratings.head()
# store the column in different dataframe
genre_df = movies_ratings[['genres']]

# splitting the columns
genre_df = genre_df['genres'].str.split('|', expand=True)

genre_df.head()
# changing the name of the columns
genre_df.rename(columns={0:'G1',1:'G2',2:'G3',3:'G4',4:'G5',5:'G6',6:'G7',7:'G8',8:'G9',9:'G10'}, inplace=True)
# create a function that return distinct genres from whole dataframe

def genre_name(dataframe):
    df = dataframe.copy()
    col = df.columns
    u = set()
    for i in col:
        s = set(df[i].value_counts().index)
        u = u.union(s)
    return(u)
# names of distinct genres (21 genres)
g = genre_name(genre_df)
g
# making columns of each of the genes with value either 1 or 0 in original dataframe 

for genre in g:
    movies_ratings[genre] = movies_ratings['genres'].apply(lambda x: 1 if genre in x else 0)
movies_ratings.head()
plt.figure(figsize=(10,5))
plt.hist(movies_ratings['rating'],bins=10, color='pink', alpha=0.7)
plt.xlabel('rating',size=12)
plt.xlim(0.5,5)
plt.ylim(0,30000)
plt.vlines(x=3.5, ymin=0, ymax=30000, color='red', label='Mean rating')
plt.ylabel('')
plt.title('count plot of ratings',size=18, color='red')
plt.legend()
plt.show()
genres_count = movies_ratings.iloc[:,6:].sum(axis=0).reset_index().rename(columns={'index':'genre',0:'count'})
genres_count.sort_values('count',ascending=False, inplace=True)

plt.figure(figsize=(15,5))
sns.barplot(x = genres_count['genre'], y=genres_count['count'], color='lightgreen')
plt.xticks(rotation=45)
plt.xlabel('Genres', size=12)
plt.ylabel('')
plt.title('Count plot of genres', size=18, color='green')
plt.show()
mr = movies_ratings.groupby('title')['title'].count().sort_values(ascending=False).head(15)

plt.figure(figsize=(10,5))
sns.barplot(y = mr.index, x=mr.values, color='skyblue')
plt.ylabel('')
plt.title('15 Most watched Movies', size=18, color='blue')
plt.show()
user = movies_ratings.groupby('userId')['title'].count().sort_values(ascending=False).head(20)

plt.figure(figsize=(15,5))
user.plot(kind="bar", color="orange", alpha=0.5)
plt.title("Top 20 users according to watched history", size=18, color='orange')
plt.xlabel('User Id', size=12)
plt.xticks(rotation=0)
plt.show()
def best_movie(dataframe):
    """
    This function will return a dataframe in which there are 3 columns. The first column is year.
    The second column is number of movies released in that year. (according to data we have)
    Third column is the most watched movie of that year. (in the given data)
    It only takes one argument which is data.
    """
    df = dataframe.copy()
    movieid = df.year.unique()
    year = list()
    nMovies= list()
    mostWatched = list()
    for i in movieid:
        year.append(i)
        nMovies.append(df[df['year']==i]['title'].nunique())
        mostWatched.append(df[df['year']==i]['title'].value_counts().index[0])
    
    df1 = pd.DataFrame({'year':year,'nMoviesReleased':nMovies, 'mostWatchedMovie':mostWatched})
    df1.sort_values('year', inplace=True)
    return(df1)
# calling the function and reading its top 10 rows
yearWiseBestMovie = best_movie(movies_ratings)
yearWiseBestMovie.head(10)
# consider only user with id=1 and subset its data from main dataframe to another dataframe

user_1 = movies_ratings[movies_ratings['userId']==1]
user_1.head()
user_1.shape
# step 1

user_rating = user_1['rating']
user_rating.head()
user_rating.shape
# step 2

movie_matrix = user_1.iloc[:,6:]
movie_matrix.head()
# step 3

weighted_genre_matrix = movie_matrix.multiply(user_rating, axis=0)
weighted_genre_matrix.head()
weighted_genre_matrix = pd.concat((user_1.iloc[:,:6], weighted_genre_matrix), axis=1)
weighted_genre_matrix.head()
# step 4

wg = weighted_genre_matrix.iloc[:,6:].sum(axis=0)/weighted_genre_matrix.iloc[:,6:].sum(axis=0).sum()
wg
# number of movies watched by the user is 232
# we store all those movies id in m
m = movies_ratings[movies_ratings['userId']==1]['movieId'].values
m = set(m)
len(m)
# total number of movies in the data is 9719
# we store all the movie ids in b
b = movies_ratings.movieId.unique()
b = set(b)
len(b)
# subtracting the watched movies from total movies
# 'r' is a set of movie ids which the user_1 didn't watch
r = b-m
r = list(r)
len(r)
# step 5

# all the data except of user_1
other_users = movies_ratings[movies_ratings['userId'] != 1]
other_users.head()
# keeping only those movies which user has not seen
other_users = other_users[other_users['movieId'].isin(r)]
other_users.shape
# storing it in 'movie_matrix_other'
movies_matrix_other = other_users.copy()
movies_matrix_other = movies_matrix_other.drop(['userId','rating'], axis=1)
movies_matrix_other.head()
movies_matrix_other.shape
movies_matrix_other.movieId.nunique()
movies_matrix_other.title.nunique()
# removing the duplicate rows of the movies
movies_matrix_other = movies_matrix_other.drop_duplicates()
movies_matrix_other.shape
# step 6

weighted_movies_matrix_other = pd.concat((movies_matrix_other.iloc[:,:4], movies_matrix_other.iloc[:,4:].multiply(wg)), axis=1)
weighted_movies_matrix_other.head()
# step 7

weighted_movies_matrix_other["final_score"] = weighted_movies_matrix_other.iloc[:,4:].sum(axis=1)
weighted_movies_matrix_other.sort_values('final_score', ascending=False, inplace=True)
topMovie = weighted_movies_matrix_other
topMovie.head()
list(topMovie[topMovie["genres"].str.contains("|")]["title"].head(10))
def recommended_movies(df):
    id = int(input('User ID:'))
    genre = input('Enter preferred Genre\n(otherwise press enter): ')
    top_movies = int(input('How many movies:'))
    
    user = df.copy()
    user = user[user['userId'] == id]
    user_rating = user['rating']
    movies_matrix = user.copy()
    weighted_genre_matrix = movies_matrix.iloc[:,6:].multiply(user_rating, axis=0)
    weighted_genre_matrix = pd.concat((movies_matrix.iloc[:,:6], weighted_genre_matrix), axis=1)
    wg = weighted_genre_matrix.iloc[:,6:].sum(axis=0)/weighted_genre_matrix.iloc[:,6:].sum(axis=0).sum()
    m = df[df['userId'] == id]['movieId'].values
    m = set(m)
    b = df.movieId.unique()
    b = set(b)
    r = b-m
    r = list(r)
    other_users = df[df['userId'] != id]
    other_users = other_users[other_users['movieId'].isin(r)]
    movies_matrix_other = other_users.copy()
    movies_matrix_other = movies_matrix_other.drop(['userId','rating'], axis=1)
    movies_matrix_other = movies_matrix_other.drop_duplicates()
    movies_matrix_other = pd.concat((movies_matrix_other.iloc[:,:4], movies_matrix_other.iloc[:,4:].multiply(wg)),axis=1)
    movies_matrix_other["final_score"] = movies_matrix_other.iloc[:,4:].sum(axis=1)
    movies_matrix_other.sort_values('final_score', ascending=False, inplace=True)
    topMovie = movies_matrix_other
    
    l = (list(topMovie[topMovie["genres"].str.contains(genre,case=False)]["title"].head(top_movies)))
    if len(l)==0:
        print('\nYou have not watched any movie of "{}" genre.'.format(genre))
        print('SORRY, NO RECOMMENDATION!')
    return(l)
recommended_movies(movies_ratings)