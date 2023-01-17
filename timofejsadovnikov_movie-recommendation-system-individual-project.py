import pandas as pd
import numpy as np
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#IMDb dataset
imdb = pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')

#MovieLens dataset
ratings = pd.read_csv('../input/movielens-latest-small/ratings.csv')
movies = pd.read_csv('../input/movielens-latest-small/movies.csv')
id_links = pd.read_csv('../input/movielens-latest-small/links.csv')
imdb.columns
imdb.shape
imdb.head()
movies.columns, ratings.columns, id_links.columns
movies.shape, ratings.shape, id_links.shape
movies.head() 
ratings.head()
id_links.head()
imdb.columns
#Picking all movie features that I reckon to be most valuable for the recommendation system
df = imdb[['imdb_title_id','title', 'year', 'genre', 'language', 'director', 'writer', 'actors', 'description']]
#Check for null values
df.isnull().sum()
#fill null values with whitespace as words like "NaN" or "Missing Values" can increase similarity between movies with missing values
df = df.fillna(' ')
#Getting keywords from description by filtering stopwords from the nltk library
stop = set(stopwords.words('english'))
df['keywords'] = df['description'].str.split().apply(lambda x: [item for item in x if item not in stop])
df['keywords'] = df['keywords'].str.join(' ')
df.drop('description', axis=1, inplace=True)
#Formatting dataframe and its' values for bag of words
df['imdb_title_id'] = df['imdb_title_id'].str.extract(r'([1-9][0-9]*|0)\b', expand=False)
df = df.rename(columns={'imdb_title_id':'movie_id'})
df['genre'] = df['genre'].str.replace(',', '')
df['actors'] = df['actors'].str.replace(',', '')
df['writer'] = df['writer'].str.replace(',', '')
df['director'] = df['director'].str.replace(',', '')
df['keywords'] = df['keywords'].str.lower()
df = df.astype(str)
df.head()
#create function that combines features into bag of words

# def combine_features(row):
#     return row['director'] + " " + row['genre'] + " "+ row['year'] + " " + row['keywords'] + " " + row['actors'] + " " + row['language'] + " " + row['writer']

def combine_features(row):
    return row['director'] + " " + row['genre'] + " " + row['actors'] + " " + row['keywords']

#combine features into bag of words
df['bow'] = df.apply(combine_features, axis=1)
df.head()
#Using a smaller sample of dataset for the model incase of memory- and runtime errors
df1 = df.head(10000)
#convert bag of words into a matrix of token counts
cv = CountVectorizer()
count_matrix = cv.fit_transform(df1['bow'])
#Find cosine similarity between the movies in the sparse matrix
cos_sim = cosine_similarity(count_matrix)
#DataFrame with indices to retrieve movie titles and other features
features_imdb = df1[['movie_id','title','year']].reset_index()


def recommend_sim_features(movie, rating):
    
    movie_index = features_imdb[features_imdb['title'] == movie]['index'].values[0] #get index from movie title to retrieve similarity scores from the similarity matrix
    
    sim_scores = pd.Series(cos_sim[movie_index]).sort_values(ascending=False) #find the highest similarity scores
    
    sim_feat_movies = pd.DataFrame({ 'score' : sim_scores[1:101]*(rating-2.5)}).reset_index() #Create dataframe with the movies' 100 highest scored movies (itself excluded)
    sim_feat_movies = sim_feat_movies.merge(features_imdb) #merge with features_imdb to retrieve movie_id and year
    sim_feat_movies = sim_feat_movies[['movie_id','title', 'year', 'score']] #change order of columns and drop index
    
    return sim_feat_movies

#Printing 10 random titles from dataframe to choose from and test in our function
df1.iloc[np.random.choice(np.arange(len(df1['year'].astype(int) > 2000)), 10, False)]['title']
recommend_sim_features('The Rocking Horse Winner', 5).head()
#Return a dataframe with top 'n' recommended movies from a list of tuples with movies and their respective ratings.
#Example: movies_watched = [(Sharknado, 5), (Sharknado 2, 5), (Pulp Fiction, 1)]

def movie_recommender_cb(user_movies, n=100):
    
    recommended_movies = pd.DataFrame() #create dataframe to append similar movies
    
    #Loop through list one by one and append every returned dataframe to recommended_movies
    for movie, rating in user_movies:
        #Error management in case of a movie in the list can't be found in the dataset
        try:
            recommended_movies = recommended_movies.append(recommend_sim_features(movie, rating))
        except: print(movie + ' was not in the IMDb dataset')
            
    #Group movies and sum the score, sort by highest score descending
    recommended_movies = recommended_movies.groupby(['movie_id', 'title', 'year']).sum().sort_values('score', ascending=False).reset_index()
    
    return recommended_movies.head(n) #return n (default=100) most similar movies
#Random test list of movie ratings
test_list_imdb = [('High Society', 4), ("Paris Playboys", 3), ('Here Come the Marines', 4)]
#return top 5 highest scored movies
movie_recommender_cb(test_list_imdb, 5)
#Using regex pattern to extract year from title and removing year from title.
movies['year'] = movies['title'].str.extract('(\d{4})', expand = False)
movies['title'] = movies['title'].str.replace('(\s.\d{4}.)', '')
#Combine datasets to one dataframe
df2 = pd.merge(ratings, movies)
df2 = df2.merge(id_links)
df2.head()
df2 = df2.rename(columns={'imdbId':'movie_id', 'userId':'user_id'})
df2.drop(['movieId','timestamp','genres','tmdbId'], axis=1, inplace=True)

#count of unique users
x_1 = df2['user_id'].nunique()
#count of unique movies
x_2 = df2['movie_id'].nunique()
#count of ratings
x_3 = len(df2)
#Average rating count by movie
x_4 = df2['rating'].count()  / df2['user_id'].nunique()
#Average rating count by movie
x_5 = df2['rating'].count()  / df2['movie_id'].nunique()

print(f'Count of unique users: {x_1} \nCount of unique movies: {x_2} \nCount of ratings: {x_3} \nAverage rating count by users: {x_4} \nAverage rating count by movie: {x_5}')
#Pivoting our dataframe to match the format of a Movie-Based CF
user_ratings = df2.pivot_table(index=['user_id'], columns=['title'], values='rating')
user_ratings.head()
#As we saw the mean rating count by movie was 10 which was quite small, so let's keep that as a minimum instead 
user_ratings = user_ratings.dropna(thresh=10 ,axis=1).fillna(0, axis=1) # fill NaN values and drop movies with ratings count less than thresh(10)
user_ratings.head()
#Apply Pearsons Correlation to find movie similarity based on users' ratings
movie_similarity = user_ratings.corr(method='pearson')
movie_similarity.head()
#Function that retrieves the movies' correlation to other movies and scoring it based on the rating.
def get_sim_ratings(movie, rating):
    sim_ratings = movie_similarity[movie]*(rating-2.5)
    sim_ratings = sim_ratings.sort_values(ascending=False)
    return sim_ratings

movie_recommender_cb(test_list_imdb, 2)
#returns dataframe with movie_id, title, year and score
get_sim_ratings('Toy Story', 5)[:10]
#Returns series object with title and respective score
#DataFrame to retrieve relevant movie features
features_movielens = df2[['movie_id','title','year']].drop_duplicates()

def recommend_sim_ratings(movie, rating):
    
    sim_ratings = get_sim_ratings(movie, rating).to_frame().reset_index() #get the series output and convert to dataframe using to_frame() and extract titles column from index
    sim_ratings.rename(columns={movie:'score'}, inplace=True) #naming the column with score values to 'score' as it's otherwise set to be the movie title
    
    sim_ratings = sim_ratings.merge(features_movielens) #merge to retrieve movie_id and year
    sim_ratings = sim_ratings[['movie_id', 'title', 'year', 'score']] #change order of columns to match the ouput order of movie_recommender_cb()
    sim_ratings = sim_ratings.sort_values('score', ascending=False)[1:101] #return
    return sim_ratings

#testing with top 3 most similar movies
recommend_sim_ratings('Toy Story', 5)[:3]
def movie_recommender_cf(user_movies, n=100):
    
    recommended_movies = pd.DataFrame() #create dataframe to append similar movies
    
    #Loop through list one by one and append every returned dataframe to recommended_movies
    for movie, rating in user_movies:
        #Error management in case of a movie in the list can't be found in the dataset
        try:
            recommended_movies = recommended_movies.append(recommend_sim_ratings(movie, rating))
        except: print(movie + ' was not in the MovieLens dataset')
            
    #Group movies and sum the score, sort by highest score descending
    recommended_movies = recommended_movies.groupby(['movie_id', 'title', 'year']).sum().sort_values('score', ascending=False).reset_index()
    
    return recommended_movies.head(n) #return n most similar movies
#Printing 10 random titles from dataframe to choose from and test in our function
df2.iloc[np.random.choice(np.arange(len(df2)), 10, False)]['title']
test_list_movielens = [('Toy Story', 5), ('Back to the Future', 4), ('Harry Potter and the Deathly Hallows: Part 2', 4)]

#return top 5 most highest scored movies
movie_recommender_cf(test_list_movielens, 5)
def hybrid_movie_recommender(user_movies, n=10, score_factor=0.5):
    if score_factor < 1 and score_factor > 0:
        print(f'Content Based impact: {round(score_factor*100)}% \nMovie-Based CF impact: {round((1-score_factor)*100)}%')
        cb = movie_recommender_cb(user_movies)
        cb['score'] = cb['score']*(1 + score_factor)
        cf = movie_recommender_cf(user_movies)
        cf['score'] = cf['score']*(1 + (1 - score_factor))
    else: 
        print('Score factor was set to default 0.5, must be between 0 and 1')
        score_factor=0.5
        cb = movie_recommender_cb(user_movies)
        cb['score'] = cb['score']*(1 + score_factor)
        cf = movie_recommender_cf(user_movies)
        cf['score'] = cf['score']*(1 + (1 - score_factor))
    
    recommended_movies = cf.append(cb).groupby(['movie_id', 'title', 'year']).sum().sort_values('score', ascending=False).reset_index() #merge, group, sum and sort

    return recommended_movies.head(n) #return n most similar movies (n default=10)


hybrid_movie_recommender((test_list_movielens + test_list_imdb), 10, 0.55) # testing hybrid recommender with score factor of 0.55
#Commonly shared movies
df1[['movie_id', 'title']].drop_duplicates().append(df2[['movie_id', 'title']].drop_duplicates()).value_counts().sort_values(ascending=False)
