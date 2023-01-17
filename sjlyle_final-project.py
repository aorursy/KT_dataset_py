import pandas as pd
import numpy as np
#read in the movies dataset
df= pd.read_csv('../input/movielens/ratings.csv',header=0,sep=',', usecols = [0,1,2],encoding = "ISO-8859-1")
#count number of rowsdf_pivot = df_small.pivot(index='Cust_Id',values='Rating',columns='Movie_Id')
len(df)
df.head()
#count how many reviews there are with each rating
count_ratings = df.groupby('rating',as_index=False).count()
count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)
count_ratings
import seaborn as sns
#plot distribution of ratings
colors = sns.color_palette("GnBu_d")
ax = sns.barplot(x='rating', y='perc_total', data=count_ratings,palette=colors)
ax.set_title(label='% of reviews by Rating', fontsize=20)
ax
#dictionarity of movies that I have chosen and rated
my_movies_dict = {'userId': [999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999,999999], 
                          'movieId': [318,104241,130073,111362,108945,108190,105884,69112,79132,103335,136020,136562,109374,112556,160438],
                          'rating': [4,4,4,4,2,3.5,3.5,3,4,4,5,4,5,3,3]}
#save as data frame
my_movies = pd.DataFrame.from_dict(my_movies_dict)
my_movies
#attach to existing ratings data frame
df = df.append(my_movies)
#check that new rows have been appended to the data set
df.loc[df['userId']==999999].head()
#count number of unique users in the ratings data set
n_users = df.userId.unique().shape[0]
#count number of unitque movies in the ratings data set
n_items = df.movieId.unique().shape[0]
print(len(df))
print(n_users) 
print(n_items)
#calculate sparsity as number of entries / total number of possible entries
sparsity = round(1.0 - len(df)/float(n_users*n_items),2)

print("The sparsity of the ratings data set is " + str(sparsity*100) + "%")
#read in the movies dataset
movies= pd.read_csv('../input/movielens/movies.csv',header=0,sep=',',encoding = "ISO-8859-1")
#view movies
print(movies.head())
n_movies = len(movies)
print(n_movies)
#Search for a particular movie
movies.loc[movies['title'].str.contains("Nym")]
#read in the links
links= pd.read_csv('../input/movielens/links.csv',header=0,sep=',',encoding = "ISO-8859-1")
#view links
links.head()
#read in the tags
tags= pd.read_csv('../input/movielens/tags.csv',header=0,sep=',',encoding = "ISO-8859-1")
#view tags
print(tags.head())
print(len(tags))
n_movies_with_tags = tags.movieId.unique().shape[0]
print(n_movies_with_tags)
print("There are " + str(round(100*n_movies_with_tags/n_movies,1)) + " percent of movies with at least one tag")
#create a string and append all genre values in the data frame
genre_list = ""
for index,row in movies.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
#inspect list of genres
new_list
movie_genres = movies.copy()
#for each genre in the list
for genre in new_list:
    #create a new column for the genre abd search if the genre is in the list of genres
    movie_genres[genre] = movie_genres.apply(lambda _: int(genre in _.genres), axis=1)
#inspect the final data frame
movie_genres.head()
#get the average rating per movie 
avg_movie_rating = pd.DataFrame(df.groupby('movieId')['rating'].agg(['mean','count']))
#create new movieId column from Index
avg_movie_rating['movieId']= avg_movie_rating.index
avg_movie_rating.head()
#get the number of movies by count
movies_with_rating = pd.DataFrame(avg_movie_rating.groupby('count')['movieId'].agg(['count']))
#calculate the percentage of total movies that have a specific number of ratings
movies_with_rating['perc_total']=round(movies_with_rating['count']*100/movies_with_rating['count'].sum(),1)
movies_with_rating.head()
import seaborn as sns
sns.distplot(avg_movie_rating['count'])
#np.percentile(movie_ratings['count'],50)
#len(movie_ratings)
len(avg_movie_rating.loc[avg_movie_rating['count']>=30])
#calculate the percentile count
np.percentile(avg_movie_rating['count'],70)
#Get the average movie rating across all movies 
avg_rating_all=df['rating'].mean()
avg_rating_all
#set a minimum threshold for number of reviews that the movie has to have
min_reviews=30
min_reviews
movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]
movie_score.head()
#create a function for weighted rating score based off count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)
movie_score.head()
#join movie details to movie ratings
movie_score = pd.merge(movie_score,movie_genres,on='movieId')
#join movie links to movie ratings
#movie_score = pd.merge(movie_score,links,on='movieId')
movie_score.head()
#list top scored movies over the whole range of movies
pd.DataFrame(movie_score.sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:10])
def best_movies_by_genre(genre,top_n):
    #return  print("The top " + str(top_n) +" " + genre + "movies are:")
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])
#run function to return top recommended movies by genre
best_movies_by_genre('Action',10)  
#join the movie names to the movies data set
ratings_movies = pd.merge(df,movies,on='movieId')
#print the new data set
ratings_movies.head()
def get_other_movies(movie_name):
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    return other_users_watched[:10]
get_other_movies('Gone Girl (2014)')
from sklearn.neighbors import NearestNeighbors
#only include movies with more than 10 ratings
movie_plus_10_ratings = avg_movie_rating.loc[avg_movie_rating['count']>=30]
print(len(df))
filtered_ratings = pd.merge(movie_plus_10_ratings, df, on="movieId")
len(filtered_ratings)
#create a matrix table with movieIds on the rows and userIds in the columns.
#replace NAN values with 0
movie_wide = filtered_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
movie_wide.head()
#specify model parameters
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
#fit model to the data set
model_knn.fit(movie_wide)
#select a random movie
#query_index = np.random.choice(movie_wide.index)
query_index=96079
#96079 for skyfall
#get the list of user ratings for a specific userId
query_index_movie_ratings = movie_wide.loc[query_index,:].values.reshape(1,-1)
#get the closest 6 movies and their distances from the movie specified
distances,indices = model_knn.kneighbors(query_index_movie_ratings,n_neighbors = 11)
#movies.head(304)
indices
#write a lopp that prints the similar movies for a specified movie.
for i in range(0,len(distances.flatten())):
    #get the title of the random movie that was chosen
    get_movie = movies.loc[movies['movieId']==query_index]['title']
    #for the first movie in the list i.e closest print the title
    if i==0:
        print('Recommendations for {0}:\n'.format(get_movie))
    else: 
        #get the indiciees for the closest movies
        indices_flat = indices.flatten()[i]
        #get the title of the movie
        get_movie = movies.loc[movies['movieId']==movie_wide.iloc[indices_flat,:].name]['title']
        #print the movie
        print('{0}: {1}, with distance of {2}:'.format(i,get_movie,distances.flatten()[i]))
movie_genres.head()
content_df = movie_genres.copy()
content_df.set_index('movieId')
content_df_drop = content_df.drop(columns=['movieId','title','genres'])
content_df_drop = content_df_drop.as_matrix()
content_df_drop
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(content_df_drop,content_df_drop)
cosine_sim
#create a series of the movie id and title
indicies = pd.Series(content_df.index, content_df['title'])
indicies
idx = indicies["Skyfall (2012)"]#
sim_scores = list(enumerate(cosine_sim[idx]))
# Sort the movies based on the similarity scores
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Get the scores of the 10 most similar movies
sim_scores = sim_scores[1:11]
# Get the movie indices
movie_indices = [i[0] for i in sim_scores]
pd.DataFrame(content_df[['title','genres']].iloc[movie_indices])
#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(df['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
item_indices.tail()
#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(df['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
user_indices.tail()
#join the movie indices
df_with_index = pd.merge(df,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
df_with_index.head()
#import train_test_split module
from sklearn.model_selection import train_test_split
#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
print(len(df_train))
print(len(df_test))
#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_train.itertuples():
    #set the value in the column and row to 
    #line[1] is movieId, line[2] is rating and line[3] is userId, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[2]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
train_data_matrix.shape
#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to 
    #line[1] is movieId, line[2] is rating and line[3] is userId, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[2]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
test_data_matrix.shape
pd.DataFrame(train_data_matrix).head()
from math import sqrt
df_test_benchmark = df_test.copy()
avg_rating_benchmark=df_train['rating'].mean()
df_test_benchmark['prediction']=avg_rating_benchmark
df_test_benchmark['diff'] = df_test_benchmark['prediction'] - df_test_benchmark['rating']
df_test_benchmark['diff_squared']=df_test_benchmark['diff']**2
sqrt(df_test_benchmark['diff_squared'].sum()/len(df_test_benchmark))
print(avg_rating_benchmark)
df_test_benchmark.head()
#create a table with movieIds on the rows and userIds in the columns.
#replace NAN values with 0
df_pivot = df_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
import sklearn
from sklearn.metrics.pairwise import pairwise_distances
#comput user similarities and item similarities with cosine
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
#len(user_similarity[1])
len(item_similarity[1])
#pd.DataFrame(item_similarity).head(10)
#pd.DataFrame(user_similarity).head(10)
#function to predict rating
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        #prediction is equal to the mean rating + similarty . ratings difference / sum of absolute similarity
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        #prediction is equal to the ratings x similarity / sum of the similarity
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
user_prediction = predict(train_data_matrix,user_similarity,type='user')
#item_prediction = predict(train_data_matrix,item_similarity,type='item')
user_prediction.max()
user_prediction.min()
#convert the prediction matrices into data frames
user_pred_df = pd.DataFrame(user_prediction)
#item_pred_df = pd.DataFrame(item_prediction)
#inspect the predictions
user_pred_df.head(20)
#item_pred_df.head()
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    prediction = prediction[ground_truth.nonzero()].flatten() 
    #select test values that are non-zero and flatten into 1 array
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    return sqrt(mean_squared_error(prediction, ground_truth))
print("User-based CF RMSE: " + str(rmse(user_prediction, test_data_matrix)))
print("Item-based CF RMSE: " + str(rmse(item_prediction, test_data_matrix)))
import scipy.sparse as sp
from scipy.sparse.linalg import svds
#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
for i in [1,2,5,20,40,60,100,200]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    #calculate rmse score of matrix factorisation predictions
    rmse_score = rmse(X_pred,test_data_matrix)
    rmse_list.append(rmse_score)
    print("Matrix Factorisation with " + str(i) +" latent features has a RMSE of " + str(rmse_score))
#Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)
mf_pred.head()
df_names = pd.merge(df,movies,on='movieId')
df_names.head()
user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
print(user_index)
#choose a user ID
user_id = 999999
#get movies rated by this user id
users_movies = df_names.loc[df_names["userId"]==user_id]
#print how many ratings user has made 
print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
users_movies
#get user in
user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
#get movie ratings predicted for this user and sort by highest rating prediction
sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
#rename the columns
sorted_user_predictions.columns=['ratings']
#save the index values as movie id
sorted_user_predictions['movieId']=sorted_user_predictions.index
print("Top 10 predictions for User " + str(user_id))
#display the top 10 predictions for this user
pd.merge(sorted_user_predictions,movies, on = 'movieId')[:10]
#just taking 1% of users to test on
#count number of unique users
numUsers = df_train.userId.unique().shape[0]
#count number of unitque movies
numMovies = df_train.movieId.unique().shape[0]
print(len(df_train))
print(numUsers) 
print(numMovies)
#Separate out the values of the df_train data set into separate variables
Users = df_train['userId'].values
Movies = df_train['movieId'].values
Ratings = df_train['rating'].values
print(Users),print(len(Users))
print(Movies),print(len(Movies))
print(Ratings),print(len(Ratings))
#import libraries
import keras
from keras.layers import Embedding, Reshape, Merge
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#print version of keras
print(keras.__version__)
n_latent_factors=100
#movie input is an array
movie_input = keras.layers.Input(shape=[1],name='Item')
#create movie embeddings to transform movies to a number of latent factors
movie_embedding = keras.layers.Embedding(numMovies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
#dropout layer to prevent overfitting
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

#user input is an array
user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(numUsers + 1, n_latent_factors,name='User-Embedding')(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

#prod = keras.layers.merge([movie_vec, user_vec], mode='dot',name='DotProduct')
prod = keras.layers.dot([movie_vec, user_vec],axes=1,name='DotProduct')

model = keras.Model([user_input, movie_input], prod)


model.compile(optimizer='adam', loss='mse')
model.summary()
#print model architecture
#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
callbacks = [EarlyStopping('val_loss', patience=2),ModelCheckpoint('weights.h5', save_best_only=True)]
history_model_one = model.fit([Users, Movies], Ratings,validation_split=0.1, epochs=10, verbose=2,batch_size=64,callbacks=callbacks)
import matplotlib.pyplot as plt
pd.Series(history_model_one.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
import math
# Show the best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history_model_one.history['val_loss']))
print('Minimum RMSE at epoch ' + str(idx+1) + ' = ' + str(round(math.sqrt(min_val_loss),2)))
## Output
#os.listdir('../input')
# Load weights'../input/ratings.csv
model.load_weights('../input/final-project/weights.h5')
# Function to predict the ratings given User ID and Movie ID
def predict_rating(user_id, movie_id):
    return model.predict([np.array([user_id]), np.array([movie_id])])[0][0]
#predict the rating for user and movie
predict_rating(500,500)
#apply prediction across whole test set
df_nn_test = df_test.copy()
df_nn_test['prediction'] =df_nn_test.apply(lambda x: predict_rating(x['userId'], x['movieId']), axis=1)
#calculate rmse
from math import sqrt
df_nn_test['diff'] = df_nn_test['prediction'] - df_nn_test['rating']
df_nn_test['diff_squared']=df_nn_test['diff']**2
rmse_nn = sqrt(df_nn_test['diff_squared'].sum()/len(df_nn_test))
print(rmse_nn)
#choose a user ID
TEST_USER = 500
#get movies rated by this user id
users_movies = df_names.loc[df_names["userId"]==TEST_USER]
#print how many ratings user has made 
print("User ID : " + str(TEST_USER) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
users_movies.head(10)
#get movies that were rated by the user
user_ratings = df[df['userId'] == TEST_USER][['userId', 'movieId', 'rating']]
#get list of movie IDs not rated by the user
recommendations = df[df['movieId'].isin(user_ratings['movieId']) == False][['movieId']].drop_duplicates()
#get a list of recommendations
recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(500, x['movieId']), axis=1)
recommendations.sort_values(by='prediction',
                          ascending=False).merge(movies,on='movieId',).head(30)
