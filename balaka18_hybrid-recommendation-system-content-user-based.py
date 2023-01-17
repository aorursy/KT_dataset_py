import pandas as pd
import numpy as np
import json,ast
from scipy.sparse import csr_matrix as csr
# SKLEARN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
movies = pd.read_csv('../input/movies/movies_data.csv')
print(movies.head(10))
ratings = pd.read_csv('../input/movies/ratings.csv')
print(ratings.head(10))
print(movies.shape,"\n")
movies.info()
# movies['movieId'] = [int(eval(s)[0]['id']) if len(eval(s))>0 else 0 for s in movies['genres'].values]
# Extracting the genres
def extractGenre(s):
  lst = eval(s)
  if len(lst) > 0:
    return lst[0]['name']
  else:
    return ''
# Removing NaN
def delNan(cell):
  if type(cell) == float:
    return ''
  else:
    return cell
# Test run
'''print(extractGenre([{'id': 14, 'name': 'Fantasy'}, {'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]))'''

movies['genres'] = movies['genres'].apply(extractGenre)
movies['title'] = movies['title'].apply(delNan)
movies['overview'] = movies['overview'].apply(delNan)
movies
# Movies without titles have no relevance in recommendation systems. So, since the number of movies with missing title are only 6 out of the 45466 movies, we remove them
movies.drop(index=[45460,45461,45462,45463,45464,45465],inplace=True)
movies['id'] = movies['id'].astype(int)
# Merging genres and overview together so that our recommender system is based on both the overview and genre
movies['genre+overview'] = [",".join([movies.iloc[i,0].lower(),movies.iloc[i,2].lower()]) for i in range(movies.shape[0])]
# Counting the number of movies under each genre
print(movies['genres'].value_counts(),"\n")
mov_enc = movies.copy()
# Label Encoding the genres
le = LabelEncoder()
mov_enc['genres'] = le.fit_transform(mov_enc['genres'])
print("ORIGINAL :\n",movies,"\n\nENCODED GENRES :\n",mov_enc)
# Training the tf-idf model on the genre+overview column for a subset of the data
movies_new = movies.iloc[:35001,:]
tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words='english',lowercase=True,encoding='utf-8')
tfidf_matrix = tfidf.fit_transform(movies_new['genre+overview'])
sim_scores = linear_kernel(tfidf_matrix,tfidf_matrix)
# Function for recommending movies
indices = pd.Series(movies_new.index,index=movies_new['id']).drop_duplicates()
def rec_movie(movieId,similarity=sim_scores):
  # User input is mapped to the corresponding movie id
  index = indices[movieId]
  # Pairwise score evaluation
  similar = list(enumerate(similarity[index]))
  # Sort in decreasing order
  similar = sorted(similar, key = lambda x: x[1], reverse=True)
  # Fetch the top 10 movies
  mov10_ind = [i[0] for i in similar[1:11]]
  recommended = pd.DataFrame()
  recommended['movieId'],recommended['title'],recommended['genre+overview'] = movies_new['id'].iloc[mov10_ind],movies_new['title'].iloc[mov10_ind],movies_new['genre+overview'].iloc[mov10_ind]
  return recommended

# Test for Despicable Me
print("ORIGINAL MOVIE : ", movies_new['title'].iloc[indices[20352]],"\nGENRE + OVERVIEW : ", movies_new['genre+overview'].iloc[indices[20352]])
print("\n\n10 RECOMMENDED MOVIES :\n")
rec_movie(20352)
print(ratings.shape,"\n")
ratings.info()
# Using a subset of the data due to RAM limitations
ratings_new = ratings.iloc[:20001,:]
# Splitting to train and test data
train,test = ratings_new.iloc[:(int(0.8*ratings_new.shape[0])+1),:],ratings_new.iloc[(int(0.8*ratings_new.shape[0])+1):,:]
print("TRAIN :\n",train,"\n\nTEST :\n",test)
userID,movieId = 2,222
test1 = rec_movie(movieId)
test1
final1 = ratings.merge(test1,on='movieId')
final1
pred = final1['rating'].median()
pred
test2 = ratings.merge(movies_new,left_on='movieId',right_on='id')
test2
temp = train.merge(movies_new,left_on='movieId',right_on='id')
temp
pvt = ratings_new.pivot_table(index='userId',columns='movieId',values='rating')
pvt.fillna(0,inplace=True)
pvt
cosine_sim = 1-pairwise_distances(pvt, metric="cosine")
cosine_sim
pearson_sim = 1-pairwise_distances(pvt, metric="correlation")
pearson_sim
# Get 10 similar users by nearest neighbors algorithm, defaulting to Pearson correlation coefficient metric
def sim10_users(user_id,pvt,metric="correlation",k=10):
  indices_sim = []
  knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
  knn.fit(pvt)
  distances, indices_sim = knn.kneighbors(pvt.iloc[user_id - 1, :].values.reshape(1, -1), n_neighbors = k+1)
  sims = 1 - distances.flatten()
  return sims,indices_sim

# Predict ratings
'''ALGORITHM :
1. Get 10 simialr users
2. Get the mean of all user ratings for that userId
3. Find sum of all ratings of the similar users obtained.
4. for all similar users:
    --> Find rating_diff = (Rating by user j on movie i) - (mean of all user ratings by that user)
    --> Get updated_rating = updated_rating + (rating_diff * similarity_score(of user j))
5. Predicted rating = (((step 2.)*updated_rating + step 3.) + 1)
'''
def predict_rating(user_id, movie_id, pvt, metric = "correlation", k = 10):
    pred = 0
    indices_mov = list(pvt.columns)
    indexm = indices_mov.index(movie_id)
    # STEP 1
    sims, indices = sim10_users(user_id, pvt, metric, k)
    # STEP 2
    mean_rating = pvt.loc[user_id,:].mean()     # Adjusting for zero based indexing
    # STEP 3
    rtSum = np.sum(sims) - 1
    pdt,updated_rating = 1,0           # Initializing product and updated rating
    # STEP 4
    for i in range(0, len(indices.flatten())):
        if (indices.flatten()[i] + 1) == user_id:
            continue
        else: 
            rating_diff = abs(pvt.iloc[indices.flatten()[i],indexm]-np.mean(pvt.iloc[indices.flatten()[i],:]))
            pdt = rating_diff * (sims[i])
            updated_rating = updated_rating + pdt
    
    pred = pred + ((mean_rating*updated_rating + rtSum) + 1)
    return pred
# Test
predict_rating(77, 4499, pvt)
def predictRating_combined(userId,movieId,pvt=pvt,metric="correlation",k=10):
  testdf1 = rec_movie(movieId)
  finaldf1 = ratings.merge(testdf1,on='movieId')
  if len(list(finaldf1.index)) > 0 :
    pred = final1['rating'].median()
    return pred,testdf1
  else:
    pred2 = predict_rating(userId,movieId,pvt=pvt,metric="correlation",k=10)
    return pred2,testdf1

# Final function for predicting user rating and recommended movies
def recommend(userId,movieId,movies_new = movies_new):
  indices = pd.Series(movies_new.index,index=movies_new['id']).drop_duplicates()
  predicted_rating,rec = predictRating_combined(userId,movieId)
  print('\nRECOMMENDED MOVIES for the movie {} : \n\n{}\n\n=== Predicted rating for user {} -> movie {}: {:.1f} ==='.format(movies_new['title'].iloc[indices[movieId]],rec,userId,movieId,predicted_rating))

# Test run 1
recommend(2,222)
print('-'*90)
# Test run 2
recommend(77,4499)
