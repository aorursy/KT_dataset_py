import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# 1. User's Dataset

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('../input/ml-100k/u.user', sep='|', names=u_cols,

                    encoding='latin-1', parse_dates=True) 

# 2. Rating dataset

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols,

                      encoding='latin-1')



# 3.Movies Dataset

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url','unknown', 'Action', 'Adventure',

'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('../input/ml-100k/u.item', sep='|', names=m_cols,

                     encoding='latin-1')
#users

print(users.shape)

users.head(4)
#ratings

print(ratings.shape)

ratings.head(4)
#items

print(movies.shape)

movies.head(4)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_train = pd.read_csv('../input/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

ratings_test = pd.read_csv('../input/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

ratings_train.shape, ratings_test.shape
ratings_train.head(4),ratings_test.head(4)
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x=ratings_train.rating,data=ratings_train)
ratings_train.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
no_of_rated_movies_per_user = ratings_train.groupby(by='user_id')['rating'].count().sort_values(ascending=False)

no_of_rated_movies_per_user.head()
ax1 = plt.subplot(121)

sns.kdeplot(no_of_rated_movies_per_user, shade=True, ax=ax1)

plt.xlabel('No of ratings by user')

plt.title("PDF")



ax2 = plt.subplot(122)

sns.kdeplot(no_of_rated_movies_per_user, shade=True, cumulative=True,ax=ax2)

plt.xlabel('No of ratings by user')

plt.title('CDF')



plt.show()
no_of_rated_movies_per_user.describe()
quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')

quantiles
plt.title("Quantiles and their Values")

quantiles.plot()

# quantiles with 0.05 difference

plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")

# quantiles with 0.25 difference

plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")

plt.ylabel('No of ratings by user')

plt.xlabel('Value at the quantile')

plt.legend(loc='best')



# annotate the 25th, 50th, 75th and 100th percentile values....

for x,y in zip(quantiles.index[::25], quantiles[::25]):

    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500)

                ,fontweight='bold')



plt.show()
quantiles[::5]
print('\n No of ratings at last 5 percentile : {}\n'.format(sum(no_of_rated_movies_per_user>= 301)) )
no_of_ratings_per_movie = ratings_train.groupby(by='movie_id')['rating'].count().sort_values(ascending=False)



fig = plt.figure(figsize=plt.figaspect(.5))

ax = plt.gca()

plt.plot(no_of_ratings_per_movie.values)

plt.title('# RATINGS per Movie')

plt.xlabel('Movie')

plt.ylabel('No of Users who rated a movie')

ax.set_xticklabels([])



plt.show()
n_users = ratings.user_id.unique().shape[0]

n_items = ratings.movie_id.unique().shape[0]
data_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():

    data_matrix[line[1]-1, line[2]-1] = line[3]

    

test_data_matrix = np.zeros((n_users, n_items))

for line in ratings_test.itertuples():

    test_data_matrix[line[1]-1, line[2]-1] = line[3]
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.metrics.pairwise import cosine_similarity 



#user_similarity = pairwise_distances(data_matrix, metric='cosine')

#item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

#print (user_similarity[0][1])



# NOTE: why use pairwise_distances? why not cosine_similarity? cosine_distance = 1-cosine_similarity. i believe cosine_similarity is right for here.

# let's change it to consine_similarity



user_similarity = cosine_similarity(data_matrix)

item_similarity = cosine_similarity(data_matrix.T)

print (user_similarity[0][1])

user_similarity.shape
def predict(ratings, similarity, type='user'):

    if type == 'user':

        mean_user_rating = ratings.mean(axis=1)

        print (mean_user_rating.shape)

        print (mean_user_rating[:, np.newaxis].shape)

        #We use np.newaxis so that mean_user_rating has same format as ratings

        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])



        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

        print (np.array([np.abs(similarity).sum(axis=1)]).shape, pred.shape)

    elif type == 'item':

        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred
user_prediction = predict(data_matrix, user_similarity, type='user')

item_prediction = predict(data_matrix, item_similarity, type='item')

print (data_matrix[0])

print (user_prediction[0])
from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse(prediction, ground_truth):

    prediction = prediction[ground_truth.nonzero()].flatten() 

    ground_truth = ground_truth[ground_truth.nonzero()].flatten()

    #print (prediction.shape, ground_truth.shape)

    return sqrt(mean_squared_error(prediction, ground_truth))
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))

print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
import turicreate



train_data = turicreate.SFrame(ratings_train)

test_data = turicreate.SFrame(ratings_test)
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
#Get recommendations for first 5 users and print them

#users = range(1,6) specifies user ID of first 5 users

#k=5 specifies top 5 recommendations to be given

popularity_recomm = popularity_model.recommend(users=list(range(1,6)),k=5)

popularity_recomm.print_rows(num_rows=25)

#Train Model

#item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')



#Make Recommendations:

item_sim_recomm = item_sim_model.recommend(users=list(range(1,6)),k=5)

item_sim_recomm.print_rows(num_rows=25)
popularity_model.evaluate(test_data)

item_sim_model.evaluate(test_data)
#model_performance = turicreate.compare(test_data, [popularity_model, item_sim_model])

#turicreate.show_comparison(model_performance,[popularity_model, item_sim_model])

model_performance = turicreate.recommender.util.compare_models = (test_data, [popularity_model, item_sim_model])

print (model_performance)

print (len(ratings_train), float(n_users*n_items))

sparsity=round(1.0-len(ratings_train)/float(n_users*n_items),3)

print ('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')
import scipy.sparse as sp

from scipy.sparse.linalg import svds



def SVD(rating_matrix):

    #get SVD components from train matrix. Choose k.

    u, s, vt = svds(rating_matrix, k = 20)

    s_diag_matrix=np.diag(s)

    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

    return X_pred



print ('User-based CF MSE: ' + str(rmse(SVD(data_matrix), test_data_matrix)))
factorization_model = turicreate.factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

factorization_recomm = factorization_model.recommend(users=list(range(1,6)),k=5)

factorization_recomm.print_rows(num_rows=25)
factorization_model.evaluate(test_data)