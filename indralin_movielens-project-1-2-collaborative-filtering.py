import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import re



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = [18, 8]
reviews = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/ratings.dat', names=['userId', 'movieId', 'rating', 'time'], delimiter='::', engine='python')

movies = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/movies.dat', names=['movieId', 'movie_names', 'genres'], delimiter='::', engine='python')

users = pd.read_csv('/kaggle/input/movielens-1m/ml-1m/users.dat', names=['userId', 'gender', 'age', 'occupation', 'zip'], delimiter='::', engine='python')



print('Reviews shape:', reviews.shape)

print('Users shape:', users.shape)

print('Movies shape:', movies.shape)
reviews.drop(['time'], axis=1, inplace=True)

users.drop(['zip'], axis=1, inplace=True)
movies['release_year'] = movies['movie_names'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
movies.head()
ages_map = {1: 'Under 18',

            18: '18 - 24',

            25: '25 - 34',

            35: '35 - 44',

            45: '45 - 49',

            50: '50 - 55',

            56: '56+'}



occupations_map = {0: 'Not specified',

                   1: 'Academic / Educator',

                   2: 'Artist',

                   3: 'Clerical / Admin',

                   4: 'College / Grad Student',

                   5: 'Customer Service',

                   6: 'Doctor / Health Care',

                   7: 'Executive / Managerial',

                   8: 'Farmer',

                   9: 'Homemaker',

                   10: 'K-12 student',

                   11: 'Lawyer',

                   12: 'Programmer',

                   13: 'Retired',

                   14: 'Sales / Marketing',

                   15: 'Scientist',

                   16: 'Self-Employed',

                   17: 'Technician / Engineer',

                   18: 'Tradesman / Craftsman',

                   19: 'Unemployed',

                   20: 'Writer'}



gender_map = {'M': 'Male', 'F': 'Female'}



users['age'] = users['age'].map(ages_map)

users['occupation'] = users['occupation'].map(occupations_map)

users['gender'] = users['gender'].map(gender_map)
final_df = reviews.merge(movies, on='movieId', how='left').merge(users, on='userId', how='left')



print('final_df shape:', final_df.shape)
final_df.head()
gender_counts = users['gender'].value_counts()



colors1 = ['dodgerblue', 'pink']



pie = go.Pie(labels=gender_counts.index,

             values=gender_counts.values,

             marker=dict(colors=colors1),

             hole=0.5)



layout = go.Layout(title='Male & Female users', font=dict(size=18), legend=dict(orientation='h'))



fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
age_reindex = ['Under 18', '18 - 24', '25 - 34', '35 - 44', '45 - 49', '50 - 55', '56+']



age_counts = users['age'].value_counts().reindex(age_reindex)



sns.barplot(x=age_counts.values,

            y=age_counts.index,

            palette='magma').set_title(

                'Users age', fontsize=24)



plt.show()
final_df[final_df['age'] == '25 - 34']['movie_names'].value_counts()[:7]
occupation_counts = users['occupation'].value_counts().sort_values(ascending=False)



sns.barplot(x=occupation_counts.values,

            y=occupation_counts.index,

            palette='dark').set_title(

                'Occupation list', fontsize=14)



plt.show()
n_users = final_df['userId'].nunique()

n_movies = final_df['movieId'].nunique()



print('Number of users:', n_users)

print('Number of movies:', n_movies)
final_df_matrix = final_df.pivot(index='userId',

                                 columns='movieId',

                                 values='rating').fillna(0)
final_df_matrix.head()
user_ratings_mean = np.mean(final_df_matrix.values, axis=1)

ratings_demeaned = final_df_matrix.values - user_ratings_mean.reshape(-1, 1)
# Check data sparsity



sparsity = round(1.0 - final_df.shape[0] / float(n_users * n_movies), 3)

print('The sparsity level of MovieLens1M dataset is ' +  str(sparsity * 100) + '%')
from scipy.sparse.linalg import svds



U, sigma, Vt = svds(ratings_demeaned, k=50)  # Number of singular values and vectors to compute
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = final_df_matrix.columns)



preds.head()
def recommend_movies(predictions, userID, movies, reviews, num_recommendations):

    

    # Get and sort the user's predictions

    user_row_number = userID - 1 # User ID starts at 1, not 0

    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = reviews[reviews.userId == (userID)]

    user_full = (user_data.merge(movies, how = 'left', on = 'movieId').

                     sort_values(['rating'], ascending=False)

                 )



    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))

    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].

         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

               left_on = 'movieId',

               right_on = 'movieId').

         rename(columns = {user_row_number: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return user_full.head(10), recommendations.sort_values('release_year', ascending=False)  # then sort by newest release year
user_already_rated, for_recommend = recommend_movies(preds, 1920, movies, reviews, 10)
user_already_rated
for_recommend
from surprise import Reader, Dataset, SVD, SVDpp

from surprise import accuracy
reader = Reader(rating_scale=(1, 5))



dataset = Dataset.load_from_df(final_df[['userId', 'movieId', 'rating']], reader=reader)



svd = SVD(n_factors=50)

svd_plusplus = SVDpp(n_factors=50)
trainset = dataset.build_full_trainset()



svd.fit(trainset)  # old version use svd.train
### It will take a LONG....TIME...., but it'll give a better score in RMSE & MAE



# svd_plusplus.fit(trainset)
id_2_names = dict()



for idx, names in zip(movies['movieId'], movies['movie_names']):

    id_2_names[idx] = names
def Build_Anti_Testset4User(user_id):

    

    fill = trainset.global_mean

    anti_testset = list()

    u = trainset.to_inner_uid(user_id)

    

    # ur == users ratings

    user_items = set([item_inner_id for (item_inner_id, rating) in trainset.ur[u]])

    

    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for

                            i in trainset.all_items() if i not in user_items]

    

    return anti_testset
def TopNRecs_SVD(user_id, num_recommender=10, latest=False):

    

    testSet = Build_Anti_Testset4User(user_id)

    predict = svd.test(testSet)  # we can change to SVD++ later

    

    recommendation = list()

    

    for userID, movieID, actualRating, estimatedRating, _ in predict:

        intMovieID = int(movieID)

        recommendation.append((intMovieID, estimatedRating))

        

    recommendation.sort(key=lambda x: x[1], reverse=True)

    

    movie_names = []

    movie_ratings = []

    

    for name, ratings in recommendation[:20]:

        movie_names.append(id_2_names[name])

        movie_ratings.append(ratings)

        

    movie_dataframe =  pd.DataFrame({'movie_names': movie_names,

                                     'rating': movie_ratings}).merge(movies[['movie_names', 'release_year']],

                                            on='movie_names', how='left')

    

    if latest == True:

        return movie_dataframe.sort_values('release_year', ascending=False)[['movie_names', 'rating']].head(num_recommender)

    

    else:

        return movie_dataframe.drop('release_year', axis=1).head(num_recommender)
TopNRecs_SVD(1920, num_recommender=10)
TopNRecs_SVD(1920, num_recommender=10, latest=True)
# Than predict ratings for all pairs (u, i) that are NOT in the training set.

testset = trainset.build_anti_testset()



predictions_svd = svd.test(testset)
print('SVD - RMSE:', accuracy.rmse(predictions_svd, verbose=False))

print('SVD - MAE:', accuracy.mae(predictions_svd, verbose=False))
from collections import defaultdict



def GetTopN(predictions, n=10, minimumRating=4.0):

        topN = defaultdict(list)



        for userID, movieID, actualRating, estimatedRating, _ in predictions:

            if (estimatedRating >= minimumRating):

                topN[int(userID)].append((int(movieID), estimatedRating))



        for userID, ratings in topN.items():

            ratings.sort(key=lambda x: x[1], reverse=True)

            topN[int(userID)] = ratings[:n]



        return topN
top_n = GetTopN(predictions_svd, n=10)



ii = 0

for uid, predict_ratings in top_n.items():

    print(uid, [iid for (iid, _) in predict_ratings])

    ii += 1

    

    if ii > 5:

        break