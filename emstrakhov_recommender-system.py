# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# movies = pd.read_csv("../input/movielens/movies.csv")

# ratings = pd.read_csv("../input/movielens/ratings.csv")
movies = pd.read_csv("../input/moviesbyresidents/Movies  Ratings Dataset - Movies.csv")

ratings = pd.read_csv("../input/movies-residents-part-2/Movies  Ratings Dataset - Ratings (1).csv")
ratings.drop('timestamp', axis=1, inplace=True)
movies.head()
ratings.head()
ratings['userId'].nunique(), ratings['movieId'].nunique()
ratings.tail(20)
ratings.columns
mini_df = pd.DataFrame([[248, 19, 1],

                       [248, 141, 1],

                       [248, 223, 1]], columns=ratings.columns)



ratings = ratings.append(mini_df)
ratings.tail(10)
dict_userID = {}

i = 0

for new_id, old_id in enumerate(set(ratings['userId'])):

    dict_userID[old_id] = new_id
ratings['new_userId'] = ratings['userId'].map(dict_userID)

ratings.tail(10)
ratings[ratings['userId']==666]
R_df = ratings.pivot(index='new_userId', columns='movieId', values='rating').fillna(0)

R_df.head()
R = R_df.values

user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df.head(10)
preds_df.shape
def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):

    

    # Get and sort the user's predictions

    user_row_number = userID # UserID starts at 1, not 0

    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = original_ratings_df[original_ratings_df['new_userId'] == (userID)]

    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').

                     sort_values(['rating'], ascending=False)

                 )



    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))

    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].

         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

               left_on = 'movieId',

               right_on = 'movieId').

         rename(columns = {user_row_number: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return user_full, recommendations
already_rated, predictions = recommend_movies(preds_df, dict_userID[248], movies, ratings, 10)
already_rated.head(10)
predictions