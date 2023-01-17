# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Функция для составления рекомендаций по восстановленной матрице рейтингов

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):

    

    # Get and sort the user's predictions

    sorted_user_predictions = predictions_df.loc[userID].sort_values(ascending=False)

    

    # Get the user's data and merge in the movie information.

    user_data = original_ratings_df[original_ratings_df['userId'] == userID]

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

        rename(columns={userID: 'Predictions'}).

         sort_values('Predictions', ascending = False).

                       iloc[:num_recommendations, :-1]

                      )



    return user_full, recommendations
movies = pd.read_csv("../input/movielens/movies.csv")

ratings = pd.read_csv("../input/movielens/ratings.csv")
movies.head()
movies.shape[0]
ratings.head()
# Количество пользователей

ratings['userId'].nunique()
# Количество фильмов

ratings['movieId'].nunique()
# Построение матрицы "пользователи-фильмы"

R_df = ratings.pivot_table(index='userId', columns='movieId', 

                           values='rating').fillna(0)

R_df.head()
R_df.shape
R = R_df.values

# "Убираем эмоции" из рейтингов

user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
# Применяем SVD-разложение

from scipy.sparse.linalg import svds

U, sigma, Vt = svds(R_demeaned, k = 500)
U.shape, sigma.shape, Vt.shape
# Приближенная матрица "пользователи-фильмы"

R_hat = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean.reshape(-1, 1)
R_hat_df = pd.DataFrame(R_hat, columns=R_df.columns, index=R_df.index)

R_hat_df.head()
already_rated, predictions = recommend_movies(R_hat_df, 

                                              100, # кому рекомендуем

                                              movies, ratings, 

                                              10) # сколько рекомендуем
predictions
sorted_user_predictions = R_hat_df.loc[100].sort_values(ascending=False)

    

# Get the user's data and merge in the movie information.

user_data = ratings[ratings['userId'] == 100]

user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').

                     sort_values(['rating'], ascending=False)

                 )



user_full.head()
user_full.shape
pd.DataFrame(sorted_user_predictions).reset_index()