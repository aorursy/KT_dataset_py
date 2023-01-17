import os

import random



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras



from learntools.core import binder; binder.bind(globals())

from learntools.embeddings.ex2_factorization import *



input_dir = '../input/movielens-preprocessing'

df = pd.read_csv(os.path.join(input_dir, 'rating.csv'), usecols=['userId', 'movieId', 'rating', 'y'])

movies = pd.read_csv(os.path.join(input_dir, 'movie.csv'), index_col=0)



model_dir = '../input/matrix-factorization'

model_fname = 'factorization_model.h5'

model_path = os.path.join(model_dir, model_fname)

model = keras.models.load_model(model_path)



print("Setup complete!")
# Id of the user for whom we're predicting ratings

uid = 26556

candidate_movies = movies[

    movies.title.str.contains('Naked Gun')

    | (movies.title == 'The Sisterhood of the Traveling Pants')

    | (movies.title == 'Lilo & Stitch')

].copy()



preds = model.predict([

    [uid] * len(candidate_movies), # User ids 

    candidate_movies.index, # Movie ids

])

# Because our model was trained on a 'centered' version of rating (subtracting the mean, so that

# the target variable had mean 0), to get the predicted star rating on the original scale, we need

# to add the mean back in.

row0 = df.iloc[0]

offset = row0.rating - row0.y

candidate_movies['predicted_rating'] = preds + offset

candidate_movies.head()[ ['movieId', 'title', 'predicted_rating'] ]
def recommend(model, user_id, n=5):

    """Return a DataFrame with the n most highly recommended movies for the user with the

    given id. (Where most highly recommended means having the highest predicted ratings 

    according to the given model).

    The returned DataFrame should have a column for movieId and predicted_rating (it may also have

    other columns).

    """

    all_movie_ids = movies.index 

    preds = model.predict([

        np.repeat(uid, len(all_movie_ids)),

        all_movie_ids,

    ])

    # Add back the offset calculated earlier, to 'uncenter' the ratings, and get back to a [0.5, 5] scale.

    movies.loc[all_movie_ids, 'predicted_rating'] = preds + offset

    reccs = movies.sort_values(by='predicted_rating', ascending=False).head(n)

    return reccs
#part1.hint()
#part1.solution()
recommend(model, 26556)
uid = 26556

user_ratings = df[df.userId==uid]

movie_cols = ['movieId', 'title', 'genres', 'year', 'n_ratings', 'mean_rating']

user_ratings.sort_values(by='rating', ascending=False).merge(movies[movie_cols], on='movieId')
part2.solution()
part3.solution()
def recommend_nonobscure(model, user_id, n=5, min_ratings=1000):

    # Add predicted_rating column if we haven't already done so.

    if 'predicted_rating' not in movies.columns:

        all_movie_ids = df.movieId.unique()

        preds = model.predict([

            np.repeat(uid, len(all_movie_ids)),

            all_movie_ids,

        ])

        # Add back the offset calculated earlier, to 'uncenter' the ratings, and get back to a [0.5, 5] scale.

        movies.loc[all_movie_ids, 'predicted_rating'] = preds + offset



    nonobscure_movie_ids = movies.index[movies.n_ratings >= min_ratings]

    return movies.loc[nonobscure_movie_ids].sort_values(by='predicted_rating', ascending=False).head(n)
#part4.hint()
#part4.solution()
recommend_nonobscure(model, uid)
part5.solution()
movie_embedding_size = user_embedding_size = 8

user_id_input = keras.Input(shape=(1,), name='user_id')

movie_id_input = keras.Input(shape=(1,), name='movie_id')



movie_r12n = keras.regularizers.l2(1e-6)

user_r12n = keras.regularizers.l2(1e-7)

user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size,

                                       embeddings_initializer='glorot_uniform',

                                       embeddings_regularizer=user_r12n,

                                       input_length=1, name='user_embedding')(user_id_input)

movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 

                                        embeddings_initializer='glorot_uniform',

                                        embeddings_regularizer=movie_r12n,

                                        input_length=1, name='movie_embedding')(movie_id_input)



dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])

out = keras.layers.Flatten()(dotted)



l2_model = keras.Model(

    inputs = [user_id_input, movie_id_input],

    outputs = out,

)
model_dir = '../input/regularized-movielens-factorization-model'

model_fname = 'movie_svd_model_8_r12n.h5'

model_path = os.path.join(model_dir, model_fname)

l2_model = keras.models.load_model(model_path)
# Use the recommend() function you wrote earlier to get the 5 best recommended movies

# for user 26556, and assign them to the variable l2_reccs.

l2_reccs = []

l2_reccs = recommend(l2_model, 26556)
#part6.solution()
uid = 26556

obscure_reccs = recommend(model, uid)

obscure_mids = obscure_reccs.index

preds = l2_model.predict([

    np.repeat(uid, len(obscure_mids)),

    obscure_mids,

])

recc_df = movies.loc[obscure_mids].copy()

recc_df['l2_predicted_rating'] = preds + offset

recc_df