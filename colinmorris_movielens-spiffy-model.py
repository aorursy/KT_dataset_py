import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras

import os

import random



RUNNING_ON_KERNELS = 'KAGGLE_WORKING_DIR' in os.environ

input_dir = '../input' if RUNNING_ON_KERNELS else '../input/movielens_preprocessed'

ratings_path = os.path.join(input_dir, 'rating.csv')

df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating', 'y'])



tf.set_random_seed(1); np.random.seed(1); random.seed(1)
movie_embedding_size = user_embedding_size = 32

user_id_input = keras.Input(shape=(1,), name='user_id')

movie_id_input = keras.Input(shape=(1,), name='movie_id')

movie_r12n = keras.regularizers.l1_l2(l1=0, l2=1e-6)

user_r12n = keras.regularizers.l1_l2(l1=0, l2=1e-7)

dropout = .2

# Had good results with 'glorot_uniform' embeddings initializer, but this seems to cause some issues

# with model deserialization

user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size,

                                       embeddings_regularizer=user_r12n,

                                       input_length=1, name='user_embedding')(user_id_input)

user_embedded = keras.layers.Dropout(dropout)(user_embedded)

movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 

                                        embeddings_regularizer=movie_r12n,

                                        input_length=1, name='movie_embedding')(movie_id_input)

movie_embedded = keras.layers.Dropout(dropout)(movie_embedded)



dotted = keras.layers.Dot(2)([user_embedded, movie_embedded])

out = keras.layers.Flatten()(dotted)



biases = 0

if biases:

    bias_r12n = None

    bias_r12n = keras.regularizers.l1_l2(l1=1e-4, l2=1e-7) # XXX 1e-6 -> 1e-4

    bias_init = 'zeros'

    movie_b = keras.layers.Embedding(df.movieId.max()+1, 1, 

                                             name='movie_bias',

                                             embeddings_initializer=bias_init,

                                             embeddings_regularizer=bias_r12n,

                                            )(movie_id_input)

    movie_b = keras.layers.Flatten()(movie_b)



    user_b = keras.layers.Embedding(df.userId.max()+1, 1, 

                                             name='user_bias',

                                             embeddings_initializer=bias_init,

                                             embeddings_regularizer=bias_r12n,

                                            )(user_id_input)

    user_b = keras.layers.Flatten()(user_b)

    out = keras.layers.Add()([user_b, movie_b, out])



model = keras.Model(

    inputs = [user_id_input, movie_id_input],

    outputs = out,

)

model.compile(

    tf.train.AdamOptimizer(0.001),

    loss='MSE',

    metrics=['MAE'],

)



tf.set_random_seed(1); np.random.seed(1); random.seed(1)

history = model.fit(

    [df.userId, df.movieId],

    df.y,

    batch_size=10**4,

    epochs=30,

    verbose=2,

    validation_split=.05,

);
model.save('movie_svd_model_32.h5')