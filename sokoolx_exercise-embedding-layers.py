# Setup code. Make sure you run this first!

import os
import random
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

from learntools.core import binder; binder.bind(globals())
from learntools.embeddings.ex1_embedding_layers import *

input_dir = '../input'

# Load a 10% subset of the full MovieLens data.
df = pd.read_csv(os.path.join(input_dir, 'mini_rating.csv'))

# Some hyperparameters. (You might want to play with these later)
LR = .005 # Learning rate
EPOCHS = 8 # Default number of training epochs (i.e. cycles through the training data)
hidden_units = (32,4) # Size of our hidden layers

def build_and_train_model(movie_embedding_size=8, user_embedding_size=8, verbose=2, epochs=EPOCHS):
    tf.set_random_seed(1); np.random.seed(1); random.seed(1) # Set seeds for reproducibility

    user_id_input = keras.Input(shape=(1,), name='user_id')
    movie_id_input = keras.Input(shape=(1,), name='movie_id')
    user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size, 
                                           input_length=1, name='user_embedding')(user_id_input)
    movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 
                                            input_length=1, name='movie_embedding')(movie_id_input)
    concatenated = keras.layers.Concatenate()([user_embedded, movie_embedded])
    out = keras.layers.Flatten()(concatenated)

    # Add one or more hidden layers
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='relu')(out)

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)

    model = keras.Model(
        inputs = [user_id_input, movie_id_input],
        outputs = out,
    )
    model.compile(
        tf.train.AdamOptimizer(LR),
        loss='MSE',
        metrics=['MAE'],
    )
    history = model.fit(
        [df.userId, df.movieId],
        df.y,
        batch_size=5 * 10**3,
        epochs=epochs,
        verbose=verbose,
        validation_split=.05,
    )
    return history

# Train two models with different embedding sizes and save the training statistics.
# We'll be using this later in the exercise.
history_8 = build_and_train_model(verbose=0)
history_32 = build_and_train_model(32, 32, verbose=0)

print("Setup complete!")
# embedding_variables should contain all the variables you would use an embedding layer for
# For your convenience, we've initialized it with all variables in the dataset, so you can 
# just delete or comment out the variables you want to exclude.
embedding_variables = {
    #'stream_id',
    'user_id',
    'song_id',
    #'timestamp',
    'artist_id',
    #'song_duration',
    #'explicit',
    #'user_country',
}
part1.check()
part1.solution()
history_FS = (15, 5)
def plot_history(histories, keys=('mean_absolute_error',), train=True, figsize=history_FS):
    if isinstance(histories, tf.keras.callbacks.History):
        histories = [ ('', histories) ]
    for key in keys:
        plt.figure(figsize=history_FS)
        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_'+key],
                           '--', label=str(name).title()+' Val')
            if train:
                plt.plot(history.epoch, history.history[key], color=val[0].get_color(), alpha=.5,
                         label=str(name).title()+' Train')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_',' ').title())
        plt.legend()
        plt.title(key)

        plt.xlim([0,max(max(history.epoch) for (_, history) in histories)])

plot_history([ 
    ('base model', history_8),
])
plot_history([ 
    ('8-d embeddings', history_8),
    ('32-d embeddings', history_32),
])
# Example: shrinking movie embeddings and growing user embeddings
history_biguser_smallmovie = build_and_train_model(movie_embedding_size=4, user_embedding_size=16)
part2.solution()
part3.a.solution()
part3.b.solution()
user_embedding_size = movie_embedding_size = 8
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')
user_embedded = keras.layers.Embedding(df.userId.max()+1, user_embedding_size, 
                                       input_length=1, name='user_embedding')(user_id_input)
movie_embedded = keras.layers.Embedding(df.movieId.max()+1, movie_embedding_size, 
                                        input_length=1, name='movie_embedding')(movie_id_input)
concatenated = keras.layers.Concatenate()([user_embedded, movie_embedded])
out = keras.layers.Flatten()(concatenated)

# Add one or more hidden layers
for n_hidden in hidden_units:
    out = keras.layers.Dense(n_hidden, activation='relu')(out)

# A single output: our predicted rating (before adding bias)
out = keras.layers.Dense(1, activation='linear', name='prediction')(out)

################################################################################
############################# YOUR CODE GOES HERE! #############################
# TODO: you need to create the variable movie_bias. Its value should be the output of calling a layer.
# I recommend giving the layer that holds your biases a distinctive name (this will help in an upcoming question)
bias_embedded = keras.layers.Embedding(df.movieId.max()+1, 1, input_length=1, name='bias',)(movie_id_input)
movie_bias = keras.layers.Flatten()(bias_embedded)
################################################################################
out = keras.layers.Add()([out, movie_bias])

model_bias = keras.Model(
    inputs = [user_id_input, movie_id_input],
    outputs = out,
)
model_bias.compile(
    tf.train.AdamOptimizer(LR),
    loss='MSE',
    metrics=['MAE'],
)
model_bias.summary()
part3.c.hint()
part3.c.solution()
history_bias = model_bias.fit(
    [df.userId, df.movieId],
    df.y,
    batch_size=5 * 10**3,
    epochs=EPOCHS,
    verbose=2,
    validation_split=.05,
);
plot_history([ 
    ('no_bias', history_8),
    ('bias', history_bias),
]);
bias_layer = model_bias.get_layer('bias')

part3.d.check()

(b,) = bias_layer.get_weights()
print("Loaded biases with shape {}".format(b.shape))
part3.d.solution()
movies = pd.read_csv(os.path.join(input_dir, 'movie.csv'), index_col=0, 
                     usecols=['movieId', 'title', 'genres', 'year'])
ntrain = math.floor(len(df) * .95)
df_train = df.head(ntrain)

# Mapping from original movie ids to canonical ones
mids = movieId_to_canon = df.groupby('movieId')['movieId_orig'].first()
# Add bias column
movies.loc[mids.values, 'bias'] = b
# Add columns for number of ratings and average rating
g = df_train.groupby('movieId_orig')
movies.loc[mids.values, 'n_ratings'] = g.size()
movies.loc[mids.values, 'mean_rating'] = g['rating'].mean()

movies.dropna(inplace=True)

movies.head()
from IPython.display import display
n = 10
display(
    "Largest biases...",
    movies.sort_values(by='bias', ascending=False).head(n),
    "Smallest biases...",
    movies.sort_values(by='bias').head(n),
)
n = 1000
mini = movies.sample(n, random_state=1)

fig, ax = plt.subplots(figsize=(13, 7))
ax.scatter(mini['mean_rating'], mini['bias'], alpha=.4)
ax.set_xlabel('Mean rating')
ax.set_ylabel('Bias');