import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

from learntools.core import binder; binder.bind(globals())
from learntools.embeddings.ex3_gensim import *

input_dir = '../input/movielens-preprocessing'
model_dir = '../input/movielens-spiffy-model'
model_path = os.path.join(model_dir, 'movie_svd_model_32.h5')
model = keras.models.load_model(model_path)

emb_layer = model.get_layer('movie_embedding')
(w,) = emb_layer.get_weights()
movie_embedding_size = w.shape[1]

movies_path = os.path.join(input_dir, 'movie.csv')
all_movies_df = pd.read_csv(movies_path, index_col=0)

threshold = 100

movies = all_movies_df[all_movies_df.n_ratings >= threshold].reset_index(drop=True)

kv = WordEmbeddingsKeyedVectors(movie_embedding_size)
kv.add(
    movies['key'].values,
    w[movies.movieId]
)
# Example: one of my favourite films by Alfred Hitchcock. Try with some of your favourite movies.
kv.most_similar('Vertigo')
# Note: if you get a KeyError when looking up a movie, you may want to run something like this
# to look up the 'key' column for your movie. For example, there's more than one movie with the 
# title 'Spellbound', so I need to either call:
#     kv.most_similar('Spellbound (1945)')
# If I want the Hitchcock thriller, or:
#     kv.most_similar('Spellbound (2002)')
# If I want the documentary on spelling bees.
movies[movies.title.str.contains('Spellbound')]
# TODO: call most_similar with the movies "Legally Blonde" and "Mission: Impossible" as positive examples,
# and assign the results to the variable legally_impossible
legally_impossible = kv.most_similar(positive=['Legally Blonde', 'Mission: Impossible'])
part2.check()
legally_impossible
# Feel free to continue experimenting here.
part2.solution()
np.linalg.norm(w[0])
norms = None
part3.a.check()
#part3.a.hint()
#part3.a.solution()
norm_series = pd.Series(norms)
ax = norm_series.plot.hist()
ax.set_xlabel('Embedding norm');
# TODO: Your code goes here. Add the column "norm" to our movies dataframe.
part3.b.check()
#part3.b.solution()
n = 5
# Movies with the smallest embeddings (as measured by L2 norm)
all_movies_df.sort_values(by='norm').head(n)
# Movies with the largest embeddings
all_movies_df.sort_values(by='norm', ascending=False).head(n)
#part3.c.solution()