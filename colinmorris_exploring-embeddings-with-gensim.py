import os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras



input_dir = '../input/movielens-preprocessing'

model_dir = '../input/movielens-spiffy-model'

model_path = os.path.join(model_dir, 'movie_svd_model_32.h5')

model = keras.models.load_model(model_path)
emb_layer = model.get_layer('movie_embedding')

(w,) = emb_layer.get_weights()

w.shape
w[0]
movies_path = os.path.join(input_dir, 'movie.csv')

movies_df = pd.read_csv(movies_path, index_col=0)

movies_df.head()
i_toy_story = 0

i_shrek = movies_df.loc[

    movies_df.title == 'Shrek',

    'movieId'

].iloc[0]



toy_story_vec = w[i_toy_story]

shrek_vec = w[i_shrek]



print(

    toy_story_vec,

    shrek_vec,

    sep='\n',

)
from scipy.spatial import distance



distance.euclidean(toy_story_vec, shrek_vec)
i_exorcist = movies_df.loc[

    movies_df.title == 'The Exorcist',

    'movieId'

].iloc[0]



exorcist_vec = w[i_exorcist]



distance.euclidean(toy_story_vec, exorcist_vec)
print(

    distance.cosine(toy_story_vec, shrek_vec),

    distance.cosine(toy_story_vec, exorcist_vec),

    sep='\n'

)
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors



# Limit to movies with at least this many ratings in the dataset

threshold = 100

mainstream_movies = movies_df[movies_df.n_ratings >= threshold].reset_index(drop=True)



movie_embedding_size = w.shape[1]

kv = WordEmbeddingsKeyedVectors(movie_embedding_size)

kv.add(

    mainstream_movies['key'].values,

    w[mainstream_movies.movieId]

)
kv.most_similar('Toy Story')


import textwrap

movies = ['Eyes Wide Shut', 'American Pie', 'Iron Man 3', 'West Side Story',

          'Battleship Potemkin', 'Clueless'

]



def plot_most_similar(movie, ax, topn=5):

    sim = kv.most_similar(movie, topn=topn)[::-1]

    y = np.arange(len(sim))

    w = [t[1] for t in sim]

    ax.barh(y, w)

    left = min(.6, min(w))

    ax.set_xlim(right=1.0, left=left)

    # Split long titles over multiple lines

    labels = [textwrap.fill(t[0] , width=24)

              for t in sim]

    ax.set_yticks(y)

    ax.set_yticklabels(labels)

    ax.set_title(movie)    



fig, axes = plt.subplots(3, 2, figsize=(15, 9))



for movie, ax in zip(movies, axes.flatten()):

    plot_most_similar(movie, ax)

    

fig.tight_layout()
kv.most_similar(

    positive = ['Scream'],

    negative = ['Psycho (1960)']

)
kv.most_similar(

    ['Pocahontas', 'Cars 2'],

    negative = ['Brave']

)