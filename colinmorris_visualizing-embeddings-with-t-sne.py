

%matplotlib inline

import random

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



movies_path = os.path.join(input_dir, 'movie.csv')

movies_df = pd.read_csv(movies_path, index_col=0)
threshold = 100

mainstream_movies = movies_df[movies_df.n_ratings >= threshold].reset_index(drop=True)

print("Went from {} to {} movies after applying threshold".format(

    len(movies_df), len(mainstream_movies),

))

w_full = w

w = w[mainstream_movies.movieId]

df = mainstream_movies
from sklearn.manifold import TSNE



# The default of 1,000 iterations gives fine results, but I'm training for longer just to eke

# out some marginal improvements. NB: This takes almost an hour!

tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")



embs = tsne.fit_transform(w)

# Add to dataframe for convenience

df['x'] = embs[:, 0]

df['y'] = embs[:, 1]


# Save a copy of our t-SNE mapping data for later use (we'll be loading this file in the exercise)

df.to_csv('movies_tsne.csv')
embs[:5]
FS = (10, 8)

fig, ax = plt.subplots(figsize=FS)

# Make points translucent so we can visually identify regions with a high density of overlapping points

ax.scatter(df.x, df.y, alpha=.1);


# Some helper functions for plotting annotated t-SNE visualizations



# TODO: adjust_text not available in kernels

try:

    from adjustText import adjust_text

except ImportError:

    def adjust_text(*args, **kwargs):

        pass



def adjust_text(*args, **kwargs):

    pass



def plot_bg(bg_alpha=.01, figsize=(13, 9), emb_2d=None):

    """Create and return a plot of all our movie embeddings with very low opacity.

    (Intended to be used as a basis for further - more prominent - plotting of a 

    subset of movies. Having the overall shape of the map space in the background is

    useful for context.)

    """

    if emb_2d is None:

        emb_2d = embs

    fig, ax = plt.subplots(figsize=figsize)

    X = emb_2d[:, 0]

    Y = emb_2d[:, 1]

    ax.scatter(X, Y, alpha=bg_alpha)

    return ax



def annotate_sample(n, n_ratings_thresh=0):

    """Plot our embeddings with a random sample of n movies annotated.

    Only selects movies where the number of ratings is at least n_ratings_thresh.

    """

    sample = mainstream_movies[mainstream_movies.n_ratings >= n_ratings_thresh].sample(

        n, random_state=1)

    plot_with_annotations(sample.index)



def plot_by_title_pattern(pattern, **kwargs):

    """Plot all movies whose titles match the given regex pattern.

    """

    match = df[df.title.str.contains(pattern)]

    return plot_with_annotations(match.index, **kwargs)



def add_annotations(ax, label_indices, emb_2d=None, **kwargs):

    if emb_2d is None:

        emb_2d = embs

    X = emb_2d[label_indices, 0]

    Y = emb_2d[label_indices, 1]

    ax.scatter(X, Y, **kwargs)



def plot_with_annotations(label_indices, text=True, labels=None, alpha=1, **kwargs):

    ax = plot_bg(**kwargs)

    Xlabeled = embs[label_indices, 0]

    Ylabeled = embs[label_indices, 1]

    if labels is not None:

        for x, y, label in zip(Xlabeled, Ylabeled, labels):

            ax.scatter(x, y, alpha=alpha, label=label, marker='1',

                       s=90,

                      )

        fig.legend()

    else:

        ax.scatter(Xlabeled, Ylabeled, alpha=alpha, color='green')

    

    if text:

        # TODO: Add abbreviated title column

        titles = mainstream_movies.loc[label_indices, 'title'].values

        texts = []

        for label, x, y in zip(titles, Xlabeled, Ylabeled):

            t = ax.annotate(label, xy=(x, y))

            texts.append(t)

        adjust_text(texts, 

                    #expand_text=(1.01, 1.05),

                    arrowprops=dict(arrowstyle='->', color='red'),

                   )

    return ax



FS = (13, 9)

def plot_region(x0, x1, y0, y1, text=True):

    """Plot the region of the mapping space bounded by the given x and y limits.

    """

    fig, ax = plt.subplots(figsize=FS)

    pts = df[

        (df.x >= x0) & (df.x <= x1)

        & (df.y >= y0) & (df.y <= y1)

    ]

    ax.scatter(pts.x, pts.y, alpha=.6)

    ax.set_xlim(x0, x1)

    ax.set_ylim(y0, y1)

    if text:

        texts = []

        for label, x, y in zip(pts.title.values, pts.x.values, pts.y.values):

            t = ax.annotate(label, xy=(x, y))

            texts.append(t)

        adjust_text(texts, expand_text=(1.01, 1.05))

    return ax



def plot_region_around(title, margin=5, **kwargs):

    """Plot the region of the mapping space in the neighbourhood of the the movie with

    the given title. The margin parameter controls the size of the neighbourhood around

    the movie.

    """

    xmargin = ymargin = margin

    match = df[df.title == title]

    assert len(match) == 1

    row = match.iloc[0]

    return plot_region(row.x-xmargin, row.x+xmargin, row.y-ymargin, row.y+ymargin, **kwargs)
# This and several other helper functions are defined in a code cell above. Hit the "code"

# button above if you're curious about how they're implemented.

plot_by_title_pattern('Harry Potter', figsize=(15, 9), bg_alpha=.05, text=False);
plot_region_around('Harry Potter and the Order of the Phoenix', 4);
docs = df[ (df.genres == 'Documentary') ]

plot_with_annotations(docs.index, text=False, alpha=.4, figsize=(15, 8));


import itertools

sample_rate = 1

genre_components = ['Comedy', 'Drama', 'Romance']

genre_combos = set()

for size in range(1, 4):

    combo_strs = ['|'.join(genres) for genres in itertools.combinations(genre_components, size)]

    genre_combos.update(combo_strs)



ax = plot_bg(figsize=(16, 10))

dromcoms = df[df.genres.isin(genre_combos)]

if sample_rate != 1:

    dromcoms = dromcoms.sample(frac=sample_rate, random_state=1)

for i, genre in enumerate(genre_components):

    m = dromcoms[dromcoms.genres.str.contains(genre)]

    marker = str(i+1)

    add_annotations(ax, m.index, label=genre, alpha=.5, marker=marker, s=150, linewidths=5)

plt.legend();