import pandas as pd

import numpy as np

import gensim.models.word2vec as w2v

import multiprocessing

import os

import re

import pprint

import sklearn.manifold

import matplotlib.pyplot as plt
songs = pd.read_csv("../input/songdata.csv", header=0)

#songs.head()

songs = songs[songs.artist != 'Lata Mangeshkar']

songs.head()
text_corpus = []

for song in songs['text']:

    words = song.lower().split()

    text_corpus.append(words)







# Dimensionality of the resulting word vectors.

#more dimensions, more computationally expensive to train

#but also more accurate

#more dimensions = more generalized

num_features = 50

# Minimum word count threshold.

min_word_count = 1



# Number of threads to run in parallel.

#more workers, faster we train

num_workers = multiprocessing.cpu_count()



# Context window length.

context_size = 7





downsampling = 1e-1



# Seed for the RNG, to make the results reproducible.

#random number generator

#deterministic, good for debugging

seed = 1



songs2vec = w2v.Word2Vec(

    sg=1,

    seed=seed,

    workers=num_workers,

    size=num_features,

    min_count=min_word_count,

    window=context_size,

    sample=downsampling

)



songs2vec.build_vocab(text_corpus)

print (len(text_corpus))
import time

start_time = time.time()







songs2vec.train(text_corpus)



if not os.path.exists("trained"):

    os.makedirs("trained")



songs2vec.save(os.path.join("trained", "songs2vectors.w2v"))



print("--- %s seconds ---" % (time.time() - start_time))
songs2vec = w2v.Word2Vec.load(os.path.join("trained", "songs2vectors.w2v"))
print(songs2vec['un-right'])

def songVector(row):

    vector_sum = 0

    words = row.lower().split()

    for word in words:

        vector_sum = vector_sum + songs2vec[word]

    vector_sum = vector_sum.reshape(1,-1)

    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)

    return normalised_vector_sum





import time

start_time = time.time()



songs['song_vector'] = songs['text'].apply(songVector)





song_vectors = []

from sklearn.model_selection import train_test_split



train, test = train_test_split(songs, test_size = 0.9)





for song_vector in train['song_vector']:

    song_vectors.append(song_vector)



train.head(10)
X = np.array(song_vectors).reshape((5761, 50))



start_time = time.time()

tsne = sklearn.manifold.TSNE(n_components=2, n_iter=200, random_state=0, verbose=2)



all_word_vectors_matrix_2d = tsne.fit_transform(X)



print("--- %s seconds ---" % (time.time() - start_time))
df=pd.DataFrame(all_word_vectors_matrix_2d,columns=['X','Y'])



df.head(10)



train.head()



df.reset_index(drop=True, inplace=True)

train.reset_index(drop=True, inplace=True)
two_dimensional_songs = pd.concat([train, df], axis=1)



two_dimensional_songs.head()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



import plotly.graph_objs as go



trace1 = go.Scatter(

    y = two_dimensional_songs['Y'],

    x = two_dimensional_songs['X'],

    text = two_dimensional_songs['song'],

    mode='markers',

    marker=dict(

        size='7',

        color = np.random.randn(5717), #set color equal to a variable

        colorscale='Viridis',

        showscale=True

    )

)

data = [trace1]



iplot(data)