import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re, itertools, random

import matplotlib.pyplot as plt

import time



from gensim.models import word2vec

from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans



from scipy.spatial.distance import cdist
def read_data():

    global input_train, input_test

    # Read full training and test sets

    input_train = pd.read_csv("../input/train.tsv", sep = "\t")

    input_test = pd.read_csv("../input/test.tsv", sep = "\t")

    print("Read input")



# Read the raw data

read_data()
sentences = [s for s in [s.strip().lower().split(" ") for i in input_train["item_description"].values for s in re.split("\.", str(i))] if len(s) > 2]
embedding_size = 20

starttime = time.time()

model = word2vec.Word2Vec(sentences, size=embedding_size, window=5, min_count=5, workers=4)

endtime = time.time()

print("Trained word2vec model in " + str(int(np.floor((endtime - starttime)/60))) + "m " + str(int((endtime - starttime)%60)) + "s.")
lookup_words = ["shorts", "xbox", "shoes"]

words_to_visualize = [] # Save for visualization below



for w in lookup_words:

    print(w)

    for s in model.wv.most_similar([w]):

        print(s)

        words_to_visualize.append(s[0])
# First get the embeddings into a matrix

embeddings = np.zeros((len(model.wv.index2word), embedding_size))

for i in range(0, len(model.wv.index2word)):

    w = model.wv.index2word[i]

    embeddings[i] = model.wv[w]



# Look at a few samples

for i in range(1000, 1003):

    print(model.wv.index2word[i] + ":\n" + str(embeddings[i]))
svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=500, random_state=101)

embeddings_2d_projection = svd.fit_transform(embeddings)
sample = np.in1d(model.wv.index2word, words_to_visualize)

x = embeddings_2d_projection[sample,0]

y = embeddings_2d_projection[sample,1]

plt.figure(figsize=(7,7))

plt.scatter(x, y)

for i, txt in enumerate([model.wv.index2word[i] for i in np.where(sample)[0]]):

    plt.annotate(txt, (x[i], y[i]))
# Train a K-means cluster model with 6 clusters

n_clusters = 6

embedding_cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
centroid_embedding_nearest_words = []

for centroid_embedding in embedding_cluster_model.cluster_centers_:

    centroid_embedding_nearest_words.append(

        np.argsort([i[0] for i in cdist(embeddings, np.array([centroid_embedding]), "euclidean")])[0:20]

    )
plt.figure(figsize=(10,10))

colors = itertools.cycle(["b","g","r","c","m","y","k","w"])

c = 0

for word_indices in centroid_embedding_nearest_words:

    clr = next(colors)

    plt.scatter(

        embeddings_2d_projection[word_indices,0],

        embeddings_2d_projection[word_indices,1],

        color=clr,

        label="Cluster " + str(c)

    )

    for ix in word_indices:

        x, y = embeddings_2d_projection[ix,:]

        plt.annotate(model.wv.index2word[ix], (x, y))

    c+=1

plt.legend(loc='lower left')

plt.show()