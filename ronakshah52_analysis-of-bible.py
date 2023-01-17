import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import spacy

import gensim

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

import math

import time

import plotly.express as px

import sys

from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())

import os

RS = 123
data = pd.read_csv("../input/bible/t_asv.csv")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data['t'])]
model = Doc2Vec(documents, vector_size=10, workers=4)
vector = [model.infer_vector([i]) for i in list(data['t'])]

vector = np.array(vector)
perplex = math.sqrt(vector.shape[0])

RS = 123

time_start = time.time()

tsne = TSNE(perplexity = perplex, learning_rate = 100, n_iter = 700, random_state=RS).fit_transform(vector)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
columns = list(data.columns)

columns.extend(['comp1', 'comp2'])

data_filter = np.concatenate((data.to_numpy(),tsne), axis = 1)

data_filter = pd.DataFrame(data_filter, columns = columns)

data_filter.head(2)
fig = px.scatter(data_filter, x="comp1", y="comp2", hover_data=["t"], color="b")

fig.show()
sse = {}

for k in range(1, 21):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(vector)

    clusters = kmeans.labels_

    #print(data["clusters"])

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()

n_components = np.arange(1, 51)

models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(vector) for n in n_components]

plt.plot(n_components, [m.bic(vector) for m in models], label='BIC')

plt.plot(n_components, [m.aic(vector) for m in models], label='AIC')

plt.legend(loc='best')

plt.xlabel('n_components');
gmm = GaussianMixture(n_components=20)

gmm.fit(vector)

labels = gmm.predict(vector)
perplex = math.sqrt(vector.shape[0])

RS = 123

time_start = time.time()

tsne = TSNE(perplexity = perplex, learning_rate = 100, n_iter = 700, random_state=RS).fit_transform(vector)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
columns = list(data.columns)

columns.extend(['comp1', 'comp2', 'labels'])

data_filter = np.concatenate((data.to_numpy(),tsne, labels.reshape(31103,1)), axis = 1)

data_filter = pd.DataFrame(data_filter, columns = columns)

data_filter.head(2)
fig = px.scatter(data_filter, x="comp1", y="comp2", hover_data=["t"], color="labels")

fig.show()