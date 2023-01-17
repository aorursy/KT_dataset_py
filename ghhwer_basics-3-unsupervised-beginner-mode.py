import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
dataset = pd.read_csv('../input/bbc-fulltext-and-category/bbc-text.csv')

dataset.head()
dataset.text[0] #The data was already prepared and stop words were removed :(
import re, string

vectorizer = TfidfVectorizer(ngram_range=(1,2),

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1, stop_words='english')

X = vectorizer.fit_transform(dataset.text)

word_vects = X.toarray()

word_vects.shape
import umap



reducer = umap.UMAP(random_state=70,metric='cosine')

embedding = reducer.fit_transform(word_vects)
import matplotlib.pyplot as plt

plt.scatter(embedding[:,0], embedding[:,1])

plt.title("UMAP dimentionality Reduction")

plt.show()
from sklearn.cluster import KMeans



clustering = KMeans(n_clusters=6, init='k-means++').fit(embedding)



dataset['cluster'] = clustering.labels_

dataset['vectX'] = embedding[:,0]

dataset['vectY'] = embedding[:,1]

dataset.cluster.unique()

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

for x in dataset.cluster.unique():

    vctsX = dataset.loc[dataset.cluster == x].vectX

    vctsY = dataset.loc[dataset.cluster == x].vectY

    c = dataset.loc[dataset.cluster == x].cluster

    plt.title("K-means Clustering")

    plt.scatter(vctsX, vctsY, c=np.random.rand(3,), label=x)

    plt.legend(loc='upper left')
cluster2cat = {}



for x in dataset.cluster.unique():

    cat = {}

    ds = dataset.loc[dataset.cluster == x]

    for y in ds.category.unique():

        cat[y] = ds.loc[ds.category == y].count()['category']

    print(x, 'Shows labeled data of:', cat)

    i = 0

    # Get the most frequent label

    selected = list(cat.values()).index(max(cat.values()))

    cluster2cat[x] = list(cat.keys())[selected]

print("Mapping is:",cluster2cat)
dataset['cluster_class'] = dataset['cluster'].map(cluster2cat)

confusion_matrix = pd.crosstab(dataset.category, dataset.cluster_class, rownames=['Actual'], colnames=['Predicted'])

accuracy = confusion_matrix.values.diagonal().sum()/(confusion_matrix.values.sum())

print("Accuracy: %.2f"%(100*accuracy)+"%")

confusion_matrix.head(10)