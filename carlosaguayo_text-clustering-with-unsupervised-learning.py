import os

import numpy as np

import pandas as pd



from sklearn.datasets import fetch_20newsgroups

from sklearn.datasets.base import get_data_home



from sklearn.cluster import KMeans



from sklearn.feature_extraction.text import TfidfVectorizer
# for reproducible results

np.random.seed(777)
data_home = get_data_home()

twenty_home = os.path.join(data_home, "20news_home")



if not os.path.exists(data_home):

    os.makedirs(data_home)

    

if not os.path.exists(twenty_home):

    os.makedirs(twenty_home)

    

# !cp ../input/20-newsgroup-sklearn/20news-bydate_py3* /tmp/scikit_learn_data

!cp ../input/20news-bydate_py3* /tmp/scikit_learn_data
# http://qwone.com/~jason/20Newsgroups/

dataset = fetch_20newsgroups(subset='all', shuffle=True, download_if_missing=False)



texts = dataset.data # Extract text

target = dataset.target # Extract target
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(texts)
number_of_clusters = 20



# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

model = KMeans(n_clusters=number_of_clusters, 

               init='k-means++', 

               max_iter=100, # Maximum number of iterations of the k-means algorithm for a single run.

               n_init=1)  # Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.



model.fit(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(number_of_clusters):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :10]:

        print(' %s' % terms[ind])
print (texts[400])
X = vectorizer.transform([texts[400]])



cluster = model.predict(X)[0]



print("Text belongs to cluster number {0}".format(cluster))
for ind in order_centroids[cluster, :10]:

    print(' %s' % terms[ind])