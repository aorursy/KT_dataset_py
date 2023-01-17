%matplotlib inline
%matplotlib inline

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

plt.style.use('ggplot')

con = sqlite3.connect('../input/database.sqlite')
email_df = pd.read_sql_query("""
SELECT * from Emails
""", con)

email_df.info()
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

data = email_df["RawText"]
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(data)
svd = TruncatedSVD(100)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)
x_vect = np.arange(3,100,5)
y_vect = np.zeros(x_vect.shape)
for i, cl in enumerate(x_vect):
    km = KMeans(n_clusters=cl, init='k-means++', max_iter=100, n_init=1,
                verbose=0)
    km.fit(X)
    dist = np.min(cdist(X,km.cluster_centers_,'euclidean'),axis=1)
    y_vect[i] = np.sum(dist)/X.shape[0]
    
plt.plot(x_vect,y_vect,marker="o")
plt.ylim([0,1])
k = 23
km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
                verbose=0)
km.fit(X)
pd.Series(km.labels_).value_counts()
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

from sklearn.decomposition import LatentDirichletAllocation
cv_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000,
                                stop_words='english')
cv = cv_vectorizer.fit_transform(data)
n_topics = 25
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(cv)

cv_feature_names = cv_vectorizer.get_feature_names()
n_words = 10
for topic_idx, topic in enumerate(lda.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([cv_feature_names[i]
                        for i in topic.argsort()[:-n_words - 1:-1]]))
cv_t = lda.transform(cv)
cv_t.shape
#cv_t/cv_t.sum(axis=1).reshape((-1,1))
lda.score(cv)
