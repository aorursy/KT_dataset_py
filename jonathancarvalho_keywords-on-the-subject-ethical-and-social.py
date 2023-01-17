import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

import nltk

import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

import os
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
metadata.head()
metadata.info()
metadata_filter = metadata[metadata.abstract.str.contains('ethics|ethical|social science|multidisciplinary research', 

                                                          regex= True, na=False)].reset_index(drop=True)
len(metadata_filter)
metadata_filter.abstract[1]
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

STOPWORDS = set(stopwords.words('english'))



def clean_text(text):

    text = text.lower()

    text = REPLACE_BY_SPACE_RE.sub(' ', text)

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)

    return text



metadata_filter['clean_abstract'] = metadata_filter['abstract'].apply(clean_text)
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

X = vectorizer.fit_transform(metadata_filter['clean_abstract'])
tsne = TSNE(perplexity=4, random_state=42)



X_tsne = tsne.fit_transform(X)

X_tsne = pd.DataFrame(data=X_tsne, columns=['D1', 'D2'])
fig, ax = plt.subplots(figsize=(12,8))

sns.scatterplot(ax=ax,x = 'D1', y = 'D2', data=X_tsne, alpha=0.7)

plt.show()
kmeans = KMeans(n_clusters=2, random_state=42)

kmeans.fit(X)
X_tsne['CLUSTER'] = kmeans.predict(X)



fig, ax = plt.subplots(figsize=(12,8))

sns.scatterplot(ax=ax,x = 'D1', y = 'D2', hue = 'CLUSTER', data=X_tsne, alpha=0.7)

plt.show()
metadata_filter['cluster'] = kmeans.predict(X)

metadata_filter.groupby('cluster')['cluster'].count()
X1 = metadata_filter.loc[metadata_filter['cluster'] == 0, 'clean_abstract']

X2 = metadata_filter.loc[metadata_filter['cluster'] == 1, 'clean_abstract']



tf_vectorizer1 = CountVectorizer(max_features=2000, stop_words='english')

tf_vectorizer2 = CountVectorizer(max_features=2000, stop_words='english')



X1 = tf_vectorizer1.fit_transform(X1)

X2 = tf_vectorizer2.fit_transform(X2)



lda1 = LatentDirichletAllocation(n_components=5, random_state=42)

lda2 = LatentDirichletAllocation(n_components=5, random_state=42)



lda1.fit(X1)

lda2.fit(X2)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)
print_top_words(lda1, tf_vectorizer1.get_feature_names(), 5)
print_top_words(lda2, tf_vectorizer2.get_feature_names(), 5)