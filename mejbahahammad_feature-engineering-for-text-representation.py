import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import re

import nltk

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

pd.options.display.max_colwidth = 200

%matplotlib inline
# building a corpus of documents

corpus = ['The sky is blue and beautiful.',

'Love this blue and beautiful sky!',

'The quick brown fox jumps over the lazy dog.',

"A king's breakfast has sausages, ham, bacon, eggs, toast and beans",

'I love green eggs, ham, sausages and bacon!',

'The brown fox is quick and the blue dog is lazy!',

'The sky is very blue and the sky is very beautiful today',

'The dog is lazy but the brown fox is quick!'

]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals',

'weather', 'animals']
corpus = np.array(corpus)

corpus_df = pd.DataFrame({'Document': corpus, 'Category': labels})

corpus_df = corpus_df[['Document', 'Category']]

corpus_df
wpt = nltk.WordPunctTokenizer()

stop_words = nltk.corpus.stopwords.words('english')
def normalize_document(doc):

    # lowercase and remove special characters\whitespace

    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)

    doc = doc.lower()

    doc = doc.strip()

    # tokenize document

    tokens = wpt.tokenize(doc)

    # filter stopwords out of document

    filtered_tokens = [token for token in tokens if token not in stop_words]

    # re-create document from filtered tokens

    doc = ' '.join(filtered_tokens)

    return doc
normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(corpus)
print(norm_corpus)
cv = CountVectorizer(min_df=0., max_df=1.)

cv_matrix = cv.fit_transform(norm_corpus)

cv_matrix
# view non-zero feature positions in the sparse matrix

print(cv_matrix)
# view dense representation

# warning might give a memory error if data is too big

cv_matrix = cv_matrix.toarray()

cv_matrix
# get all unique words in the corpus

vocab = cv.get_feature_names()

# show document feature vectors

pd.DataFrame(cv_matrix, columns=vocab)
# you can set the n-gram range to 1,2 to get unigrams as well as bigrams

bv = CountVectorizer(ngram_range=(2,2))

bv_matrix = bv.fit_transform(norm_corpus)
bv_matrix = bv_matrix.toarray()

vocab = bv.get_feature_names()

pd.DataFrame(bv_matrix, columns=vocab)
tt = TfidfTransformer(norm='l2', use_idf=True)

tt_matrix = tt.fit_transform(cv_matrix)
tt_matrix = tt_matrix.toarray()

vocab = cv.get_feature_names()

pd.DataFrame(np.round(tt_matrix, 2), columns=vocab)
tv = TfidfVectorizer(min_df=0., max_df=1., norm='l2',

use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)

tv_matrix = tv_matrix.toarray()
vocab = tv.get_feature_names()

pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
vocab = tv.get_feature_names()

pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
# get unique words as feature names

unique_words = list(set([word for doc in [doc.split() for doc in norm_corpus]

for word in doc]))
def_feature_dict = {w: 0 for w in unique_words}

print('Feature Names:', unique_words)

print()

print('Default Feature Dict:', def_feature_dict)
# build bag of words features for each document - term frequencies

bow_features = []

for doc in norm_corpus:

    bow_feature_doc = Counter(doc.split())

    all_features = Counter(def_feature_dict)

    bow_feature_doc.update(all_features)

    bow_features.append(bow_feature_doc)

bow_features = pd.DataFrame(bow_features)

bow_features
import scipy.sparse as sp

feature_names = list(bow_features.columns)

# build the document frequency matrix

df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)

df = 1 + df # adding 1 to smoothen idf later

# show smoothened document frequencies

pd.DataFrame([df], columns=feature_names)
# compute inverse document frequencies

total_docs = 1 + len(norm_corpus)

idf = 1.0 + np.log(float(total_docs) / df)

# show smoothened idfs

pd.DataFrame([np.round(idf, 2)], columns=feature_names)
# compute idf diagonal matrix

total_features = bow_features.shape[1]

idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)

idf_dense = idf_diag.todense()

# print the idf diagonal matrix

pd.DataFrame(np.round(idf_dense, 2))
# compute tfidf feature matrix

tf = np.array(bow_features, dtype='float64')

tfidf = tf * idf

# view raw tfidf feature matrix

pd.DataFrame(np.round(tfidf, 2), columns=feature_names)
from numpy.linalg import norm

# compute L2 norms

norms = norm(tfidf, axis=1)

# print norms for each document

print (np.round(norms, 3))
# compute normalized tfidf

norm_tfidf = tfidf / norms[:, None]

# show final tfidf feature matrix

pd.DataFrame(np.round(norm_tfidf, 2), columns=feature_names)
new_doc = 'the sky is green today'

pd.DataFrame(np.round(tv.transform([new_doc]).toarray(), 2),

columns=tv.get_feature_names())
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)

similarity_df = pd.DataFrame(similarity_matrix)

similarity_df
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_matrix, 'ward')

pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2',

'Distance', 'Cluster Size'], dtype='object')
plt.figure(figsize=(8, 3))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('Data point')

plt.ylabel('Distance')

dendrogram(Z)

plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
from scipy.cluster.hierarchy import fcluster

max_dist = 1.0

cluster_labels = fcluster(Z, max_dist, criterion='distance')

cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])

pd.concat([corpus_df, cluster_labels], axis=1)
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)

dt_matrix = lda.fit_transform(cv_matrix)

features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])

features
tt_matrix = lda.components_

for topic_weights in tt_matrix:

    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]

    topic = sorted(topic, key=lambda x: -x[1])

    topic = [item for item in topic if item[1] > 0.6]

    print(topic)

    print()