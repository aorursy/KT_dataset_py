# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import nltk
import gensim
import multiprocessing
import re
import string as str

# from nltk.corpus import stopwords

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
dt = pd.read_csv('/kaggle/input/data-final-1/quran_translasi_final.csv')
dt.head()
dt['text_lower'] = dt['text'].str.replace('[^a-zA-Z]',' ').str.lower()
dt['text_bersih'] = dt['text_lower']
text_bersih = []

for i in dt['text_bersih']:
    text_bersih.append(re.sub("\s\s+", " ", i))

dt['text_bersih'] =  text_bersih
corpus = []

for i in dt['text_bersih']:
    corpus.append(i)
dt.head()
#Identify common words
freq = pd.Series(' '.join(dt['text_bersih']).split()).value_counts()[:50]
freq
# # Tokenize words
dt['text_bersih'] = dt['text_bersih'].str.split()
dt['text_bersih']
corpusX = []

for i in dt['text_lower']:
    corpusX.append(i)
corpusY = []

for i in dt['text_lower'].str.split():
    corpusY.append(i)
from sklearn.feature_extraction.text import TfidfVectorizer

# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True,max_features=200)
 
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpusX)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
important_vocab = tfidf_vectorizer.get_feature_names()
len(important_vocab)
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
test = word2vec.Word2Vec(corpusY, size=200, window=20, min_count=20, workers=4,sg=1, iter = 10)
tsne_plot(test)
len(test.wv.vocab)
filtered_vocab = []
filtered_vector = []
for e in test.wv.vocab:
    if e in important_vocab:
        filtered_vocab.append(e)
        filtered_vector.append(test.wv.get_vector(e))
print('length:', len(filtered_vocab))
filtered_vector = np.array(filtered_vector)
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

l = linkage(filtered_vector, method='complete', metric='euclidean')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='top',
    leaf_label_func=lambda v: (filtered_vocab[v])
)
plt.show()
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=20, affinity='euclidean', linkage='complete')
cluster.fit_predict(filtered_vector)
labels = cluster.labels_
df = pd.DataFrame()
df['word'] = [e for e in filtered_vocab]
df['cluster'] =  labels
groups = list(df.groupby('cluster'))
for g in groups:
    print(g[1])
dt = pd.read_csv('/kaggle/input/indonesian-cleancsv/Indonesian_clean.csv')
dt['text_lower'] = dt['text'].str.replace('[^a-zA-Z]',' ').str.lower()
!pip install PySastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

stop_re = '\\b'+'\\b|\\b'.join(stopwords)+'\\b'
dt['text_bersih'] = dt['text_lower'].str.replace(stop_re,'')

text_bersih = []

for i in dt['text_bersih']:
    text_bersih.append(re.sub("\s\s+", " ", i))

dt['text_bersih'] =  text_bersih
corpus = []

for i in dt['text_bersih']:
    corpus.append(i)
#Identify common words
freq = pd.Series(' '.join(dt['text_bersih']).split()).value_counts()[:50]
freq
# # Tokenize words
dt['text_bersih'] = dt['text_bersih'].str.split()
dt['text_bersih']
!pip install nlp-id
from nlp_id.postag import PosTag
postagger = PosTag()

pos_tag = []

for i in corpus :
    pos_tag.append(postagger.get_pos_tag(i))
    
dt['pos_tag'] = pos_tag
postag_stopword = []
for i in range(len(pos_tag)):
    for j in range(len(pos_tag[i])):
        if pos_tag[i][j][1]!='NN' and pos_tag[i][j][1]!='VB' and pos_tag[i][j][1]!='JJ' and pos_tag[i][j][1]!='FW' and pos_tag[i][j][1]!='NNP' and pos_tag[i][j][1]!='NUM':
                postag_stopword.append(pos_tag[i][j])

postag_stopword
stopword_postag_unique = []
for i in set(postag_stopword) :
    stopword_postag_unique.append(i[0])

stopword_postag_unique.sort()
stopword_postag_unique
dt['text_baru'] = dt['text_bersih'].apply(lambda x: [item for item in x if item not in stopword_postag_unique])
dt['text_baru2'] = dt['text_baru'].str.join(" ")
dt['text_baru2']
corpusX = []

for i in dt['text_baru2']:
    corpusX.append(i)
corpusY = []

for i in dt['text_lower'].str.split():
    corpusY.append(i)
from sklearn.feature_extraction.text import TfidfVectorizer

# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True,max_features=200)
 
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpusX)

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
 
# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
important_vocab = tfidf_vectorizer.get_feature_names()
len(important_vocab)
test = word2vec.Word2Vec(corpusY, size=200, window=20, min_count=20, workers=4,sg=1, iter = 10)
tsne_plot(test)
len(test.wv.vocab)
filtered_vocab = []
filtered_vector = []
for e in test.wv.vocab:
    if e in important_vocab:
        filtered_vocab.append(e)
        filtered_vector.append(test.wv.get_vector(e))
print('length:', len(filtered_vocab))
filtered_vector = np.array(filtered_vector)
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

l = linkage(filtered_vector, method='average', metric='euclidean')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='top',
    leaf_label_func=lambda v: (filtered_vocab[v])
)
plt.show()
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=40, affinity='euclidean', linkage='average')
cluster.fit_predict(filtered_vector)
labels = cluster.labels_
df = pd.DataFrame()
df['word'] = [e for e in filtered_vocab]
df['cluster'] =  labels
groups = list(df.groupby('cluster'))
for g in groups:
    print(g[1])
