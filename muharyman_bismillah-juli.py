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
dt = pd.read_csv('/kaggle/input/data-final/quran_translasi_final.csv')

# dt = pd.read_csv('/kaggle/input/indonesian-cleancsv/Indonesian_clean.csv')

dt.head()
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
# # Tokenize words

dt['text_bersih'] = dt['text_bersih'].str.split()
!pip install nlp-id
from nlp_id.postag import PosTag

postagger = PosTag()



pos_tag = []



for i in corpus :

    pos_tag.append(postagger.get_pos_tag(i))

    

dt['pos_tag'] = pos_tag
postag_aaa = []

for i in range(len(pos_tag)):

    for j in range(len(pos_tag[i])):

        postag_aaa.append(pos_tag[i][j])
postag_unique = []

for i in set(postag_aaa) :

    postag_unique.append(i)

postag_unique.sort()
postag_uniquea = []

for i in set(postag_aaa) :

    postag_uniquea.append(i[1])

postag_uniquea.sort()

postag_uniquea = set(postag_uniquea)

postag_uniquea
postag_stopword = []

for i in range(len(pos_tag)):

    for j in range(len(pos_tag[i])):

        if pos_tag[i][j][1]!='NN' and pos_tag[i][j][1]!='VB' and pos_tag[i][j][1]!='JJ' and pos_tag[i][j][1]!='FW' and pos_tag[i][j][1]!='NNP' and pos_tag[i][j][1]!='NUM':

                postag_stopword.append(pos_tag[i][j])
stopword_postag_unique = []

for i in set(postag_stopword) :

    stopword_postag_unique.append(i[0])



stopword_postag_unique.remove('mahapengasih')

stopword_postag_unique.remove('salat')

stopword_postag_unique.sort()
dt['text_baru'] = dt['text_bersih'].apply(lambda x: [item for item in x if item not in stopword_postag_unique])
dt['text_baru2'] = dt['text_baru'].str.join(" ")
corpusX = []



for i in dt['text_baru2']:

    corpusX.append(i)
corpusY = []



for i in dt['text_lower'].str.split():

    corpusY.append(i)
#instantiate CountVectorizer()

cv=CountVectorizer(max_features=3000)

 

# this steps generates word counts for the words in your docs

word_count_vector=cv.fit_transform(corpusX)



tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(word_count_vector)



# print idf values

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

 

# sort ascending

df_idf.sort_values(by=['idf_weights'],ascending = False)
from sklearn.feature_extraction.text import TfidfVectorizer



# settings that you use for count vectorizer will go here

tfidf_vectorizer=TfidfVectorizer(max_features=250)

# just send in all your docs here

tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpusX)



# get the first vector out (for the first document)

first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

 

# place tf-idf values in a pandas data frame

df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

df.sort_values(by=["tfidf"],ascending=False)
important_vocab = tfidf_vectorizer.get_feature_names()

len(important_vocab)
pengurangan = ['untukmu','tuhannya','tuhanmu','tuhanku','sisi','penuh','manakah','ketahuilah','kepadamu','kepadaku','kebanyakan',

              'kaumnya','kaumku','hatinya','engkau','dirimu','darinya','bertakwalah','berlaku','berilah','berdua','barangsiapa',

              'barang','balasan','baginya','bagimu','alasan','adakan','allah']
hasil_pengurangan = []



for i in important_vocab:

    if i not in pengurangan:

        hasil_pengurangan.append(i)

        

important_vocab = hasil_pengurangan
tambahan = ['injil','sabar','angin','beruntung','gaib','adam','batu','laut','mekah','zakat','putra','iblis','yakub','sihir','iblis','jahat'

       ,'mahateliti','tahun','keluarga','mahatinggi','samud','sapi','sulaiman','unta','sujud','kota','burung','berjihad','berserah'

       ,'islam','keturunan','kurma','muslim','nasrani','suami','kikir','harun','ayah','bersabar','bertasbih','dawud','dusta','ishak','jibril','kubur','mahaesa','mahakaya','petang'

       ,'syuaib''zabur']
for i in tambahan:

    important_vocab.append(i)
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
test = word2vec.Word2Vec(corpusY, size=200, window=10, min_count=3, workers=4,sg=1, iter = 10)
tsne_plot(test)
test.most_similar('kitab')
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

plt.figure(figsize=(150, 60))

plt.title('Hierarchical Clustering Dendrogram')

plt.ylabel('word')

plt.xlabel('distance')



dendrogram(

    l,

    leaf_rotation=0.,  # rotates the x axis labels

    leaf_font_size=16.,  # font size for the x axis labels

    orientation='right',

    leaf_label_func=lambda v: (filtered_vocab[v])

)

plt.show()
from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='average',distance_threshold=2.73)

# cluster = AgglomerativeClustering(n_clusters=77, affinity='euclidean', linkage='average',compute_full_tree = False)

cluster.fit_predict(filtered_vector)

labels = cluster.labels_
df = pd.DataFrame()

df['word'] = [e for e in filtered_vocab]

df['cluster'] =  labels
groups = list(df.groupby('cluster'))
for i in groups:

    print(i)
df.head()
# df.to_csv('konsep_relasi.csv')