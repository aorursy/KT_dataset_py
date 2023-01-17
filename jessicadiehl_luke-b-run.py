

import gc      # ipython requires us to force garbage collection
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import json as js
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.stem.porter import *
import numpy as np # linear algebra
from operator import itemgetter
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score,silhouette_score
from sklearn.svm import SVC
# the number of words in the vocabulary
vocab_size = 1000

# the dataset has gotten bigger; this ensures that everything fits into memory
# set to -1 to load entire set.
data_set_size = 16000 

# The following two functions are used to clean and normalize the text of the
# papers
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(text)
    #return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess_stem_clean(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result
# Read and preprocess the data sets, creating an array where each row consists of ALL text from one 
# article
i = 0
k = 0

corona_all_text = []
Y=[]
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        i += 1
        if data_set_size != -1 and k > data_set_size:
            break
        if i % 500 == 0:
            print ("Working (number %d)..." % i)
        
        if filename.split(".")[-1] == "json":
            
            f = open(os.path.join(dirname, filename))
            j = js.load(f)
            f.close()
            
            try:
                abstract_text = ' '.join([x['text'] for x in j['abstract']])
            except:
                abstract_text = ""
            body_text = ' '.join(x['text'] for x in j['body_text'])
            body_text += " " + abstract_text
            if "corona" in body_text.lower() or "covid" in body_text.lower():
                Y.append(1)
            else:
                Y.append(0)
            k+=1
            corona_all_text.append(preprocess_stem_clean(body_text))
            
print (k)
print (i)
import csv

with open("caronaOutfile", 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(corona_all_text)
import ast
# read csv file as a list of lists
lines=[]
with open('../input/prestoredoutfile/caronaOutfile.csv', 'r') as read_obj:
    lines = [ast.literal_eval(line) for line in read_obj if line.strip()]
corona_all_text = lines

print(corona_all_text)
# create bag-of-words feature vector
dictionary = corpora.Dictionary(corona_all_text)
dictionary.filter_extremes(no_below=5, no_above=0.8, keep_n=vocab_size, keep_tokens=['corona', 'covid'])
# returns a sparse bag-of-words vector (each dimension represents one word)
# value is the number of times word appears in texts
corpus = [dictionary.doc2bow(text) for text in corona_all_text]

# Space is tight, so force garbage collection on the input data
#corona_all_text = []

# returns a matrix where the rows are the features and columnts are data items
corpus_matrix = gensim.matutils.corpus2dense(corpus, len(dictionary))

# center the features on zero
corpus_mean = np.mean(corpus_matrix, axis=0)
corpus_matrix = corpus_matrix - corpus_mean
print(dictionary)
# Perform PCA by computing the singular value decomposition of the 
# mean-adjusted features (i.e., bag-of-word wordcounts)
#
# u are the eigenvectors of the covariance matrix of the features 
#   (where each column is one eigenvector)
#
# s are the square roots of the eigenvectors
u,s,w = np.linalg.svd(corpus_matrix)

# Plot the eigenvalues. The convergence point represents a good place
# to cut off the dimensions. 
plt.plot(s*s)
plt.show()

# inspect the top eigenvalue (dimension of highest variance)
top = u[:,0] 
top_words = [(dictionary[i], top[i]) for i in range(vocab_size)]
top_words.sort(key=itemgetter(1))

corona_id = dictionary.token2id['corona']

# note: sign here is arbitrary. Any words w/ same valence as corona are important
best_ev_pos = np.argmax(u[corona_id,:])
pos_words = [(dictionary[i], u[i,best_ev_pos]) for i in range(vocab_size)]
pos_words.sort(key=itemgetter(1))
print (pos_words[-20:])

# NEVER do this in your own code (repeat a pattern without making it a subroutine ;)
best_ev_neg = np.argmin(u[corona_id,:])
neg_words = [(dictionary[i], u[i,best_ev_neg]) for i in range(vocab_size)]
neg_words.sort(key=itemgetter(1))
print (neg_words[:20])
# Run k-means
kmeans = KMeans(n_clusters=2).fit(corpus_matrix.T)
# Project into first 50 PCA dimensions
corpus_reduced = corpus_matrix.T.dot(u[:,:50])

# Now lets run a few tests to try to estimate the best number of clusters
db_score = []
ch_score = []
s_score = []
models = {}
lb = 4
ub = 20
for k in range(lb,ub):
    kmeans = KMeans(n_clusters=k).fit(corpus_reduced)
    db_score.append(davies_bouldin_score(corpus_reduced, kmeans.labels_)) # lower is better
    ch_score.append(calinski_harabasz_score(corpus_reduced, kmeans.labels_)) # higher is better
    s_score.append(silhouette_score(corpus_reduced, kmeans.labels_)) # higher is better 
    models[k] = kmeans
    
plt.clf()
plt.plot(range(lb,ub), db_score, color="red")
plt.plot(range(lb,ub), s_score, color ="blue")
plt.show()
plt.plot(range(lb,ub), ch_score)
km8 = models[8]
print(km8.cluster_centers_.shape)
cluster_center_bows = km8.cluster_centers_.dot(u[:,:50].T)
print (cluster_center_bows.shape)
def uncode_BOW(fv):
    # zip might work here
    x = [(dictionary[i], fv[i]) for i in range(vocab_size)]
    return sorted(x,key=itemgetter(1))
 

cluster_report = [print(uncode_BOW(x)[-20:]) for x in cluster_center_bows]
#print (cluster_report)
svm = SVC(kernel='linear').fit(corpus_matrix.T, Y)
print (svm.coef_)