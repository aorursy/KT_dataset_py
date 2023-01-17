# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, sys

from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation, NMF

print(os.listdir("../input"))

import datetime



# Any results you write to the current directory are saved as output.
a1 = pd.read_csv('../input/articles1.csv')

a2 = pd.read_csv('../input/articles2.csv')

a3 = pd.read_csv('../input/articles3.csv')

os.system('read in files!')

sys.stdout.write('read in files')
df = pd.concat([a1,a2,a3])



# save memory

del a1, a2, a3

#df.shape()

df.tail()
split_date = datetime.date(2016,11,8)

pre_election = df[(pd.to_datetime(df['date']) <split_date)]["content"]

post_election = df[(pd.to_datetime(df['date']) >split_date)]["content"]
print(post_election.tail(15))
vector = TfidfVectorizer(stop_words = 'english')

tfidf_post = vector.fit_transform(post_election)

terms_post = vector.get_feature_names()
vector = TfidfVectorizer(stop_words = 'english')

tfidf_pre = vector.fit_transform(pre_election)

terms_pre = vector.get_feature_names()

os.system('finished tfidf')
K = range(1,15)

SSE = []

for k in K:

    kmeans = MiniBatchKMeans(n_clusters = k,batch_size = 300)

    kmeans.fit(tfidf_pre)

    SSE.append(kmeans.inertia_)

    

import matplotlib.pyplot as plt

plt.plot(K,SSE,'bx-')

plt.title('Elbow Method')

plt.xlabel('cluster numbers')

plt.show()

os.system('finished elbow')
# kmeans = KMeans(n_clusters = 8)

# kmeans.fit(tfidf_pre)

# centers = kmeans.cluster_centers_.argsort()[:,::-1]

# terms = vector.get_feature_names()



# for i in range(0,8):

#     word_list=[]

#     print("cluster%d:"% i)

#     for j in centers[i,:15]:

#         word_list.append(terms[j])

#     print(word_list) 

# sys.stdout.write('finished kmeans pre')
# kmeans = KMeans(n_clusters = 8)

# kmeans.fit(tfidf_post)

# centers = kmeans.cluster_centers_.argsort()[:,::-1]

# terms = vector.get_feature_names()



# for i in range(0,k):

#     word_list=[]

#     print("cluster%d:"% i)

#     for j in centers[i,:15]:

#         word_list.append(terms[j])

#     print(word_list)

# sys.stdout.write('finished kmeans post')
nmf  = NMF(n_components = 5)

nmf.fit(tfidf_pre)

for i in range(0,5):

    word_list=[]

    print("Topic%d:"% i)

    for j in nmf.components_.argsort()[i,-16:-1]:

        word_list.append(terms_pre[j])

    print(word_list)

os.system('finished nmf pre')

print(nmf.reconstruction_err_)

nmf  = NMF(n_components = 5)

nmf.fit(tfidf_post)

for i in range(0,5):

    word_list=[]

    print("Topic%d:"% i)

    for j in nmf.components_.argsort()[i,-16:-1]:

        word_list.append(terms_post[j])

    print(word_list)

sys.stdout.write('finished nmf post')

print(nmf.reconstruction_err_)

# nmf  = NMF(n_components = 4)

# nmf.fit(tfidf_pre)

# for i in range(0,k):

#     word_list=[]

#     print("Topic%d:"% i)

#     for j in nmf.components_.argsort()[i,-16:-1]:

#         word_list.append(terms[j])

#     print(word_list)

# sys.stdout.write('finished nmf pre')
# nmf  = NMF(n_components = 4)

# nmf.fit(tfidf_post)

# for i in range(0,k):

#     word_list=[]

#     print("Topic%d:"% i)

#     for j in nmf.components_.argsort()[i,-16:-1]:

#         word_list.append(terms[j])

#     print(word_list)

# sys.stdout.write('finished nmf post')