import pandas as pd

import numpy as np

import random

#data imported from https://www.kaggle.com/jatindersehdev/poetry-analysis-data

data=pd.read_csv('../input/all.csv')
data.head()
data['type'].value_counts()
data['age'].value_counts()
data['content'].str.split().str.len().sum()
data['content'].str.split().str.len().value_counts().head()
# Taking only small poems

new_data=data[data['content'].str.split().str.len()<245]

new_data['content'].str.split().str.len().sum()
new_data['type'].value_counts()

#My dataset is similar even after removing long poems
new_data['age'].value_counts()
poems=new_data['content']
poems.str.split().head()
#applying PorterStemming

from nltk.stem import PorterStemmer

ps = PorterStemmer()

def stem_sentences(sentence):

    tokens = sentence.split()

    stemmed_tokens=[]

    for token in tokens:

        #Tokenize and also get rid of any punctuation

        token.replace(' @$/#.-:&*+=[]?!(){},''">_<;%','')

        #Remove any non alphanumeric characters

        token.replace('[^a-zA-Z0-2019]', '')

        #applying porter stemming to my content

        if len(token)>1:

            #porter stemming does not work sometimes

            try:

                temp=ps.stem(token)

                stemmed_tokens.append(temp)

            except temp == '':

                continue

    #stemmed_tokens = [ps.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)

poems_stem= poems.apply(stem_sentences)

poems_stem=poems_stem.str.replace('\n', " ")

poems_stem=poems_stem.str.replace("\t", " ")

poems_stem=poems_stem.str.replace("\r", " ")

poems_stem=poems_stem.str.replace(","," ").replace("."," ")

poems_stem=poems_stem.str.replace("\W"," ")

poems_stem.head()
#No. of words in total

poems_stem.str.split().str.len().sum()
all_poems=[]

for poem in poems_stem:

    all_poems.append(poem)
# using Hashing Vectorizer

from sklearn.feature_extraction.text import HashingVectorizer

# create the transform

vectorizer = HashingVectorizer(n_features=20)

# encode document

vector = vectorizer.transform(all_poems)

# summarize encoded vector

print(vector.shape)

X=vector.toarray()

X
m,n=vector.shape

#K-means Clustering

K=4



def findClosestCentroids(X, centroids):

    idx = np.zeros(m,int)

    for i in range(m):

        Mindistance=sum(np.square(X[i]-centroids[0]))

        idx[i]=0

        for j in range(1,K):

            distance=sum(np.square(X[i]-centroids[j]))

            if (distance<Mindistance):

                Mindistance=distance

                idx[i]=j

    return idx           

                

def computeCentroids(X, idx, K):

    sum=np.zeros(n)

    count=0

    for i in range(K):

        sum=np.zeros(n)

        count=0

        for j in range(m):

            if(i==idx[j]):

                count=count+1

                sum=sum+X[j]

        if(count==0):

            centroids[i]=X[random.randint(1,m)]

        else:

            centroids[i]=sum/count

    return centroids



#from np.random import permutation

randidx= np.random.permutation(m)

centroids=X[randidx[1:K+1]]

max_iter=100

idx = np.zeros(m,int)

new_idx= np.zeros(m,int)

for i in range(max_iter):

    # For each example in X, assign it to the closest centroid

    if i==0:

        idx = findClosestCentroids(X, centroids)

    else:

        new_idx=findClosestCentroids(X, centroids)

    if np.array_equal(idx,new_idx) and i!=0:

        temp=i

        break

    else:

        idx=new_idx

    # Given the memberships, compute new centroids

    centroids = computeCentroids(X, idx, K)

centroids

idx_kmeans=idx

y = np.bincount(idx)

ii = np.nonzero(y)[0]

np.vstack((ii,y[ii])).T

y[ii]
from sklearn.cluster import SpectralClustering

clustering=SpectralClustering(n_clusters=4,assign_labels="discretize",random_state=0).fit(X)

idx=clustering.labels_

idx_spectral=idx

y = np.bincount(idx)

ii = np.nonzero(y)[0]

np.vstack((ii,y[ii])).T

y[ii]
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])

principalDf.head()
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

legend = ['Cluster-1', 'Cluster-2', 'Cluster-3','Cluster-4']

targets=[0,1,2,3]

colors = ['r', 'g', 'b','y']

for target, color in zip(targets,colors):

    indicesToKeep= idx_spectral==target

    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']

               , principalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(legend)

ax.grid()