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
allTweets=pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',header=None,encoding='cp1252')
allTweets=allTweets.reset_index()

allTweets.head()
tweetsTXT=allTweets[5].tolist()

tweetsIndex=allTweets['index'].tolist()
import re

import nltk



from string import punctuation

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer



stopword = stopwords.words('english')

snowball_stemmer = SnowballStemmer('english')

wordnet_lemmatizer = WordNetLemmatizer()



def cleanTweet(tweet):

    tweetNoAT= re.sub(r'\@\w+','', tweet)

    tweetNoURL=re.sub(r'http\S+','',tweetNoAT)

    lower=tweetNoURL.lower()

    lower=''.join(c for c in lower if c not in punctuation)

    word_tokens = nltk.word_tokenize(lower)

    word_tokens = [word for word in word_tokens if word not in stopword]

    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]

    stemmed_word = [snowball_stemmer.stem(word) for word in lemmatized_word]

    return stemmed_word
cleanTweet('@bestCATFRD check this out, the cat video, https:/asdasd!')
masterRAW=list(map(lambda x:cleanTweet(x), tweetsTXT))
from collections import Counter

## Create index2word and word2index

masterWord=[]

for indivList in masterRAW:

    masterWord+=indivList

counterDict=Counter(masterWord)

commonWords=counterDict.most_common(2000)
indexList=list(range(2000))

word2idx={}

idx2word={}

for key,val in zip(indexList,commonWords):

    word2idx[val[0]]=key

    idx2word[key]=val[0]
totLen=len(masterRAW)

def makeBatch(batchNum,batchSize):

    currentX=[]

    currentY=[]

    currentIdx=[]

    slips=masterRAW[batchNum*batchSize:(batchNum+1)*batchSize]

    counter=0

    for sent in slips:

        if len(sent)<2:

            continue

        else: 

            for wordIdx in range(len(sent)-1):

                if sent[wordIdx] in word2idx and sent[wordIdx+1] in word2idx:

                    currentX.append(word2idx[sent[wordIdx]])

                    currentIdx.append(batchNum*batchSize+counter)

                    currentY.append([word2idx[sent[wordIdx+1]]])

                    currentX.append(word2idx[sent[wordIdx+1]])

                    currentIdx.append(batchNum*batchSize+counter)

                    currentY.append([word2idx[sent[wordIdx]]])

        counter+=1

    return np.array(currentX),np.array(currentIdx),np.array(currentY)

                    
import math

import tensorflow as tf

batch_size = 160

embedding_size = 32

doc_embedding_size=5



train_inputs=tf.placeholder(tf.int32, shape=[None])

train_docs=tf.placeholder(tf.int32,shape=[None])

train_labels=tf.placeholder(tf.int32, shape=[None,1])



embeddings = tf.Variable(tf.random_uniform((2000, embedding_size), -1, 1))

embeddingDoc=tf.Variable(tf.random_uniform((1600000,doc_embedding_size),-1,1))



embedWord = tf.nn.embedding_lookup(embeddings, train_inputs)

embedDoc=tf.nn.embedding_lookup(embeddingDoc,train_docs)



embed=tf.concat([embedWord,embedDoc],axis=1,name='concat')





nce_weights = tf.Variable(tf.truncated_normal([2000, embedding_size+doc_embedding_size],

                                              stddev=1.0 / math.sqrt(embedding_size+doc_embedding_size)))

nce_biases = tf.Variable(tf.zeros([2000]))



nce_loss = tf.reduce_mean(

    tf.nn.nce_loss(weights=nce_weights,

                   biases=nce_biases,

                   labels=train_labels,

                   inputs=embed,

                   num_sampled=200,

                   num_classes=2000))



optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)
for epoch in range(10):

    idx=0

    tempLossTOT=0.0

    for batchIndex in range(int(len(masterRAW)/128)-1):

        trainX,trainIndex,trainY=makeBatch(batchIndex,128)

        loss,_ = sess.run([nce_loss,optimizer],feed_dict={train_inputs:trainX,train_docs:trainIndex,train_labels:trainY})

        tempLossTOT+=loss

        if batchIndex%1000==0:

            print('Current Loss: '+str(tempLossTOT/(batchIndex+1)*1.0 ))

embeddingMat=sess.run(embeddings)
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(embeddingMat)

X_embedded.shape


col1=[x[0] for x in X_embedded]

col2=[x[1] for x in X_embedded]

keys=word2idx.keys()
tsnedEmbedding=pd.DataFrame()

tsnedEmbedding['word']=keys

tsnedEmbedding['dim1']=col1

tsnedEmbedding['dim2']=col2

tsnedEmbedding.to_csv('2DEmbedding.csv')

import plotly.express as px

import matplotlib.pyplot as plt



x = tsnedEmbedding['dim1'].tolist()

y = tsnedEmbedding['dim2'].tolist()

n = tsnedEmbedding['word'].tolist()



fig = px.scatter(tsnedEmbedding, x="dim1", y="dim2", text="word", size_max=60)

fig.update_traces(textposition='top center')

fig.update_layout(

    height=800,

    title_text='Embedding Two-D Plot'

)

fig.show()

totLen
