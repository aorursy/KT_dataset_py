import os

import jieba

import re



import numpy as np

import pandas as pd

import tensorflow as tf

import jieba.analyse as analyse
os.listdir('../input/jieba-helper/')
rawReviews=pd.read_csv('../input/doubanmovieshortcomments/DMSC.csv')
rawReviews.head(5)
comments=rawReviews['Comment'].tolist()
swRaw=open('../input/jieba-helper/stopwords.txt','r')

wLst=swRaw.readlines()

swRaw.close()

wLst=[x.strip() for x in wLst]

def processLine(singleReview):

    seg_list = jieba.cut(singleReview, cut_all=False, HMM=True)

    tempToken=[]

    tokenList=[]

    for i in seg_list:

        tempToken=re.findall(r'[\u4e00-\u9fff]+',i)

        if len(tempToken)==0:

            pass

        else:

            if tempToken[0] not in wLst:

                tokenList.append(tempToken[0])

    return tokenList

masterReview=[processLine(x) for x in comments]
print('Total number of reviews: '+str(len(masterReview)))
from collections import Counter

allWords=[]

for lst in masterReview:

    allWords+=lst

cnter=Counter(allWords)
mostFrequent=cnter.most_common(2000)
word2idx={}

idx2word={}



for idx,freqWord in zip(list(range(2000)),mostFrequent):

    word2idx[freqWord[0]]=idx

    idx2word[idx]=freqWord[0]

cleanMasterReview=[]

tempRev=[]

freqWord=[x[0] for x in mostFrequent]

for sent in masterReview:

    tempRev=[]

    for wd in sent:

        if wd in freqWord:

            tempRev.append(wd)

        else:

            continue

    cleanMasterReview.append(tempRev)   
def makeBatch(batchNum,batchSize):

    temp=cleanMasterReview[batchNum*batchSize:(batchNum+1)*batchSize]

    currentX=[]

    currentY=[]

    currentIdx=[]

    counter=0

    for rev in temp:

        if len(rev)>=2:

            for wordIdx in range(len(rev)-1):

                currentX.append(word2idx[rev[wordIdx]])

                currentIdx.append(batchNum*batchSize+counter)

                currentY.append([word2idx[rev[wordIdx+1]]])

                currentX.append(word2idx[rev[wordIdx+1]])

                currentIdx.append(batchNum*batchSize+counter)

                currentY.append([word2idx[rev[wordIdx]]])

        counter+=1

    return np.array(currentX),np.array(currentIdx),np.array(currentY)
import math

import tensorflow as tf

batch_size = 128

embedding_size = 32

doc_embedding_size=5



train_inputs=tf.placeholder(tf.int32, shape=[None])

train_docs=tf.placeholder(tf.int32,shape=[None])

train_labels=tf.placeholder(tf.int32, shape=[None,1])



embeddings = tf.Variable(tf.random_uniform((2000, embedding_size), -1, 1))

embeddingDoc=tf.Variable(tf.random_uniform((2125056,doc_embedding_size),-1,1))



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

    for batchIndex in range(int(len(cleanMasterReview)/128)-1):

        trainX,trainIndex,trainY=makeBatch(batchIndex,128)

        loss,_ = sess.run([nce_loss,optimizer],feed_dict={train_inputs:trainX,train_docs:trainIndex,train_labels:trainY})

        tempLossTOT+=loss

        if batchIndex%5000==0:

            print('Current Loss: '+str(tempLossTOT/(batchIndex+1)*1.0 ))
from sklearn.manifold import TSNE

embeddingMat=sess.run(embeddings)

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
import plotly

import plotly_express as px

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