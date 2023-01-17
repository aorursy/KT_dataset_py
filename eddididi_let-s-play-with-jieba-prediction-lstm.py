# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import jieba

import re



import numpy as np

import pandas as pd

import tensorflow as tf

import jieba.analyse as analyse



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
embeddings=pd.read_csv('/kaggle/input/customized-embeddings/FullEmbedding.csv')

rawReviews=pd.read_csv('/kaggle/input/doubanmovieshortcomments/DMSC.csv')
rawReviews.head()
embeddings.head()
comments=rawReviews['Comment'].tolist()

frequentWord=embeddings['keys'].tolist()

embeddings.drop(embeddings.columns[[0]], axis=1, inplace=True)

embeddings.head()
def processLine(singleReview):

    seg_list = jieba.cut(singleReview, cut_all=False, HMM=True)

    tempToken=[]

    tokenList=[]

    for i in seg_list:

        tempToken=re.findall(r'[\u4e00-\u9fff]+',i)

        if len(tempToken)==0:

            pass

        else:

            if tempToken[0] in frequentWord:

                tokenList.append(tempToken[0])

    return tokenList
masterReview=[processLine(x) for x in comments]

print('Total number of reviews: '+str(len(masterReview)))
score=rawReviews['Star'].tolist()
print("Total length of dataset: ",len(masterReview))
trainText=masterReview[:1800000]

testText=masterReview[1800000:]

trainScr=score[:1800000]

testScr=score[1800000:]

lookupdict={}

for wd in frequentWord:

    lookupdict[wd]=np.array(embeddings[embeddings['keys']==wd].iloc[:,0:32])[0]
def makeBatch(batchIndex,batchLength):



    tempList=trainText[batchIndex*batchLength:(batchIndex+1)*batchLength]

    train_X=[]

    temp=[]



    for rev in tempList: 

        if len(rev)>=20:

            temp=[lookupdict[tok] for tok in rev[-20:]]

        else: 

            temp=[np.zeros(32)]*(20-len(rev))+[lookupdict[tok] for tok in rev]

        train_X.append(temp)

    return np.array(train_X), np.array([[x] for x in trainScr[batchIndex*batchLength:(batchIndex+1)*batchLength]],dtype='float')

tf.reset_default_graph()

num_hidden = 128



data = tf.placeholder(tf.float32, [None,20,32]) 

target = tf.placeholder(tf.float32, [None,1])

## LSTM

cell = tf.nn.rnn_cell.LSTMCell(num_hidden,activation='tanh',initializer=tf.random_normal_initializer())

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])

last = tf.gather(val, int(val.get_shape()[0]) - 1)



## Dense 1

weight = tf.Variable(tf.truncated_normal([num_hidden, 16]))

bias = tf.Variable(tf.constant(0.1, shape=[16]))

prediction1=tf.nn.relu(tf.matmul(last, weight) + bias)

## Dense 2

weight1 = tf.Variable(tf.truncated_normal([16,1]))

bias1 = tf.Variable(tf.constant(0.1, shape=[1]))

prediction = tf.matmul(prediction1, weight1) + bias1
mse = tf.reduce_sum(tf.keras.losses.MSE(prediction,target))

optimizer = tf.train.AdamOptimizer(0.001).minimize(mse)
init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

batch_size=64

cnter=0

for i in range(10):

    ptr = 0

    for j in range(int(1800000/batch_size)-1):

        cnter+=1

        inp, out = makeBatch(j,64)

        _,pred=sess.run([optimizer,last],{data: inp, target: out})

        loss=sess.run(mse,{data: inp, target: out})

        if (cnter+1)%5000==0:

            print("Epoch - "+str(cnter)+": the loss is: " +str(loss))
def makeBatchTest(batchIndex,batchLength):



    tempList=testText[batchIndex*batchLength:(batchIndex+1)*batchLength]

    train_X=[]

    temp=[]



    for rev in tempList: 

        if len(rev)>=20:

            temp=[lookupdict[tok] for tok in rev[-20:]]

        else: 

            temp=[np.zeros(32)]*(20-len(rev))+[lookupdict[tok] for tok in rev]

        train_X.append(temp)

    return np.array(train_X), np.array([[x] for x in testScr[batchIndex*batchLength:(batchIndex+1)*batchLength]],dtype='float')

batch_size=64

masterPred=[]

masterActual=[]

for j in range(int((2125056-1800000)/batch_size)-1):

    cnter+=1

    inp, out = makeBatchTest(j,64)

    pred=sess.run(prediction,{data: inp})

    pred=[x[0] for x in pred]

    masterActual+=[x[0] for x in out]

    masterPred+=pred
y_pred=[x.round() for x in masterPred]
y_actual=masterActual
raw=pd.DataFrame()

raw['actual']=y_actual

raw['predicted']=y_pred

raw.to_csv('checkResult.csv',index=False)