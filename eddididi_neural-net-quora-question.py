# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import nltk

import re

import os

import collections

import operator



import sklearn as sk

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords 



from sklearn.decomposition import PCA

pd.options.display.max_columns = 10

print(os.listdir("../input"))

print(os.listdir("../"))

stop_words=set(stopwords.words('english'))



# Any results you write to the current directory are saved as output.
trainMaster=pd.read_csv('../input/quora-question-pairs/train.csv')

print(trainMaster.head())

print(trainMaster.shape)
filename='../input/embedding/glove.6B.50d.txt'

def loadGloVe(filename):

    vocab=[]

    embed=[]

    file=open(filename,'r',encoding='utf8')

    for lin in file.readlines():

        try:

            row=lin.strip().split(' ')

            vocab.append(row[0])

            embed.append(row[1:])

        except:

            pass

    file.close()

    return vocab,embed

vocab,embed=loadGloVe(filename)

embeddnig_dim=len(embed[0])

embedding=np.asarray(embed)
## Make word2index and index2word

word2idx={}

idx2word={}

for val,idx in zip(vocab,range(len(vocab))):

    word2idx[val]=idx

    idx2word[idx]=val

pca = PCA(n_components=10)

embedding=pca.fit_transform(embedding)
def digitalizeSent(string,pdSize):

    tempIdx=2

    pdIdx=2

    masterSent=[]

    string=str(string).lower()

    loTxt=string.split()

    counter=0

    for wd in loTxt:

        if wd in vocab and counter<pdSize:

            tempIdx=word2idx[wd]

            masterSent.append(tempIdx)

            counter+=1

        elif counter>=pdSize:

            break

        

    

    for fill in range(counter,pdSize):

        masterSent.append(pdIdx)

    return np.array(masterSent)
def makeBatch(batchSize,batchNumber,paddingSize=70):

    questionEmbeddingBatch1=[]

    questionEmbeddingBatch2=[]

    labelBatch=[]

    for sent in trainMaster['question1'].tolist()[batchNumber*batchSize:batchNumber*batchSize+batchSize]:

        questionEmbeddingBatch1.append(digitalizeSent(sent,paddingSize))



    for sent in trainMaster['question2'].tolist()[batchNumber*batchSize:batchNumber*batchSize+batchSize]:

        questionEmbeddingBatch2.append(digitalizeSent(sent,paddingSize))

        

    return np.array(questionEmbeddingBatch1),np.array(questionEmbeddingBatch2),np.array(trainMaster['is_duplicate'].tolist()[batchNumber*batchSize:batchNumber*batchSize+batchSize])

import tensorflow as tf



from tensorflow.python.framework import ops

ops.reset_default_graph()

sess=tf.Session()

def snn(address1, address2, dropout_keep_prob,num_features, input_length):

    def siamese_nn(input_vector, num_hidden):

        cell_unit = tf.contrib.rnn.BasicLSTMCell # OR tf.nn.rnn_cell.BasicLSTMCell

        lstm_forward_cell = cell_unit(num_hidden, forget_bias=1.0)

        lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)

        lstm_backward_cell = cell_unit(num_hidden, forget_bias=1.0)

        lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=dropout_keep_prob)

    

        # Split title into a character sequence to accommodate the TF requirment

        input_embed_split = tf.split(axis=1, num_or_size_splits=input_length, value=input_vector)

        input_embed_split = [tf.squeeze(x, axis=[1]) for x in input_embed_split]



        try:

            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,lstm_backward_cell,

                                                                    input_embed_split,dtype=tf.float32)

        except Exception:

            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,lstm_backward_cell,

                                                              input_embed_split,dtype=tf.float32)

        temporal_mean = tf.add_n(outputs) / input_length

        output_size = 10

        A = tf.get_variable(name="A", shape=[2*num_hidden, output_size],dtype=tf.float32,

                            initializer=tf.random_normal_initializer(stddev=0.1))

        b = tf.get_variable(name="b", shape=[output_size], dtype=tf.float32,

                            initializer=tf.random_normal_initializer(stddev=0.1))

        

        final_output = tf.matmul(temporal_mean, A) + b

        final_output = tf.nn.dropout(final_output, dropout_keep_prob)

        return(final_output)

        

    output1 = siamese_nn(address1, num_features)



    with tf.variable_scope(tf.get_variable_scope(), reuse=True):

        output2 = siamese_nn(address2, num_features)



    output1 = tf.nn.l2_normalize(output1, 1)

    output2 = tf.nn.l2_normalize(output2, 1)

    dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)

    

    return dot_prod





def get_predictions(scores):

    predictions = tf.sign(scores, name="predictions")

    return predictions





def loss(scores, y_target, margin):

    pos_loss_term = 0.25 * tf.square(tf.subtract(1., scores))

    pos_mult = tf.add(tf.multiply(0.5, tf.cast(y_target, tf.float32)), 0.5)

    pos_mult = tf.cast(y_target, tf.float32)



    positive_loss = tf.multiply(pos_mult, pos_loss_term)

    neg_mult = tf.add(tf.multiply(-0.5, tf.cast(y_target, tf.float32)), 0.5)

    neg_mult = tf.subtract(1., tf.cast(y_target, tf.float32))

    

    negative_loss = neg_mult*tf.square(scores)

    loss = tf.add(positive_loss, negative_loss)

    target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)

    less_than_margin = tf.less(scores, margin)

    both_logical = tf.logical_and(target_zero, less_than_margin)

    both_logical = tf.cast(both_logical, tf.float32)

    multiplicative_factor = tf.cast(1. - both_logical, tf.float32)

    total_loss = tf.multiply(loss, multiplicative_factor)

    avg_loss = tf.reduce_mean(total_loss)

    return avg_loss
address1_ph = tf.placeholder(tf.int32, [None, 70], name="q1_ph")

address2_ph = tf.placeholder(tf.int32, [None, 70], name="q2_ph")

y_target_ph = tf.placeholder(tf.int32, [None], name="y_target_ph")

dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

address1_embed = tf.nn.embedding_lookup(embedding, address1_ph)

address1_embed=tf.cast(address1_embed,tf.float32)

address2_embed = tf.nn.embedding_lookup(embedding, address2_ph)

address2_embed=tf.cast(address2_embed,tf.float32)
num_features=128

text_snn = snn(address1_embed, address2_embed,dropout_keep_prob_ph, num_features, 70)
batch_loss = loss(text_snn, y_target_ph, 0.5)

optimizer = tf.train.AdamOptimizer(0.01)

train_op = optimizer.minimize(batch_loss)

init = tf.global_variables_initializer()

sess.run(init)
train_loss_vec = []

train_acc_vec = []

dropout_keep_prob=0.75

for b in range(2000): ## Modify this on your local machine

    address1,address2,target_similarity=makeBatch(100,b,paddingSize=70)    

    train_feed_dict = {address1_ph: address1,address2_ph: address2,

                       y_target_ph: target_similarity, dropout_keep_prob_ph: dropout_keep_prob}

    _, train_loss = sess.run([train_op, batch_loss],feed_dict=train_feed_dict)

    train_loss_vec.append(train_loss)

    if b%10==0:

        print('Training Metrics, Batch {0}: Loss={1:.3f}.'.format(b, train_loss))