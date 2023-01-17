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
from scipy import *

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

data='There’s something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I’ve in fact reached the opposite conclusion). Fast forward about a year: I’m training RNNs all the time and I’ve witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you.'

data=data.lower()

tokenizer=Tokenizer()

tokenizer.fit_on_texts([data])

sequences=tf.keras.preprocessing.text.text_to_word_sequence(data)
encoded=tokenizer.texts_to_sequences([data])[0]

word2idx=tokenizer.word_index

idx2word=tokenizer.index_word

print(idx2word)

#print(word2idx)

vocab_size=len(word2idx)

print('data has %d characters, %d unique words,%d total words' % (len(data), vocab_size,len(sequences)) )

print(encoded)
hidden_size=100  #size of hidden layer of neurons

seq_length=6 # number of steps to unroll the RNN for 

learning_rate=1e-1 # learning rate for backprop
Wxh=np.random.randn(hidden_size,vocab_size)*0.01  #from input to hidden,the second dimension vocab_size coud be embedding_size in general

Whh=np.random.randn(hidden_size,hidden_size)*0.01 #from hidden to hidden 

Why=np.random.randn(vocab_size,hidden_size)*0.01 #from hidden to output

bh=np.zeros((hidden_size,1)) #hidden bias

by=np.zeros((vocab_size,1))  #output bias

def sample(h,seed_ix,n):

    """

    sample a sequence of integers from the model

    h is initial memory sate in hidden layer,seed_ix is the seed letter for first time step

    n is the length of sequence generated

    returned:generated indexs of sequence

    """

    x=np.zeros((vocab_size,1)) 

    x[seed_ix]=1  ##note that the input word vector is one-hot encoded throughout the model

    ixes=[]

    for t in range(n):

        h=np.tanh(np.dot(Whh,h)+np.dot(Wxh,x)+bh)

        y=np.dot(Why,h)+by

        p=np.exp(y)/np.sum(np.exp(y))

        ix=np.random.choice(range(vocab_size),p=p.ravel()) #generate next word by sampling according to p distribution

        ixes.append(ix)

        x=np.zeros((vocab_size,1)) 

        x[ix]=1

    return ixes

    
def lossFun(inputs,targets,hprev):

    """

    inputs and targets are list of integers.

    hprev is the initial hidden state

    returns the loss,gradients on model parameters and last hidden state

    """

    xs,hs,ys,ps={},{},{},{}

    hs[-1]=np.copy(hprev)

    loss=0.0

    #forward pass

    for t in range(len(inputs)):

        xs[t]=np.zeros((vocab_size,1))

        xs[t][inputs[t]]=1

        hs[t]=np.tanh(np.dot(Whh,hs[t-1])+np.dot(Wxh,xs[t])+bh) #hidden state

        ys[t]=np.dot(Why,hs[t])+by # unnormalized log probabilities for next word

        ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t]))  # probabilities for next word

        loss += -np.log (ps[t][targets[t],0]) #softmax(cross entropy) loss

    #backward pass:compute gradients going backwards

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):

        dp = np.copy(ps[t])

        dp[targets[t]] -= 1.

        dWhy += np.dot(dp,hs[t].T)

        dby += dp

        dh=np.dot(Why.T,dp)+dhnext

        dhraw=(1-hs[t]*hs[t])*dh

        dhnext=np.dot(Whh.T,dhraw)

        dbh+=dhraw

        dWhh+=np.dot(dhraw,hs[t-1].T)

        dWxh+=np.dot(dhraw,xs[t].T)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:

        np.clip(dparam,-5,5,out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

        





        

    
Nit=5e5

print_freq=2000

n,p=0,0

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)

mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad

for i in range(int(Nit)):

    # prepare inputs (we're sweeping from left to right in steps seq_length long)

    # note that seq_length acts as the function of batch_size 

    if p+seq_length+1>=len(sequences) or n==0: #reset hidden state if pointer of the final word in seq exceed the end of the data

        hprev=np.zeros((hidden_size,1)) #reset RNN memory 

        p=0 # go from state of data 

    inputs=[word2idx[word] for word in sequences[p:p+seq_length]]

    targets=[word2idx[word] for word in sequences[p+1:p+seq_length+1]]

    #sample from model now and then

    if n % print_freq == 0:

       sample_ix=sample(hprev,inputs[0],39)

       txt = ' '.join(idx2word[ix+1] for ix in sample_ix)

       print ('----\n %s \n----' % (txt, ))

    #forward seq_length words through the net and fetch gradient

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

    if n==0:

      smooth_loss = loss

    else:

      smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % print_freq == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress



    # perform parameter update with Adagrad

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],[dWxh, dWhh, dWhy, dbh, dby],[mWxh, mWhh, mWhy, mbh, mby]):

        mem += dparam * dparam

        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update





    

        

    n += 1 #iteration counter

    p += seq_length #move data pointer