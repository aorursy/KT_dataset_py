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
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional

from tensorflow.keras.optimizers import Adam
data='There’s something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience I’ve in fact reached the opposite conclusion). Fast forward about a year: I’m training RNNs all the time and I’ve witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me. This post is about sharing some of that magic with you.'

corpus=data.lower().split('.')[:-1]  ##first split into lines

tokenizer=Tokenizer()

tokenizer.fit_on_texts(corpus)

word2idx=tokenizer.word_index

total_words=len(word2idx)+1

print(word2idx)
input_sequences=[]

for line in corpus:

    token_list=tokenizer.texts_to_sequences([line])[0]

    #print(token_list)

    for i in range(1,len(token_list)):

        ngram_sequence=token_list[:i+1]

       # print(ngram_sequence)

        input_sequences.append(ngram_sequence)
max_sequence_length=max([len(x) for x in input_sequences])

# create pad

input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_length,padding='pre'))

# create predictors and labels

xs,labels=input_sequences[:,:-1],input_sequences[:,-1]

print(xs,labels)

ys=tf.keras.utils.to_categorical(labels,num_classes=total_words)

print(ys)
model=Sequential()

model.add(Embedding(total_words,64,input_length=max_sequence_length-1))

model.add(Bidirectional(LSTM(20)))

model.add(Dense(total_words,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(xs,ys,epochs=500,verbose=1)
import matplotlib.pyplot as plt

def plot_graph(history,string):

    plt.plot(history.history[string])

    plt.xlabel('Epochs')

    plt.ylabel(string)

    plt.legend([string,'val_'+string])

    plt.show()

plot_graph(history,'accuracy')

plot_graph(history,'loss')
seed_text='Neural Networks'

next_words=20

for _ in range(next_words):

    token_list=tokenizer.texts_to_sequences([seed_text])[0]

    token_list=np.array(pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre'))

    predicted=model.predict_classes(token_list,verbose=0)

    for word,index in word2idx.items():

        if predicted==index:

            output_word=word

            break

    seed_text +=' '+ output_word

print(seed_text)
