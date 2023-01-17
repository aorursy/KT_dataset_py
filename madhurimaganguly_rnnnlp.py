# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bz2
import random
import re
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/glove6b300dtxt"))

# Any results you write to the current directory are saved as output.
bzfile = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2','r')
lines = bzfile.readlines()
lines[0]
# Make everything Lower Case
lines = [x.decode('utf-8') for x in lines]
labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in lines]
sentences = [x.split(' ', 1)[1][:-1].lower() for x in lines]
for i in range(len(sentences)):
    sentences[i] = re.sub('\d','0',sentences[i])
sentences[0]


words_list = []
y = []
tokenizer = RegexpTokenizer(r'\w+')
#tokenizer.tokenize(sentences[0])
for i in range(len(sentences)):
    if(len(tokenizer.tokenize(sentences[i]))<25):
        words_list.append(tokenizer.tokenize(sentences[i]))
        y.append(labels[i])
words_collection=[]
for x in words_list:
    for z in x:
        words_collection.append(z)
temp_vocab = nltk.FreqDist(words_collection)
vocab=[]
vocab.append('<PAD>')
vocab.append('_UNK')
vocab2id={}
vocab2id['<PAD>']=0
vocab2id['_UNK']=1
indexnum=2
for x in temp_vocab:
    if temp_vocab[x]>5:
        vocab.append(x)
        vocab2id[x] = indexnum
        indexnum = indexnum+1
len(vocab)
#vocab.sort()
id2vocab = dict(enumerate(vocab))
EMBEDDING_FILE = "../input/glove6b300dtxt/glove.6B.300d.txt"
EMBEDDING_DIM = 300

embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype="float32")
  embeddings_index[word] = coefs
embedding_matrix = np.zeros((len(vocab2id), EMBEDDING_DIM))
for i in range(len(vocab)):
    if id2vocab[i] in embeddings_index:
       embedding_matrix[i] = embeddings_index[id2vocab[i]]
    else:
      arr = np.random.uniform(-0.5,0.5,(1,300))
      embedding_matrix[i] = arr
from keras.layers import *
from keras.models import *


# define model
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(len(vocab), embedding_vecor_length,weights=[embedding_matrix], input_length=25))
model.add(LSTM(300,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
X = np.zeros((len(words_list), 25))
y = np.asarray(y)
for i in range(len(words_list)):
    j=0
    for z in words_list[i]:
        if z in vocab2id:
          X[i,j]= vocab2id[z]
        else :
          X[i,j]= 1
        j=j+1
len(words_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model_hist= model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
acc = model_hist.history['acc']
val_acc = model_hist.history['val_acc']
loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 6));
plt.subplot(1,2,1)
#plt.plot(epochs, acc, color='#0984e3',marker='o',linestyle='none',label='Training Accuracy')
plt.plot(epochs, acc, color='#eb4d4b',label='Training Accuracy')
plt.plot(epochs, val_acc, color='#0984e3',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
#plt.plot(epochs, loss, color='#eb4d4b', marker='o',linestyle='none',label='Training Loss')
plt.plot(epochs, loss, color='#eb4d4b',label='Training Loss')
plt.plot(epochs, val_loss, color='#0984e3',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')