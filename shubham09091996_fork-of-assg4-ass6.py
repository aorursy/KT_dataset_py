# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import bz2
import nltk
import chardet
# Any results you write to the current directory are saved as output.
trainfile=bz2.BZ2File("../input/train.ft.txt.bz2")
train_file_lines = trainfile.readlines()
train_file_lines[0]
train_file_lines = [x.decode('utf-8') for x in train_file_lines]
train_file_lines[0]
import re
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
                                                           
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
len(train_labels)
train_sentences[192]
import string
for i in range(len(train_sentences)):
    train_sentences[i] = train_sentences[i].translate(str.maketrans('','',string.punctuation))
train_sentences[192]
train = pd.DataFrame(data=list(zip(train_sentences, train_labels)), columns=['review_text', 'sentiment_class_label'])
train
train['word_count'] = [len(text.split()) for text in train.review_text]
train.head()
train = train[train.word_count < 20]
train.shape
train = train.drop(columns=['word_count'], axis=1)
train.head()
train.shape
train = train.set_index(np.arange(len(train)))
train.head()
y=np.asmatrix(train['sentiment_class_label'])
mp={}
for i in train.review_text:
    for j in i.split():
        if j in mp:
            mp[j]+=1
        else:
            mp[j]=1
list_of_words=[]
for key, value in mp.items():
    if value>5:
        list_of_words.append(key)
review=[]
for i in train.review_text:
    j =i.split()
    review.append([w for w in j if w in list_of_words])
            
train['review_text']
review
from gensim.models import word2vec
models=word2vec.Word2Vec(review,size=100)
models['man'].shape
X=np.zeros([len(review),20,100])
c1=0
c2=0
for k in review:
    c2=0
    for j in k:
        X[c1][c2]=models[j]
        c2+=1
    c1+=1
X.shape
y=train['sentiment_class_label'].values
y.shape
mp
list_of_words
len(list_of_words)
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(X,y,train_size=0.9)
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model1= Sequential()
model1.add(LSTM(100,input_shape=(20,100), activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
history = model1.fit(trainX, trainY, batch_size=32, epochs=5, verbose=1, 
                   validation_split=0.1)
model1.evaluate(testX,testY,batch_size=120)
import matplotlib.pyplot as plt
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
