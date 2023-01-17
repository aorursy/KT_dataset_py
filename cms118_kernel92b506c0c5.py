# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from __future__ import print_function
from functools import reduce
import re
import sys
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from keras.models import model_from_json
import h5py
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = open('../input/train-data/train_LSTM_26112016.txt',"r")
print(data)
test = open('../input/testfile/test.txt',"r")
print(test)

for line in data:
    print (line)
def extract(data):
    dicto={}
    idx=-1
    l=[]
    for line in data:
        if(line[0]=="1"):
            idx=idx+1
            if(idx!=0):
                dicto[idx]=(l)
            l=[]
        l.append(line)
        
    return dicto
val = extract(data)
def extract_query(dicton):
    query=[]
    statement=[]
    for i in range(0,len(dicton)):
        query.append(dicton[i+1][-1])
        statement.append(dicton[i+1][:-1])
    return query,statement
query,statement = extract_query(val)

def clean_data(lst):
    state_list=[]
    for idx in range(0,len(lst)):
        state_list.append(' '.join(lst[idx]))
    return state_list
state_list = clean_data(statement)
print(state_list)

stop_words1=['1','2','3','4','5','6','7','8','9','0','.',',']
def processing(lst):
    filtered_sentence=[[]]
    for i in range(0,len(lst)):
        word_tokens = word_tokenize(lst[i])
        filtered_sentence .append([w for w in word_tokens if not w in stop_words1])
    filtered_sentence.remove([])
    return filtered_sentence

filtered_sentence = processing(state_list)
print(filtered_sentence)
        
        

query_sentence = processing(query)
print(query_sentence)
stop_words2 = ['num1','num2','num3','+','-','*','/','?'] ## to remove num1....

def extract_operator(query_sentence):
    operator=[]
    for idx in range(0,len(query_sentence)):
        operator.append(query_sentence[idx][-1])
        query_sentence[idx] = [w for  w in query_sentence[idx] if w not in stop_words2 ]
        
    return operator,query_sentence

operators,queries = extract_operator(query_sentence)
print(operators)
print(queries)   
# encoded target variable
dic2={'+':1,'-':2,'*':3,'/':4}
def fun1(s):
    return dic2[s]

operator = [fun1(ops) for ops in operators]
print(operator)
stemmer = PorterStemmer()

def list_to_string(lst):
    lst1 = []
    for i in range(0,len(lst)):
        singles = [stemmer.stem(plural) for plural in lst[i]]
        lst1.append(' '.join(singles))
    return lst1
        
query_sent = list_to_string(queries)
word_sent = list_to_string(filtered_sentence)
print(word_sent)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size, split=' ')
tokenizer.fit_on_texts(word_sent+query_sent)
word_numeric = tokenizer.texts_to_sequences(word_sent)
word_numeric=  pad_sequences(word_numeric)
query_numeric = tokenizer.texts_to_sequences(query_sent)
query_numeric=  pad_sequences(query_numeric)
print(query_numeric[0])
max_len_word=len(word_numeric[100]) # this is numerical mapping of word sentence and vacabulary word
max_len_query=len(query_numeric[100])

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,input_length=max_len_word))
sentrnn.add(Dropout(0.5))

qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,input_length=max_len_query))
qrnn.add(Dropout(0.5))
qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
qrnn.add(RepeatVector(max_len_word))

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)

print('Build model...')
sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                      input_length=max_len_word))
sentrnn.add(Dropout(0.3))
sentrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
sentrnn.add(RepeatVector(max_len_word))
qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                   input_length=max_len_query))
qrnn.add(Dropout(0.3))
qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
qrnn.add(RepeatVector(max_len_word))

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='sum'))
model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(1, activation='softmax')) # as we have only 4 operators





model.fit([word_numeric,query_numeric], operator, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
val = (word_numeric[0])

q = query_numeric[0]
print(model.summary())
loss, acc = model.evaluate([word_numeric, query_numeric], operator, batch_size=BATCH_SIZE)
loss
acc
l=(model.predict([word_numeric[:], query_numeric[:]]))
m=[]
for i in range(0,len(l)):
    m.append(np.argmax(l[i]))
print (m)

def fun(l1,l2):
    print(len(l1))
    print(type(l2))
    count = 0
    for i in range(0,len(l1)):
        if(l1[i]==l2[i]):
            count = count+1
    return count,len(l1)

a,b = fun(m,operator)
print(a/b)
print(b)

def convert_to_d(test):
    val_test = extract(test)
    query_test,statement_test = extract_query(val_test)
    state_list_test = clean_data(statement_test)
    filtered_sentence_test = processing(state_list_test)
    query_sentence_test = processing(query_test)
    operators_test,queries_test = extract_operator(query_sentence_test)
    operator_test = [fun1(ops) for ops in operators_test]
    query_sent_test = list_to_string(queries_test)
    word_sent_test = list_to_string(filtered_sentence_test)
    word_numeric_test = tokenizer.texts_to_sequences(word_sent_test)
    word_numeric_test=  pad_sequences(word_numeric_test,max_len_word)
    query_numeric_test = tokenizer.texts_to_sequences(query_sent_test)
    query_numeric_test=  pad_sequences(query_numeric_test,max_len_query)
    return word_numeric_test,query_numeric_test,operator_test
print (test)
word_numeric_test,query_numeric_test,operator_test = convert_to_d(test)
print(word_numeric_test)
b=(model.predict([word_numeric_test[:], query_numeric_test[:]]))
print(b)


    
for line in test:
    print(line)
# for test set

m1=[]
for i in range(0,len(b)):
    m1.append(np.argmax(b[i]))
print (m1)
a1,b1 = fun(m1,operator_test)
print(a1/b1)
print(b1)
model.save('../input/dilton/saved')
