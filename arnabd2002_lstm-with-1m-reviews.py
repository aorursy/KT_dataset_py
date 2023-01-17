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
import re
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dropout,Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# Any results you write to the current directory are saved as output.
trainFile='../input/train.ft.txt.bz2'
file=bz2.BZ2File(trainFile,'r')
lines=file.readlines()
print('Done!!!')
del docSentimentList,docDF
docSentimentList=[]
def getDocumentSentimentList(docs,splitStr='__label__'):
    for i in range(len(docs)):
        #print('Processing doc ',i,' of ',len(docs))
        text=str(lines[i])
        #print(text)
        splitText=text.split(splitStr)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        #print('First half:',secHalf[0],'\nsecond half:',secHalf[2:len(secHalf)-1])
        docSentimentList.append([text,sentiment])
    print('Done!!')
    return docSentimentList
docSentimentList=getDocumentSentimentList(lines[:1000000],splitStr='__label__')
docDF=pd.DataFrame(docSentimentList,columns=['TEXT','SENTIMENT'])
docDF['SENTIMENT'].value_counts()
X=docDF['TEXT']
y=docDF['SENTIMENT']
y=y.astype('int32')
lb=LabelBinarizer(pos_label=1,neg_label=0)
y_binarized=lb.fit_transform(y)
y=to_categorical(num_classes=2,y=y_binarized)
y.shape
tok=Tokenizer(num_words=100000,lower=True)
tok.fit_on_texts(X)
print('Toekizing...done')
seqs=tok.texts_to_sequences(X)
print('Sequencing...done')
padded_seqs=pad_sequences(seqs,maxlen=100)
print('Padding sequences...done')
padded_seqs.shape,y.shape
def createLSTM():
    model=Sequential()
    model.add(Embedding(1000000,100))
    model.add(LSTM(256,return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='sigmoid'))
    return model
model=createLSTM()
model.summary()
X_train,X_test,y_train,y_test=train_test_split(padded_seqs,y,train_size=0.80,test_size=0.20,random_state=43)
np.shape(X_train),np.shape(y_train)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
earlyStopping=EarlyStopping(patience=3,monitor='acc',min_delta=0.0001,verbose=1)
callbackslist=[earlyStopping]
model.fit(X_train,y_train,batch_size=1024,epochs=1,verbose=1,callbacks=callbackslist)
model.evaluate(X_test,y_test)
idx=np.random.randint(len(X))
test=[X[idx]]
print(test)
print('RESULT:')
pred=model.predict(pad_sequences(tok.texts_to_sequences(test),maxlen=100))
print(np.argmax(pred))
if np.argmax(pred)==0:
    print('NEG')
else:
    print('POS')