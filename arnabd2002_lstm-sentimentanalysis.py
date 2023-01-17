# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.manifold import TSNE
from gensim.models import word2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
# Any results you write to the current directory are saved as output.
filesList=os.listdir('../input/sentiment labelled sentences/sentiment labelled sentences')
os.listdir('../input/sentiment labelled sentences/sentiment labelled sentences')
imdb_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt'
amazon_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt'
yelp_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt'
def getReviewSentimentFromFile(file):
    fr=open(file)
    lines=fr.readlines()
    fr.close()
    reviewsentimentList=[]
    for l in lines:
        x=l.split('\t')
        reviewsentimentList.append([str.lstrip(str.rstrip(x[0])),str.lstrip(str.rstrip(x[1]))])
    return reviewsentimentList
rsList=getReviewSentimentFromFile(imdb_labelFile)+getReviewSentimentFromFile(amazon_labelFile)+getReviewSentimentFromFile(yelp_labelFile)
len(rsList[:])
rsDF=pd.DataFrame(rsList,columns=['REVIEW','SENTIMENT'])
rsDF.head(5)
X=rsDF['REVIEW']
y=rsDF['SENTIMENT']
y=to_categorical(num_classes=2,y=y)
np.shape(y)
tok=Tokenizer(lower=True,num_words=10000)
tok.fit_on_texts(X)
seqs=tok.texts_to_sequences(X)
padded_seqs=pad_sequences(seqs,maxlen=100)
def createLSTM():
    model=Sequential()
    model.add(Embedding(10000,100))
    model.add(LSTM(256))
    model.add(Dense(100,activation='sigmoid'))
    model.add(Dense(2,activation='sigmoid'))
    return model
model=createLSTM()
model.summary()
del X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=train_test_split(padded_seqs,y,train_size=0.85,test_size=0.15,random_state=43)
np.shape(X_train),np.shape(y_train)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(X_train,y_train,batch_size=32,epochs=5,verbose=1)
model.evaluate(X_test,y_test)[0]*100
idx=np.random.randint(len(rsDF['REVIEW']))
print(rsDF['REVIEW'].iloc[idx],'Sentiment:',rsDF['SENTIMENT'].iloc[idx])
test=[rsDF['REVIEW'].iloc[idx]]
test_seq=pad_sequences(tok.texts_to_sequences(test),maxlen=100)
pred=model.predict(test_seq)
proba=model.predict_proba(test_seq)
if np.argmax(pred)==0:
    print('NEG',proba[0][0]*100)
else:
    print('POS',proba[0][1]*100)