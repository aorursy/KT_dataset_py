# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import bz2
import os
print(os.listdir("../input"))
trainfile = bz2.BZ2File('../input/train.ft.txt.bz2','r')
lines = trainfile.readlines()
sent_analysis = []
def sent_list(docs,splitStr='__label__'):
    for i in range(1,len(docs)):
        text=str(lines[i])
        splitText=text.split(splitStr)
        #print(i)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        sent_analysis.append([text,sentiment])
    return sent_analysis

sentiment_list=sent_list(lines[:1000000],splitStr='__label__')
train_df = pd.DataFrame(sentiment_list,columns=['Text','Sentiment'])
train_df.head()
train_df['Sentiment'][train_df['Sentiment']=='1'] = 0
train_df['Sentiment'][train_df['Sentiment']=='2'] = 1
import matplotlib.pyplot as plt
import seaborn as sns
ax = plt.axes()
sns.countplot(train_df.Sentiment,ax=ax)
ax.set_title('Sentiment Distribution')
plt.show()
print("Proportion of positive review:", len(train_df[train_df.Sentiment==1])/len(train_df))
print("Proportion of negative review:",len(train_df[train_df.Sentiment==0])/len(train_df))
train_df['word_count'] = train_df['Text'].str.lower().str.split().apply(len)
train_df.head()
reviews = train_df.Text.values
import string 
def punctuation_remove(s):
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)
train_df['Text'] = train_df['Text'].apply(punctuation_remove)
train_df.head()
train_df1 = train_df[:][train_df['word_count']<=25]
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
c_vector.fit(train_df1['Text'].values)
word_list = list(c_vector.vocabulary_.keys())
stop_words = list(c_vector.stop_words) 
def remove_words(raw_sen,stop_words):
    sen = [w for w in raw_sen if w not in stop_words]
    return sen
def reviewEdit(raw_sen_list,stop_words):
    sen_list = []
    for i in range(len(raw_sen_list)):
        raw_sen = raw_sen_list[i].split()
        sen_list.append(remove_words(raw_sen,stop_words))
    return sen_list
sen_list = reviewEdit(list(train_df1['Text']),stop_words)

from gensim.models import word2vec
wv_model = word2vec.Word2Vec(sen_list,size=50)
wv_model.wv.syn0.shape

def AvgSen(sen_list,wv_model):
    word_set = set(wv_model.wv.index2word)
    X_avg = np.zeros([len(sen_list),50])
    c=0
    for sen in sen_list:
        temp = np.zeros([50,])
        nw=0
        for w in sen:
            if w in word_set:
                nw=nw+1
                temp = temp + wv_model[w]
        X_avg[c] = temp/nw
        c=c+1
    return X_avg
X_avg = AvgSen(sen_list,wv_model)
X_avg
def fun(sen_list,wv_model):
    word_set = set(wv_model.wv.index2word)
    X = np.zeros([len(sen_list),25,50])
    c = 0
    for sen in sen_list:
        nw=0
        for w in sen:
            if w in word_set:
                X[c,nw] = wv_model[w]
                nw=nw+1
        c=c+1
    return X
X = fun(sen_list,wv_model)

from sklearn.model_selection import train_test_split
y = train_df1['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LSTM

model1=  Sequential()
model1.add(LSTM(100,input_shape=(25,50),activation='relu'))
model1.add(Dense(50,activation='relu'))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model1.fit(X_train,y_train,validation_split=0.1,epochs=6,batch_size=64,verbose=1)
model1.evaluate(X_test, y_test, batch_size=128)

model1.evaluate(X_train, y_train, batch_size=128)

import matplotlib.pyplot as plt
loss_curve = hist.history['loss']
epoch_c = list(range(len(loss_curve)))

plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.plot(epoch_c,loss_curve,label='1 Hidden layer')
plt.show()
acc_curve = hist.history['acc']
epoch_c = list(range(len(loss_curve)))

plt.xlabel('Epochs')
plt.ylabel('Accuracy value')
plt.plot(epoch_c,acc_curve,label='1 Hidden layer')
plt.show()
