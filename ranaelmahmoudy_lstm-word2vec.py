# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import re

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import stopwords
import gensim
from gensim.models import word2vec
from gensim import corpora, models ,similarities
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
from tensorflow.keras import layers



data=pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")




def get_data_ready(data):
    
    statusNum={'positive': 1 ,'neutral': 2 , 'negative': 0}

    data['status']=data['airline_sentiment'].map(statusNum)

   

def get_class_counts(df):
    c=df.groupby(data['airline_sentiment'])['tweet_id'].nunique()
    return {key:c[key] for key in list(c.keys())}
    


#convert text into words sequence
def text2words(text):
    #remove ay 7aga msh letter
    text=re.sub("[^a-zA-Z]"," ",text)
    words=text.lower().split()
    return (words)

def text2sentences(text):
    tokenized_text=tokenizer.tokenize(text.strip())
    sentences=[] #hyb2a 3ndy list of lists list of goml kol gmla gwaha list of words
    for token in tokenized_text:
        if(len(token)!=0):
            sentences.append(text2words(token))
    return sentences    

get_data_ready(data)

#train.dropna()
#test.dropna()
#test.head()
#train_prob=get_class_counts(train)
#test_prob=get_class_counts(test)
#print("train",train_prob)
#print("test",test_prob)
sentences=[]
for text in data['text']:
    sentences+=text2sentences(text)
#print(sentences)    
sizeof=300
#################### building model ####################
#size equal no.of features
#sg->1 (skip diagram)
#workers =1 3shn 22dr 23ml reproducible run , to avoid scheduling threads w irritated ordering
model=word2vec.Word2Vec(sentences,workers=1,size=sizeof,min_count=40,window=5,sample=1e-3,sg=1,seed=50,alpha=0.05)
# To make the model memory efficient
model.init_sims(replace=True)
#model.wv.most_similar("bad")

###feature vector for words 

def checkIfExists(words,model):
    flag=-1
    index2word_set = set(model.wv.index2word)
    for word in  words:
        if index2word_set.__contains__(word):
            flag=1
            return flag
    return flag    

def average_words(words,model,num_features):
    #zero initialized array asra3
    featureVec = np.zeros(num_features,dtype="float32")
    nofwords=0
    atleastone=0
    index2word_set = set(model.wv.index2word)
   
    for word in  words:
        if index2word_set.__contains__(word):
            atleastone=1
            nofwords = nofwords + 1
            featureVec = np.add(featureVec,model[word])
    if(atleastone==1):
        
        featureVec = np.divide(featureVec, nofwords) 
        return featureVec,atleastone
    else:
        print("canot predict")
        return
    #print(featureVec)
    
def sum_words(words,model,num_features):
    #zero initialized array asra3
    featureVec = np.zeros(num_features,dtype="float32")
    nofwords=0
    atleastone=0
    index2word_set = set(model.wv.index2word)
   
    for word in  words:
        if index2word_set.__contains__(word):
            atleastone=1
            
            featureVec = np.add(featureVec,model[word])
    if(atleastone==1):
        
         
        return featureVec,atleastone
    else:
        print("canot predict")
        return
    #print(featureVec)

def getTrainDataReady(name_of_func):
    cleanTextToWords=[]
    for text in data['text']:
        cleanTextToWords.append(text2words(text))
    flag=0
    count=0
    AvgFeatureVecs = np.zeros((len(cleanTextToWords),sizeof),dtype="float32")

    for text in cleanTextToWords:
        AvgFeatureVecs[count],flag=name_of_func(text,model,sizeof)
        count=count+1
    return AvgFeatureVecs


def LSTM(model, name_of_func):
    X = getTrainDataReady( name_of_func)
    y = np.array(data['status'], dtype=int)
    y = to_categorical(y, num_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    lmodel = tf.keras.Sequential()
    lmodel.add(layers.LSTM(128,input_shape=(1, 300)))
    lmodel.add(layers.Dense(3, activation=tf.nn.softmax))
    lmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    lmodel.fit(X_train,y_train, batch_size=64, epochs=10)
    scores = lmodel.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    return lmodel
sum_model = LSTM(model,average_words)
average_model = LSTM(model, sum_words)