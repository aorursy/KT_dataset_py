# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/trainingandtestdata/"))

# Any results you write to the current directory are saved as output.

from sklearn.utils import shuffle

dataset_path = "../input/trainingandtestdata/training.1600000.processed.noemoticon.csv"



def demap(x):

    if x==0:return 0

    else:return 1



df = pd.read_csv(dataset_path,encoding="latin-1")

df=shuffle(df)

df['0']=df['0'].apply(lambda x: demap(x))

df=df.values

x = df[0:,5]

y = df[0:,0]

size = len(x)

print(x[0:5],y[0:5])

print("Length:",size)

print(df)
import re

import string 

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



stopwordList = stopwords.words('english')

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = text.strip()

    text = text.lower()

    text = re.sub(r'@[a-zA-Z0-9]+',' ',text) #mentions

    text = re.sub(r'[^a-zA-Z]',' ',text) # remove digits & other chars

    text = re.sub('https?://[a-zA-Z0-9./]+',' ',text) #links

    try:

        text = text.decode("utf-8-sig").replace(u"\ufffd", "<UKN>")   

    except:

        #nothing

        pass

    text = text.split()

    text = " ".join([lemmatizer.lemmatize(x) for x in text if x not in stopwordList])

    return text



x = [clean_text(sample) for sample in x]

print(x[0:5])

import pickle

num_words = 70

BOW = set()



for line in x:

    words = line.split()

    for word in words:

        BOW.add(word)

BOW = list(BOW)

BOW_len = len(BOW)



word_index = { BOW[ptr-1]:ptr for ptr in range(1,len(BOW)+1) }  

word_index["<PAD>"] = 0

reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])

del BOW

newX = []



for line in x:

    t=[]

    words = line.split()

    for word in words:

        t.append(word_index[word])

    if len(t) < num_words:

        t+= [0]*(num_words-len(t))

    newX.append(t)

newX = np.array(newX)

print(newX)

print(newX.shape)



filePath = "word_index.pkl"

fileout = open(filePath,'wb')

pickle.dump(word_index,fileout)

fileout.close()
from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer

x=newX



SPLIT = 80

limit = len(x)*SPLIT//100



xtrain = x[:limit]

ytrain = y[:limit]

xtest = x[limit:]

ytest = y[limit:]



train_len = len(xtrain)

test_len = len(xtest)

print(np.array(x))

print(y)



print(xtrain.shape)



from keras.layers import Dropout

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D,MaxPooling1D

from keras.layers.embeddings import Embedding

model = Sequential()

model.add(Embedding(BOW_len,128,input_length=num_words))

model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling1D(2))

model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling1D(2))

model.add(LSTM(128))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

h = model.fit(xtrain,ytrain,epochs=20,batch_size=128,validation_data=(xtest,ytest),verbose=1)

model.save('model.h5')
plt.plot(h.history['acc'])

plt.plot(h.history['val_acc'])

plt.title('Model accuracy')

plt.show()



plt.plot(h.history['loss'])

plt.plot(h.history['val_loss'])

plt.title('Model Loss')

plt.show()
word_index = pickle.load(open('word_index.pkl','rb'))



def conv2Test(text):

    text=clean_text(text)

    text=text.split()

    text=[word_index[x] for x in text]

    text+=[0]*(num_words-len(text))

    return np.array(text)



def result(prob):

    if prob > 0.5:print("Result: Positive",prob)

    else: print("Result: Negative",prob)

samples = ["this is good watch","hi there","how are you?","okay","bad weather"]

for text in samples:

    print("-"*100)

    sample = conv2Test(text).reshape((1,num_words))

    a=model.predict(sample)[0]

    result(a)
