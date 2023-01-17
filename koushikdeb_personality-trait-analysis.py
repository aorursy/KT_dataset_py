# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import collections

import decimal as dc

from nltk.stem import PorterStemmer 

import statistics as st

# define sequences

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import re

import os

import re

import nltk

import keras

#import math

#nltk.download('stopwords')

from nltk.corpus import stopwords

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
files=[dir for dir in os.walk('../input/transcripts')]

vlogs=os.listdir("../input/transcript-data/transcripts/transcripts")

#print(vlogs)
def fetch_text(num):

    with open("../input/transcript-data/transcripts/transcripts/"+vlogs[num])as f:

        data=f.read().replace('\t','').replace('\n','')

        ind=re.findall(r'\d+',f.name)

        if len(ind)==1:

            return data,int(ind[0])

        else:

            print("Error:Multiple numbers are appeared as File Index")

      #function to do basic data cleaning takes a single argument Return Data                                       

    

    

#data,index=fetch_text(2)

#print(index)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

FLOAT=re.compile('[-+]?\d*\.\d+|\d+')

ps = PorterStemmer() 

def clean_text(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = FLOAT.sub('', text)

    text = text.lower() # lowercase text

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.

    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 

    

    text = text.replace('x', '')

#    text = re.sub(r'\W+', '', text)

    text = ' '.join(ps.stem(word) for word in text.split() if word not in STOPWORDS) # remove stopwors from text

    return text
#Speech_Data = np.array([])

#indList=[]

sp_data={}

for seq in range(0,len(vlogs)):

    data,index=fetch_text(seq)

    data=clean_text(data)

    #print(data)

    #indList.append(index)

    #Speech_Data = np.append(Speech_Data, data)

    sp_data[index]=data

sorted_data = collections.OrderedDict(sorted(sp_data.items()))

data_val=list(sorted_data.values()) # Preparing input data for train and test

#print(data_val[:3])

def Conv2Bin(listOflst):

    lst1=[]

    lst1.append([lst[0] for lst in listOflst])

    avg1=float(dc.Decimal("%.3f"%np.mean(lst1)))

    lst1=[]

    lst1.append([lst[1] for lst in listOflst])

    avg2=float(dc.Decimal("%.3f"%np.mean(lst1)))

    lst1=[]

    lst1.append([lst[2] for lst in listOflst])

    avg3=float(dc.Decimal("%.3f"%np.mean(lst1)))

    lst1=[]

    lst1.append([lst[3] for lst in listOflst])

    avg4=float(dc.Decimal("%.3f"%np.mean(lst1)))

    #print(avg4)

    bin_Score=[]

    for lst in listOflst:

        bn_lst=[]

        if lst[0]>avg1:

            bn_lst.append(1)

        else:

            bn_lst.append(0)

            

        if lst[1]>10:

        #if lst[1]>avg2:

            bn_lst.append(1)

        else:

            bn_lst.append(0)



        if lst[2]>10:

        #if lst[1]>avg3:

            bn_lst.append(1)

        else:

            bn_lst.append(0)



        if lst[3]>10:

        #if lst[1]>avg4:

            bn_lst.append(1)

        else:

            bn_lst.append(0)

        bin_Score.append(bn_lst)

    return bin_Score
ind_score=[]

#Preparing data for label

with open("../input/personality-score/scores.csv")as f:

    data=f.read()

    data_pro=data.splitlines()

    for i in range(1,len(data_pro)):

        d_score=data_pro[i].split()

        ds= map(float,d_score[1:])

        data_sc = [float(dc.Decimal("%.3f" % e)) for e in ds] # Gold Standard score on each trait for all subject

        #data_sc = [1 if e>=4.6 else e*0 for e in ds]

        #inx=re.findall(r'\d+',d_score[0])

        ind_score.append(data_sc[1:])

        #print(ind_score)

        

ind_score=Conv2Bin(ind_score)       

        

print(ind_score)
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 50000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 400

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(data_val)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(data_val)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)
fscore=pd.DataFrame(ind_score)

Y = pd.get_dummies(fscore).values

#Y = np.ndarray(ind_score) 

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = keras.models.Sequential()

model.add(keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(keras.layers.SpatialDropout1D(0.2))

model.add(keras.layers.recurrent.LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(keras.layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



epochs = 2

batch_size = 33



history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1

                    ,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.00001)])

accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();

plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show();