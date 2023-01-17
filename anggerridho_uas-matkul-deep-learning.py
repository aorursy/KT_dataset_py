# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import nltk

import os

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup

import re

from keras.utils import to_categorical

import random

from tensorflow import set_random_seed

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense,Dropout,Embedding,LSTM

from keras.callbacks import EarlyStopping

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.models import Sequential

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

lemmatizer = WordNetLemmatizer()



#set random seed for the session and also for tensorflow that runs in background for keras

set_random_seed(123)

random.seed(123)





print(os.listdir("../input"))
train= pd.read_csv("../input/train.tsv", sep="\t")

test = pd.read_csv("../input/test.tsv", sep="\t")



train.head()
train.shape
test.head()
def clean_sentences(df):

    reviews = []



    for sent in tqdm(df['Phrase']):

        

        #remove html content

        review_text = BeautifulSoup(sent).get_text()

        

        #remove non-alphabetic characters

        review_text = re.sub("[^a-zA-Z]"," ", review_text)

    

        #tokenize the sentences

        words = word_tokenize(review_text.lower())

    

        #lemmatize each word to its lemma

        lemma_words = [lemmatizer.lemmatize(i) for i in words]

    

        reviews.append(lemma_words)



    return(reviews)



#cleaned reviews for both train and test set retrieved

train_sentences = clean_sentences(train)

test_sentences = clean_sentences(test)

print(len(train_sentences))

print(len(test_sentences))
target=train.Sentiment.values

y_target=to_categorical(target)

num_classes=y_target.shape[1]
X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)
 #It is needed for initializing tokenizer of keras and subsequent padding



unique_words = set()

len_max = 0



for sent in tqdm(X_train):

    

    unique_words.update(sent)

    

    if(len_max<len(sent)):

        len_max = len(sent)

        

#length of the list of unique_words gives the no of unique words

print(len(list(unique_words)))

print(len_max)
tokenizer = Tokenizer(num_words=len(list(unique_words)))

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

X_test = tokenizer.texts_to_sequences(test_sentences)



#padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.

#Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.

X_train = sequence.pad_sequences(X_train, maxlen=len_max)

X_val = sequence.pad_sequences(X_val, maxlen=len_max)

X_test = sequence.pad_sequences(X_test, maxlen=len_max)

print(X_train.shape,X_val.shape,X_test.shape)
early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_acc', patience = 2)

callback = [early_stopping]



#Model using Keras LSTM

model=Sequential()

model.add(Embedding(len(list(unique_words)),300,input_length=len_max))

model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))

model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

model.add(Dense(100,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])

model.summary()
#This is done for learning purpose only. One can play around with different hyper parameters combinations

#and try increase the accuracy even more. For example, a different learning rate, an extra dense layer 

# before output layer, etc. Cross validation could be used to evaluate the model and grid search 

# further to find unique combination of parameters that give maximum accuracy. This model has a validation

#accuracy of around 66.5%

history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=6, batch_size=256, verbose=1, callbacks=callback)