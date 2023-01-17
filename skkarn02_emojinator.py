import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#For emojis

import emoji

#For Data Manipulation

import pandas as pd

#For numerical array calculations

import numpy as np

#For natutal language tasks

import nltk

import re

#To ignore warning 

import warnings

warnings.filterwarnings('ignore')

#For IMDB dataset

import keras
# We have a  training and testing dataset in form of csv let's import and explore it 

print('TRAINING DATA')

train_data=pd.read_csv('/kaggle/input/emojify/data/train_emoji.csv',header=None)

#Let's look at the data and its shape

print(train_data.head())

print()

print('Size of the  training data is {}'.format(train_data.shape))

print('TESTING DATA')

test_data=pd.read_csv('/kaggle/input/emojify/data/tess.csv',header=None)

print(test_data.head())

print()

print('Size of testing data {}'.format(test_data.shape))
#making a dictionary to encode label into emoji

emoji_dictionary={

    "0" : "\u2764\uFE0F" ,

    "1" : ":baseball:" ,

    "2" : ":grinning_face_with_big_eyes:" ,

    "3" : ":disappointed_face:" ,

    "4" : ":fork_and_knife:" ,

}
for i , w in emoji_dictionary.items() :

    print('{} reprsent {}'.format(i,emoji.emojize(w)))
#We will write a function to do basic cleaning

#Import word tokenize form nltk to perform tokenization

from nltk.tokenize import word_tokenize

def cleaning(text) :

    #Normalizing the data

    text=str(text)

    text=text.lower()

    #removing punctuation

    text=re.sub('[^A-Za-z0-9]',' ',text)

    #Tokenization

    text=word_tokenize(text)

    return text
X_train=train_data[0]

Y_train=train_data[1]

X_test=test_data[0]

Y_test=test_data[1]
#Performing datacleaning

X_train=X_train.apply(cleaning)

X_test =X_test.apply(cleaning)
#Making one hot_vectors

Y_train=pd.get_dummies(Y_train.values)

y_test=pd.get_dummies(Y_test.values)
#Importing tokenizer

from keras.preprocessing.text import Tokenizer

#Making object of Tokenizer Class

tokenizer=Tokenizer()

#Fitting object on review column

tokenizer.fit_on_texts(X_train)

tokenizer.fit_on_texts(X_test)

#Claculating the vocabulary size as it is useful in Embedding layer jst adding 1 for embedding matrix

vocab_size = len(tokenizer.word_index)+1

#Printing size of vocabulary

print('Vocubalary size is equal to {}'.format(vocab_size-1))
#Converting texts to sequences

X_train_token=tokenizer.texts_to_sequences(X_train)

X_test_token=tokenizer.texts_to_sequences(X_test)
#Calculating maximum length which will be useful in padding

maxlength=max([len(s) for s in X_train_token])

print('Maximum length of sequence in training set is {}'.format(maxlength))

maxlength_test=max([len(s) for s in X_test_token])

print('Maximum length of sequence in testing set is {}'.format(maxlength_test))
#Importing module

from keras.preprocessing.sequence import pad_sequences

#Performing padding

X_train_token_pd=pad_sequences(X_train_token,maxlen=maxlength,padding='pre')

X_test_token_pd=pad_sequences(X_test_token,maxlen=maxlength_test,padding='pre')
#Importing necessary modules for Deep learning

import tensorflow

import keras

from keras.layers import Dense,Embedding,SimpleRNN,Dropout

from keras.models import Sequential
#Creating a sequential model

model=Sequential()

#Creating a embedding layer which will learn embedding for us which will 128 dimensional vector

model.add(Embedding(vocab_size,128))

#Creating a two stacked Simple RNN layer of hidden  units = 128

model.add(SimpleRNN(128,return_sequences=True))

model.add(Dropout(0.5))

model.add(SimpleRNN(128,return_sequences=False))

model.add(Dropout(0.5))

#Adding dense layer

model.add(Dense(100,activation='relu'))

model.add(Dense(5,activation='softmax'))
# COMPILING THE MODEL with adam optimization algorithm

model.compile(optimizer='adam',loss='categorical_crossentropy')
#Fitiing the model

training=model.fit(X_train_token_pd,Y_train,validation_data=[X_test_token_pd,y_test],batch_size=32,epochs=200)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.figure(figsize=(10,6))

plt.plot(training.history['loss'],label='Training loss',color='red')

plt.title('Training loss Graphical reprsentation')

plt.legend(loc='upper right')

plt.show()
## A function which will take in the input sentence and predict the emoji

def prediction (data) :

    encoded=cleaning([data])

    encoded=tokenizer.texts_to_sequences([encoded])

    return encoded

def model_out () :

    data=input("Enter your sentence------>")

    encod=prediction(data)

    emoj=str(model.predict_classes(encod)[0])

    print(encod)

    print('Predicted emoji {}'.format(emoji.emojize(emoji_dictionary[str(emoj)])))

        
#we have a glove 50 dimensional word embedding let's unbox it and we will make embedding matrix using it 

f = open('/kaggle/input/glove50d/glove.6B.50d.txt',encoding='utf-8')

#Making a dictionary to store word embeddings

word_to_vec={}

for line in f :

    values=line.split()

    word_to_vec[values[0]]=np.array(values[1:])

f.close
np.random.seed(42)

embedding_matrix=np.random.rand(vocab_size,50)

for w , i in tokenizer.word_index.items() :

    encoding_vector=word_to_vec[w]

    if encoding_vector is not None :

        embedding_matrix[i] = encoding_vector
from keras.layers import LSTM

model1 = Sequential()

model1.add(Embedding(vocab_size,50,weights=[embedding_matrix],trainable=False))

model1.add((LSTM(256, return_sequences=True)))

model.add(Dropout(0.5))

model1.add((LSTM(256)))

model1.add(Dropout(0.25))

model1.add(Dense(units=5, activation='softmax'))

model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#Fitiing the model

training1=model1.fit(X_train_token_pd,Y_train,validation_data=[X_test_token_pd,y_test],batch_size=32,epochs=200)
plt.figure(figsize=(10,6))

plt.plot(training1.history['loss'],label='Training loss')

plt.title('TRAINING LOSS')

plt.show()
def prediction (data) :

    encoded=cleaning([data])

    encoded=tokenizer.texts_to_sequences([encoded])

    return encoded

def model_out () :

    data=input("Enter your sentence------>")

    encod=prediction(data)

    emoj=str(model1.predict_classes(encod)[0])

    print(encod)

    print('Predicted emoji {}'.format(emoji.emojize(emoji_dictionary[str(emoj)])))