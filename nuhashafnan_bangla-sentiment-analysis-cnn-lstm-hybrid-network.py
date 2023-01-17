import pandas as pd

from pandas import read_excel

import numpy as np

import re

from re import sub

import multiprocessing

from unidecode import unidecode

import os

from time import time 

import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout,Activation,Embedding,Flatten,Bidirectional,MaxPooling2D, Conv1D, MaxPooling1D

from keras.optimizers import SGD,Adam

from keras import regularizers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

import h5py

import csv

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
def text_to_word_list(text):

    text = text.split()

    return text



def replace_strings(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           u"\u00C0-\u017F"          #latin

                           u"\u2000-\u206F"          #generalPunctuations

                               

                           "]+", flags=re.UNICODE)

    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)

    #latin_pattern=re.compile('[A-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]*',)

    

    text=emoji_pattern.sub(r'', text)

    text=english_pattern.sub(r'', text)



    return text



def remove_punctuations(my_str):

    # define punctuation

    punctuations = '''````£|¢|Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰⚽️✌�￰৷￰'''

    

    no_punct = ""

    for char in my_str:

        if char not in punctuations:

            no_punct = no_punct + char



    # display the unpunctuated string

    return no_punct







def joining(text):

    out=' '.join(text)

    return out



def preprocessing(text):

    out=remove_punctuations(replace_strings(text))

    return out
df=pd.read_excel('/kaggle/input/pseudolabel/predicted_unsupervised_sentiment.xlsx')

display(df)
sns.countplot(df['sentiment']);
df['sentence'] = df.sentence.apply(lambda x: preprocessing(str(x)))
df.reset_index(drop=True, inplace=True)
train1, test1 = train_test_split(df,random_state=69, test_size=0.2)

training_sentences = []

testing_sentences = []







train_sentences=train1['sentence'].values

train_labels=train1['sentiment'].values

for i in range(train_sentences.shape[0]): 

    #print(train_sentences[i])

    x=str(train_sentences[i])

    training_sentences.append(x)

    

training_sentences=np.array(training_sentences)











test_sentences=test1['sentence'].values

test_labels=test1['sentiment'].values



for i in range(test_sentences.shape[0]): 

    x=str(test_sentences[i])

    testing_sentences.append(x)

    

testing_sentences=np.array(testing_sentences)





train_labels=keras.utils.to_categorical(train_labels)





test_labels=keras.utils.to_categorical(test_labels)

print("Training Set Length: "+str(len(train1)))

print("Testing Set Length: "+str(len(test1)))

print("training_sentences shape: "+str(training_sentences.shape))

print("testing_sentences shape: "+str(testing_sentences.shape))

print("train_labels shape: "+str(train_labels.shape))

print("test_labels shape: "+str(test_labels.shape))

print(training_sentences[1])

print(train_labels[0])
vocab_size = 25000

embedding_dim = 300

max_length = 100

trunc_type='post'

oov_tok = "<OOV>"
print(training_sentences.shape)

print(train_labels.shape)
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

print(len(word_index))

print("Word index length:"+str(len(tokenizer.word_index)))

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)





test_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(test_sequences,maxlen=max_length)
print("Sentence :--> \n")

print(training_sentences[2]+"\n")

print("Sentence Tokenized and Converted into Sequence :--> \n")

print(str(sequences[2])+"\n")

print("After Padding the Sequence with padding length 100 :--> \n")

print(padded[2])
print("Padded shape(training): "+str(padded.shape))

print("Padded shape(testing): "+str(testing_padded.shape))
with tf.device('/gpu:0'):

    model= Sequential()

    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

    model.add(Conv1D(200, kernel_size=3, activation = "relu"))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))

    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(64)))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Flatten())

    #l2 regularizer

    model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation="relu"))

    model.add(Dense(2, activation='softmax'))

    #sgd= SGD(lr=0.0001,decay=1e-6,momentum=0.9,nesterov=True)

    adam=Adam(learning_rate=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    history=model.fit(padded,train_labels,epochs=5,batch_size=256,validation_data=(testing_padded,test_labels),use_multiprocessing=True, workers=8)
print(history.history.keys())

loss = history.history['loss']

val_loss = history.history['val_loss']

plt.plot(loss)

plt.plot(val_loss)

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['loss', 'val_loss'])

plt.show()



accuracy = history.history['accuracy']

val_accuracy= history.history['val_accuracy']

plt.plot(accuracy)

plt.plot(val_accuracy)

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['accuracy', 'val_accuracy'])

plt.show()
#accuracy calculation

loss_and_metrics = model.evaluate(padded,train_labels,batch_size=256)

print("The train accuracy is: "+str(loss_and_metrics[1]))

loss_and_metrics = model.evaluate(testing_padded,test_labels,batch_size=256)

print("The test accuracy is: "+str(loss_and_metrics[1]))