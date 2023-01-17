# load packages

from sklearn import metrics,preprocessing,model_selection

from sklearn.metrics import accuracy_score

import keras

from keras.layers import Input, Lambda, Dense

from keras.models import Model

import keras.backend as K

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import string

import pandas as pd

import re

import spacy

from nltk.corpus import stopwords

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from spacy.lang.en import English

spacy.load('en')

parser = English()
# get elmo from tensorflow hub



import tensorflow_hub as hub

import tensorflow as tf



embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)



# ELMo Embedding

def ELMoEmbedding(x):

    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
# Stop words and special characters 

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)) 

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]
# Data Cleaner and tokenizer

def tokenizeText(text):

    

    text = text.strip().replace("\n", " ").replace("\r", " ")

    text = text.lower()

    

    tokens = parser(text)

    

    # lemmatization

    lemmas = []

    for tok in tokens:

        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas

    

    # reomve stop words and special charaters

    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]

    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    

    tokens = [tok for tok in tokens if len(tok) >= 3]

    

    # remove remaining tokens that are not alphabetic

    tokens = [tok for tok in tokens if tok.isalpha()]

    

    tokens = list(set(tokens))

    

    return ' '.join(tokens[:])
def encode(le_enc, labels):

    enc = le_enc.transform(labels)

    return keras.utils.to_categorical(enc)



def decode(le_enc, one_hot):

    dec = np.argmax(one_hot, axis=1)

    return le_enc.inverse_transform(dec)
# load the dataset

trainDF_Sheet_1 = pd.read_csv('../input/Sheet_1.csv',usecols=['response_id','class','response_text'],encoding='latin-1')
trainDF_Sheet_1.head(10)
trainDF_Sheet_1.shape
trainDF_Sheet_1['class'].unique()
trainDF_Sheet_1['class'].value_counts()
sns.set(rc={'figure.figsize':(8,8)})

sns.countplot(trainDF_Sheet_1['class'])
# Data cleaning

trainDF_Sheet_1['response_text'] = trainDF_Sheet_1['response_text'].apply(lambda x:tokenizeText(x))
# Data preparation

X = trainDF_Sheet_1['response_text'].tolist()

y = trainDF_Sheet_1['class'].tolist()



# Lebel encoding

le_enc = preprocessing.LabelEncoder()

le_enc.fit(y)



y_en = encode(le_enc, y)
# split the dataset into training and testing datasets

x_train, x_test, y_train, y_test = model_selection.train_test_split(np.asarray(X), np.asarray(y_en), test_size=0.2, random_state=42)
x_train.shape
# Build Model

input_text = Input(shape=(1,), dtype=tf.string)

embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

dense = Dense(256, activation='relu')(embedding)

pred = Dense(2, activation='softmax')(dense)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())  

    session.run(tf.tables_initializer())

    history = model.fit(x_train, y_train, epochs=20, batch_size=16)

    model.save_weights('./response-elmo-model.h5')



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('./response-elmo-model.h5')  

    predicts = model.predict(x_test, batch_size=16)
# decode test labels

y_test = decode(le_enc, y_test)

# decode predicted labels

y_preds = decode(le_enc, predicts)
print(metrics.confusion_matrix(y_test, y_preds))
print(metrics.classification_report(y_test, y_preds))
print("Accuracy of ELMO is:",accuracy_score(y_test,y_preds))
# load the dataset

trainDF_Sheet_2 = pd.read_csv('../input/Sheet_2.csv',encoding='latin-1')
trainDF_Sheet_2.head(10)
trainDF_Sheet_2.shape
trainDF_Sheet_1['class'].unique()
trainDF_Sheet_2['class'].value_counts()
sns.countplot(trainDF_Sheet_2['class'])
# Data cleaning

trainDF_Sheet_2['resume_text'] = trainDF_Sheet_2['resume_text'].apply(lambda x:tokenizeText(x))
# Data preparation

X = trainDF_Sheet_2['resume_text'].tolist()

y = trainDF_Sheet_2['class'].tolist()



# Lebel encoding

le_enc = preprocessing.LabelEncoder()

le_enc.fit(y)



y_en = encode(le_enc, y)
# split the dataset into training and testing datasets

x_train, x_test, y_train, y_test = model_selection.train_test_split(np.asarray(X), np.asarray(y_en), test_size=0.2, random_state=42)
x_train.shape
# Build Model

input_text = Input(shape=(1,), dtype=tf.string)

embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

dense = Dense(256, activation='relu')(embedding)

pred = Dense(2, activation='softmax')(dense)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())  

    session.run(tf.tables_initializer())

    history = model.fit(x_train, y_train, epochs=20, batch_size=16)

    model.save_weights('./resume-elmo-model.h5')



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('./resume-elmo-model.h5')  

    predicts = model.predict(x_test, batch_size=16)
# decode test labels

y_test = decode(le_enc, y_test)

# decode predicted labels

y_preds = decode(le_enc, predicts)
print(metrics.confusion_matrix(y_test, y_preds))
print(metrics.classification_report(y_test, y_preds))
print("Accuracy of ELMO is:",accuracy_score(y_test,y_preds))