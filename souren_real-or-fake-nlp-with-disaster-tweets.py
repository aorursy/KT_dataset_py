from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf

#tf.compat.v1.enable_eager_execution()

tf.disable_eager_execution()

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import re

import seaborn as sns

from keras import regularizers

import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate,Conv1D,MaxPooling1D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from tensorflow.keras import backend as K

np.random.seed(10)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from absl import logging

from nltk.stem import PorterStemmer 

ps = PorterStemmer()

import spacy

nlp=spacy.load("en_core_web_lg")
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.isnull().sum()
from collections import Counter

train.keyword = train['keyword'].str.replace("[^a-zA-Z#]", " ")

keyword = train.keyword[train.keyword.notnull()].tolist()

keyword = Counter(keyword)

keywords = pd.DataFrame(keyword.most_common(10), columns=['Keyword', 'Count'])

sns.set(rc={'figure.figsize':(14,6)})

sns.barplot(data = keywords, x = 'Keyword', y = 'Count')

plt.title("Most Common Keywords")

plt.show()
sns.countplot(train['target'])

plt.title("Distribution Of Target")

sns.set(rc={'figure.figsize':(10,8)})

plt.show()
from spacy.lang.en.stop_words import STOP_WORDS

stopwords = list(STOP_WORDS)

import string

punct=string.punctuation





def text_data_cleaning(sentence):

    doc = nlp(sentence)

    

    tokens = []

    for token in doc:

        if token.lemma_ != "-PRON-":

            temp = token.lemma_.lower().strip()

        else:

            temp = token.lower_

        tokens.append(temp)

    

    cleaned_tokens = []

    for token in tokens:

        if token not in stopwords and token not in punct:

            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)
'''import re

import string

def clean_text(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub('\n', '', text)

    return text'''



train['text'] = train.text.apply(lambda x: text_data_cleaning(x))

#train['text'] = train['text'].str.replace("[^a-zA-Z#]", " ")

#train['text'] = train['text'].apply(lambda x: ' '.join([ps.stem(w) for w in x.split() if len(w)>3]))
train.keyword = train.keyword.fillna("")

train['new_text'] = train.text

test.keyword = test.keyword.fillna("")

test['text'] = test.text

test['text'] = test.text.apply(lambda x: text_data_cleaning(x))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.new_text.values, train.target.values, random_state = 42, test_size=0.2)

test_data = test.text.values
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
#elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=True, name="{}_module".format("mymod"))

#elmo_layer = hub.KerasLayer(elmo,trainable=True)
def build_model(embed):

    model = Sequential([

        Input(shape=[], dtype=tf.string),

        embed,

        Dense(1024, activation='elu'),

        BatchNormalization(),

        Dropout(0.5),

        Dense(512, activation='elu'),

        BatchNormalization(),

        Dropout(0.35),

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.1),

        Dense(1, activation='sigmoid')

    ])

    model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model(embed)

model.summary()
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=2, epsilon=1e-4, mode='min')
with tf.compat.v1.Session() as session:

    tf.compat.v1.keras.backend.set_session(session)

    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    history = model.fit(

        X_train, y_train,

        validation_data=(X_test,y_test),

        epochs=35,

        callbacks=[earlyStopping,reduce_lr_loss,mcp_save],

        batch_size=128

    )
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

#plt.show()

plt.subplot(2,2,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
with tf.Session() as session:

    tf.compat.v1.keras.backend.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('model.hdf5')

    y_pred = model.predict(X_test)

    

from sklearn.metrics import confusion_matrix, classification_report

#print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred.round().astype(int)))
with tf.Session() as session:

    tf.compat.v1.keras.backend.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('model.hdf5')

    sub = model.predict(test_data)

    



subm = pd.DataFrame()

subm['id'] = test['id']

subm['target'] = sub.round().astype(int)

subm.to_csv("pred.csv", index = False)