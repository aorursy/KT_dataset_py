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
import re

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

import tensorflow as tf

import matplotlib.pyplot as plt

from gensim.models import Word2Vec

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D, Embedding, Activation

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn import preprocessing
train_data = pd.read_csv('/kaggle/input/ag-news-classification-dataset/train.csv')

test_data = pd.read_csv('/kaggle/input/ag-news-classification-dataset/test.csv')
train_data.head()
train_data['summary'] = train_data['Title'] + ' ' + train_data['Description']

test_data['summary'] = test_data['Title'] + ' ' + test_data['Description']



train_data = train_data.drop(columns=['Title', 'Description'])

test_data = test_data.drop(columns=['Title', 'Description'])



labels = {1:'World News', 2:'Sports News', 3:'Business News', 4:'Science-Technology News'}



train_data['label'] = train_data['Class Index'].map(labels)

test_data['label'] = test_data['Class Index'].map(labels)
train_data = train_data.drop(columns=['Class Index'])

test_data = test_data.drop(columns=['Class Index'])

train_data.head()
# remove punctuation



def remove_punc(text):

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



train_data['summary'] = train_data['summary'].apply(lambda x: remove_punc(x))

test_data['summary'] = test_data['summary'].apply(lambda x: remove_punc(x))
# data cleaning and remove stopwords



def data_cleaner(text):        

    lower_case = text.lower()

    tokens=word_tokenize(lower_case)

    return (" ".join(tokens)).strip()



def remove_stopwords (text):        

    list1=[word for word in text.split() if word not in stopwords.words('english')]

    return " ".join(list1)



train_data['summary'] = train_data['summary'].apply(lambda x: data_cleaner(x))

test_data['summary'] = test_data['summary'].apply(lambda x: data_cleaner(x))



train_data['summary'] = train_data['summary'].apply(lambda x: remove_stopwords(x))

test_data['summary'] = test_data['summary'].apply(lambda x: remove_stopwords(x))
# split the data into train and test data



X_train, X_validation, y_train, y_validation = train_test_split(train_data['summary'], train_data['label'],

                                                                test_size=0.2, random_state=1)
t_d = []

for i in train_data['summary']:

    t_d.append(i.split())

print(t_d[:2])
# initiate word2vec model



w2v_model = Word2Vec(t_d, size=50, workers=32, min_count=1, window=3)

print(w2v_model)
# tokenize the data



token = Tokenizer(89740)

token.fit_on_texts(train_data['summary'])

token_text = token.texts_to_sequences(train_data['summary'])

token_text = pad_sequences(token_text)
la = preprocessing.LabelEncoder()

y = la.fit_transform(train_data['label'])

y = to_categorical(y)

print(y[:5])
# spilt the data into training and testing data



X_train, X_test, y_train, y_test = train_test_split(np.array(token_text), y, test_size=0.2)
# build the model



keras_model = Sequential()

keras_model.add(w2v_model.wv.get_keras_embedding(True))

keras_model.add(Dropout(0.2))

keras_model.add(Conv1D(50, 3, activation='relu', padding='same', strides=1))

keras_model.add(MaxPool1D())

keras_model.add(Dropout(0.2))

keras_model.add(Conv1D(100, 3, activation='relu', padding='same', strides=1))

keras_model.add(MaxPool1D())

keras_model.add(Dropout(0.2))

keras_model.add(Conv1D(200, 3, activation='relu', padding='same', strides=1))

keras_model.add(GlobalMaxPool1D())

keras_model.add(Dropout(0.2))

keras_model.add(Dense(200))

keras_model.add(Activation('relu'))

keras_model.add(Dropout(0.2))

keras_model.add(Dense(4))

keras_model.add(Activation('softmax'))

keras_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

keras_model.summary()
# train the model



keras_model.fit(X_train, y_train, batch_size=256, epochs=10, validation_data=(X_test, y_test))
labels = la.classes_

print(labels)
# check prediction



predicted = keras_model.predict(X_test)
for i in range(10,50,3):

    print(train_data['summary'].iloc[i][:50], "...")

    print("Actual category: ", labels[np.argmax(y_test[i])])

    print("predicted category: ", labels[np.argmax(predicted[i])])