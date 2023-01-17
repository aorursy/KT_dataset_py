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
import matplotlib.pyplot as plt

import re

import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



from sklearn import preprocessing



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, MaxPool1D, Dropout, Dense, GlobalMaxPooling1D, Embedding, Activation

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
train_data = pd.read_csv('/kaggle/input/hate-speech-detection/toxic_train.csv')

test_data = pd.read_csv('/kaggle/input/hate-speech-detection/toxic_test.csv')
train_data.head()
# drop unnamed column



train_data = train_data.drop(columns=['Unnamed: 0'])

train_data.head()
test_data.head()
test_data = test_data.drop(columns=['Unnamed: 0'])

test_data.head()
def preprocess_text(sen):

    # lower the character

    sentence = sen.lower()

    

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)

    

    stops = stopwords.words('english')

    

    for word in sentence.split():

        if word in stops:

            sentence = sentence.replace(word, '')

    return sentence
# preprocess data



train_data['comment_text'] = train_data['comment_text'].apply(lambda x : preprocess_text(x))

test_data['comment_text'] = test_data['comment_text'].apply(lambda x : preprocess_text(x))
# tokenize the data



token = Tokenizer(28164)

token.fit_on_texts(train_data['comment_text'])

text = token.texts_to_sequences(train_data['comment_text'])

text = pad_sequences(text, maxlen=100)
y = train_data['toxic'].values
# split the data into training and testing data



X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=1, stratify=y)
# build the model



max_features = 28164

embedding_dim = 32



model = Sequential()

model.add(Embedding(max_features, embedding_dim))

model.add(Dropout(0.2))

model.add(LSTM(32, return_sequences=True))

model.add(Dropout(0.2))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()
# compile and train model



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=1024, validation_data=(X_test, y_test), epochs=5)