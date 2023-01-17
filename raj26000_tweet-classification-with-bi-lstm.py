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

import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import re

import nltk

from nltk.corpus import stopwords
StopWords = set(stopwords.words('english'))

def text_preprocess(text):

    text = ' '.join([i for i in text.split() if i not in StopWords]) #removing stopwords from each tweet

    text = ' '.join([i for i in text.split() if ('http' not in i and '@' not in i and 'https' not in i)]) #removing mentions and urls

    text = re.sub('#', '', text) #removing hashtags while keeping the tagged word :p

    return text



#The Keras Tokenizer deals with lower-casing of text and editing out punctuations.
train_data = pd.read_csv("../input/nlp-getting-started/train.csv")

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')



X_train = train_data['text']

y_train = train_data['target']

X_test = test_data['text']

test_idx = test_data['id']



train_data['text'] = train_data['text'].apply(text_preprocess)

X_train = train_data['text']

test_data['text'] = test_data['text'].apply(text_preprocess)

X_test = test_data['text']



X_train = X_train.tolist()

X_test = X_test.tolist()

print(X_train)

        

        

       



tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index)

max_len = 150

train_seq = tokenizer.texts_to_sequences(X_train)

train_pad = pad_sequences(train_seq, maxlen=max_len, padding='pre')

test_seq = tokenizer.texts_to_sequences(X_test)

test_pad = pad_sequences(test_seq, maxlen=max_len, padding='pre')







#sanity check to make sure stopwords have been removed. Just a useful way of checking.

for idx, word in tokenizer.word_index.items():

    if word in StopWords:

        print("F")

    else:

        print("T")
model = keras.Sequential([

    keras.layers.Embedding(vocab_size+1, 200, input_length=max_len),

    keras.layers.SpatialDropout1D(0.5),

    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),

    keras.layers.Bidirectional(keras.layers.LSTM(32)),

    keras.layers.Dense(200, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])

optimizer = keras.optimizers.Adam(learning_rate = 0.01)



model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



history = model.fit(train_pad, y_train, epochs=40)

model.summary()
def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.xlabel('epochs')

  plt.ylabel(string)

  plt.show()



plot_graphs(history, 'loss')

plot_graphs(history, 'accuracy')
model.predict_classes(test_pad)
pred = pd.DataFrame()

pred['id'] = test_idx

pred['target'] = model.predict_classes(test_pad)

print(pred)
pred.to_csv('submission.csv', index=False)

print("Submission has been saved")