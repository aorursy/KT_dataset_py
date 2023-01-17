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

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



%matplotlib inline
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test.head()
train.head()
def cleantext(text):

    text = re.sub(r'@[A-Za-z0-9]+' , ' ',text)  # remove @mention

    text = re.sub(r'#',' ',text)                # remove the '#' symbol

    text = re.sub(r'RT[\S]+', ' ', text)        # remove Retweet

    text = re.sub(r'https?:\/\/\S + ', ' ',text)# remove the hyper link

    return text



train['text'] = train['text'].apply(cleantext)
test['text']  = test['text'].apply(cleantext)
test.head()
train.head()
train_sequences = train['text']

test_sequences = test['text']

train_target = train['target']

print(train_target.head(10))
print(train_sequences.shape[:])

print(train_target.shape[:])

print(len(train_sequences))



[X_train, X_test, Y_train, Y_test] = train_test_split(train_sequences, train_target, test_size = 0.1, random_state=2)

print(X_train.shape[:])

print(X_test.shape[:])

print(Y_train.shape[:])

print(Y_test.shape[:])
train_target.value_counts()


train_target.value_counts().plot.bar()
train_sequences.describe()
max_length = train_sequences.map(lambda x: len(x)).max()

print(max_length)
train_sequences.isnull().any()
vocal_size = 10000

embedding_dim = 10

max_length = 157

trunc_type = 'post'

oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words=vocal_size,oov_token = oov_tok)

tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(X_train)

padding = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)





testing_sequences = tokenizer.texts_to_sequences(X_test)

test_padded = pad_sequences(testing_sequences, maxlen = max_length)



val_sequence = tokenizer.texts_to_sequences(test_sequences)

val_padded = pad_sequences(val_sequence,maxlen= max_length)
model = tf.keras.models.Sequential([

    tf.keras.layers.Embedding(vocal_size, embedding_dim,input_length=max_length),

    tf.keras.layers.Flatten(),

    #tf.keras.layers.Dense(157,activation='relu'),

    tf.keras.layers.Dense(157,activation='relu'),

    tf.keras.layers.Dense(75,activation='relu'),

    tf.keras.layers.Dense(20,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epochs = 5

history = model.fit(padding,Y_train,epochs=num_epochs,validation_data=(test_padded, Y_test))
model.save_weights('model.h5')
fig, (ax1,ax2)  = plt.subplots(2,1, figsize=(12,12))

ax1.plot(history.history['loss'],color='b',label='Training_loss')

ax1.plot(history.history['val_loss'], color='g', label='Validation_loss')

ax1.legend(loc='best', shadow=True)



ax2.plot(history.history['accuracy'], color='b',label='Trainning_accuracy')

ax2.plot(history.history['val_accuracy'], color='g',label= 'validation_accuracy')

ax2.legend(loc='best',shadow=True)