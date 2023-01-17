# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import spacy
from spacy.util import minibatch, compounding
from spacy import displacy


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
ss = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train.head()
train.dropna(inplace=True)
def clean_text(text):

    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    return text
train['text_cleaned'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text_cleaned'] = train['selected_text'].apply(lambda x:clean_text(x))
t = train["text_cleaned"].values
s_t = train["selected_text_cleaned"].values
sen = train["sentiment"].values
tokenizer = Tokenizer(num_words=25290, oov_token='OOV')

tokenizer.fit_on_texts(t)

word_index = tokenizer.word_index
 
train_sequences = tokenizer.texts_to_sequences(t)

selected_sequences = tokenizer.texts_to_sequences(s_t)


print(len(word_index))
tokenizer_y = Tokenizer(num_words=4, oov_token='OOV')

tokenizer_y.fit_on_texts(sen)
label_index = tokenizer_y.word_index
label_sequences = tokenizer_y.texts_to_sequences(sen)
del label_index["OOV"]
label_index["neutral"] = 0
label_index["positive"] = 1
label_index["negative"] = 2

label_sequences = tokenizer_y.texts_to_sequences(sen)
label_index

for i in range(len(train_sequences)):
    l1, l2 = len(train_sequences[i]), len(selected_sequences[i])
    
    label = label_sequences[i]
    
    label_sequences[i] = [0] * len(train_sequences[i])
    
    for j in range(l1):
        if train_sequences[i][j:j+l2] == selected_sequences[i]:
            
            label_sequences[i][j:j+l2] = label * l2

            
padded_inputs = pad_sequences(train_sequences, padding="post")
padded_labels = pad_sequences(label_sequences, padding="post")
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(padded_inputs,padded_labels, test_size=0.1)
padded_inputs.shape
padded_labels.shape
print(padded_labels)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index), 128, input_length = 35),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=0.5)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(x_tr, y_tr, batch_size=32, epochs=10, validation_split=0.1, verbose=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
i = 0
p = model.predict(np.array([x_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15}:{} -- {}".format("Word", "True", "Pred"))
print()
for w, t, pred in zip(x_te[i],y_te[i], p[0]):
    if w != 0:
        print("{:15}: {} -- {}".format(tokenizer.index_word[w],t, pred))


