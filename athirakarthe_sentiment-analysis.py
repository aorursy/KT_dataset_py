# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

print(train_df.columns,test_df.columns)
train_text = list(train_df.text)

train_sentiment = list(train_df.sentiment)

test_text = list(test_df.text)

test_sentiment = list(test_df.sentiment)

import re

test_curated_text = []

train_curated_text = []

for text in train_text:

    train_curated_text.append( re.sub(r"http\S+", "", str(text)))

for text in test_text:

    test_curated_text.append(re.sub(r'http\S+',"",str(text)))

from tensorflow.keras.preprocessing.text import Tokenizer

clean_text = []

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_curated_text)

word_index = tokenizer.word_index
max_length = max([len(text) for text in train_curated_text])

from tensorflow.keras.preprocessing.sequence import pad_sequences

sequences = tokenizer.texts_to_sequences(train_curated_text)

padded = pad_sequences(sequences,maxlen=max_length, truncating='post')



testing_sequences = tokenizer.texts_to_sequences(test_curated_text)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

print('sentence:',train_curated_text[2],'\nencoding:',sequences[2],'\npadded encoding:',padded[2])
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()

lb.fit(train_sentiment)

training_labels_final = lb.transform(train_sentiment)

testing_labels_final = lb.transform(test_sentiment)

print('Before Binarizing:',train_sentiment[4:7],'\nAfter Binarizing',training_labels_final[4:7])
vocab_size = len(word_index)

embedding_dim = 13

import tensorflow as tf

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAvgPool1D(),

    tf.keras.layers.Dense(3, activation='softmax')

])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 20

history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
from matplotlib import pyplot as plt

plt.plot(history.history['val_accuracy'])

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
inputs = np.append(padded,testing_padded,axis=0)

labels = np.append(training_labels_final,testing_labels_final,axis=0)

num_epochs = 8

history = model.fit(inputs,labels, epochs=num_epochs)
example = tokenizer.texts_to_sequences(["i feel nausea",'the model is doing pretty good','But not great'])

example = pad_sequences(example,maxlen=max_length)

pred = model.predict(example)

print(pred[0],pred[1],pred[2])