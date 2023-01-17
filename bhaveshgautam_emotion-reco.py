import pandas as pd

import numpy as np

import glob

import pickle

from itertools import chain 

from keras.preprocessing.sequence import pad_sequences

import keras
files = glob.glob("/kaggle/input/emotion-classification/")

data=pd.read_csv("/kaggle/input/emotion-classification/emotion.data")

texts=data.text

emotions=list(data.emotions)
sentences=[text.split() for text in texts]

text_words=list(chain.from_iterable(sentences))

words_with_id=dict()

emotions_with_id=dict()

for word in text_words:

    if word not in words_with_id:

        words_with_id[word]=len(words_with_id)

for emo in emotions:

    if emo not in emotions_with_id:

        emotions_with_id[emo]=len(emotions_with_id)
X=[[words_with_id[word] for word in sentence] for sentence in sentences]

Y=[emotions_with_id[emo] for emo in emotions]

X = pad_sequences(X,200)

Y = keras.utils.to_categorical(Y, num_classes=len(emotions_with_id), dtype='float32')
print(X.shape[1])

print(Y[0:10])
model = keras.Sequential()

model.add(keras.layers.Embedding(X.shape[0], 100))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.LSTM(100, dropout_U = 0.2, dropout_W = 0.2))

model.add(keras.layers.Dense(len(emotions_with_id), activation="softmax"))

model.summary()  # prints a summary of the model

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

model.fit(X, Y, epochs=2, batch_size=64, validation_split=0.1, shuffle=True)

model.save("/kaggle/working/find_emotions2.h5")