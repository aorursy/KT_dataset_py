import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

import os

import csv

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
true=pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

fake=pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

true.head()
true['result']=1

fake['result']=0
true.head()
df=pd.concat([true,fake])

df.tail()
df.isna().sum()
df['text']=df['title']+""+df['text']+""+df['subject']

del df['title']

del df['date']

del df['subject']

df.head()
sentence = df['text'].values.tolist()

result= df['result'].values.tolist()

X_train, X_test, Y_train,Y_test= train_test_split(sentence, result, test_size=0.2)

Y_train=np.array(Y_train)

Y_test=np.array(Y_test)
tokenizer=Tokenizer(num_words=10000, oov_token='<OOV>')

tokenizer.fit_on_texts(X_train)

word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(X_train)

padded_train=pad_sequences(sequences,5000,truncating='post')





sequences_test=tokenizer.texts_to_sequences(X_test)

padded_test=pad_sequences(sequences_test,5000,truncating='post')

padded_test.shape
Y_test.shape
model= tf.keras.Sequential([

    tf.keras.layers.Embedding(10000,16,input_length=5000),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history=model.fit(padded_train, Y_train, epochs=10, validation_data=(padded_test, Y_test))

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()