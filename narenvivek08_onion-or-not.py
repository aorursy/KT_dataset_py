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
data=pd.read_csv('../input/onion-or-not/OnionOrNot.csv')
data.head()
sentence = data['text'].values.tolist()
result= data['label'].values.tolist()
X_train, X_test, Y_train,Y_test= train_test_split(sentence, result, test_size=0.2)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)
tokenizer=Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(X_train)
padded_train=pad_sequences(sequences,250,truncating='post')
sequences_test=tokenizer.texts_to_sequences(X_test)
padded_test=pad_sequences(sequences_test,250,truncating='post')
print(padded_train.shape)
print(padded_test.shape)
print(Y_train.shape)
print(Y_test.shape)
model= tf.keras.Sequential([
    tf.keras.layers.Embedding(5000,16,input_length=250),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history=model.fit(padded_train, Y_train, epochs=15, validation_data=(padded_test, Y_test))
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
