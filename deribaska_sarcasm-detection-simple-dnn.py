import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

print('Tensorflow version ', tf.__version__)

df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json',lines=True)
sentences = list(df['headline'])

vocab_size = 16000

embedding_dim = 32

max_length = df.headline.map(len).max()



tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size)

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print('Word index length = ' + str(len(word_index)))



sequences = tokenizer.texts_to_sequences(sentences)

padded = tf.keras.preprocessing.sequence.pad_sequences(sequences,padding='post',maxlen=max_length)



print(sentences[213])

print(padded[213])

print(padded.shape)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(padded, df['is_sarcastic'].values, batch_size=500,epochs=6,validation_split=0.2)
print('Accuracy on validation data  - ',np.max(history.history['val_accuracy']))

print('Loss on validation data      - ',np.min(history.history['val_loss']))
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.plot(history.history['accuracy'])

ax1.plot(history.history['val_accuracy'])

vline_cut = np.where(history.history['val_accuracy'] == np.max(history.history['val_accuracy']))[0][0]

ax1.axvline(x=vline_cut, color='k', linestyle='--')

ax1.set_title("Accuracy")

ax1.legend(['Train', 'Test'])



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]

ax2.axvline(x=vline_cut, color='k', linestyle='--')

ax2.set_title("Loss")

ax2.legend(['Train', 'Test'])

plt.show()