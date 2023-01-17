import pandas as pd

import tensorflow as tf

df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True)

df.head()
sentences=list(df['headline'])

labels=list(df['is_sarcastic'])

url=list(df['article_link'])
vocab_size= 1000

embedding_dim=32

max_length = 16

trunc_type='post'

padding_type='post'

oov_tok='<OOV>'

training_size=20000
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
training_sentences= sentences[:training_size]

test_sentences = sentences[training_size:]

training_labels = labels[:training_size]

test_labels = labels[training_size:]
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)

padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)



test_sequences = tokenizer.texts_to_sequences(test_sentences)

test_padded = pad_sequences(test_sequences, maxlen = max_length, truncating = trunc_type)
model1 = tf.keras.Sequential([

     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

num_epochs = 15

history=model1.fit(padded, training_labels, epochs=num_epochs, validation_data=(test_padded, test_labels),verbose=1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

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