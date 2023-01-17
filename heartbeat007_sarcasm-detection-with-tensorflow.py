import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
## download the data in json format

! wget http://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json
import json



with open('sarcasm.json','r') as f:

    data = json.load(f)
sentences = []

labels = []

urls = []



for item in data:

    sentences.append(item['headline'])

    labels.append(item['is_sarcastic'])
sentences.__len__()/2
vocab_size = 10000

embedding_dim = 16

max_length = 32

trunc_type = 'post'

padding_type = 'post'

oov_token = "<OOV>"

training_size = 20000
training_sentences = sentences[0:training_size]

testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]

testing_labels = labels[training_size:]
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_sequences
training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
training_padded
testing_sequnces = tokenizer.texts_to_sequences(testing_sentences)
testing_sequnces
testing_padded = pad_sequences(testing_sequnces,maxlen=max_length,padding=padding_type,truncating=trunc_type)
testing_padded
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(24,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
model.summary()
import numpy as np

training_labels = np.array(training_labels)

testing_labels = np.array(testing_labels)
num_epochs = 30

history = model.fit(training_padded,training_labels,epochs=num_epochs,validation_data=(testing_padded,testing_labels))
# summarize history for accuracy

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()