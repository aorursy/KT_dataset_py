import pandas as pd

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers

import tensorflow.keras.utils as ku

import numpy as np
print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
data_dir = "../input/sentiment140/training.1600000.processed.noemoticon.csv"

total_words = 8000

max_seq_len = 240

embedding_dim_size = 200

TRAIN_PROP=0.9
df = pd.read_csv(data_dir, encoding ="ISO-8859-1" , names= ["target", "ids", "date", "flag", "user", "text"])

df
#Taken from another notebook

df=df.sample(frac=0.002)

text_target = df[['text', 'target']]

targets, texts = df.iloc[:,0], df.iloc[:,5]

targets = targets/4 # 0 and 4 --> 0 and 1 

targets.astype(int)

print(len(texts))

print(len(targets))
## Split into train and test
TRAIN_SIZE=int(len(texts)*TRAIN_PROP)



train_texts = texts[0:TRAIN_SIZE]

train_targets = targets[0:TRAIN_SIZE]



validation_texts = texts[TRAIN_SIZE:]

validation_targets = targets[TRAIN_SIZE:]

print("len(train_texts)",len(train_texts))

print("len(train_targets)",len(train_targets))



print("len(validation_texts)",len(validation_texts))

print("len(validation_targets)",len(validation_targets))
tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')

tokenizer.fit_on_texts(train_texts)



train_sequences = tokenizer.texts_to_sequences(train_texts)

validation_sequences = tokenizer.texts_to_sequences(validation_texts)



train_sequences_padded = pad_sequences(train_sequences,maxlen=max_seq_len)

validation_sequences_padded = pad_sequences(validation_sequences,maxlen=max_seq_len)



word_index = tokenizer.word_index

print("total words in model", total_words)

print(len(word_index))
# Taken from:  https://www.kaggle.com/imvkhandelwal/tensorflow-2-0-rnn-with-glove-vectors

embeddings_index = {}

with open('../input/glove6b/glove.6B.100d.txt') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs



embeddings_matrix = np.zeros((len(word_index),100))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector
train = train_sequences_padded

train_target = train_targets

validation = validation_sequences_padded

validation_target = validation_targets
model = Sequential()

model.add(Embedding(len(word_index), 100, input_length=max_seq_len, weights=[embeddings_matrix], trainable=False))

model.add(Bidirectional(LSTM(64, return_sequences = True)))

model.add(Dropout(0.05))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(Dropout(0.05))

model.add(Bidirectional(LSTM(8, return_sequences = False)))

model.add(Dropout(0.05))



model.add(keras.layers.Dense(100))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(keras.layers.Dense(50))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))



model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train, train_target, epochs=1, validation_data=(validation, validation_target))
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_num =range(0,epochs_num)



plt.plot(1, acc, 'b', label='Training acc')

plt.plot(1, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(1, loss, 'b', label='Training loss')

plt.plot(1, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
# serialize model to JSON

import os

model_json = model.to_json()

os.mkdir("..output")

with open("..output/model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")