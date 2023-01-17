# # Char-level Recurrent Neural Network

# 

# Let's try and build an LSTM-based neural network that uses chars instead of words as its input.





import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

from keras.models import Sequential

from keras.layers import LSTM, Dense, Bidirectional, BatchNormalization, Dropout

from keras import optimizers

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, EarlyStopping

import re

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/imdb-reviews/dataset.csv",encoding ='latin1')

#train.index = train['id']

x_train = train['SentimentText'].apply(lambda x : re.sub("\s", " ", re.sub("[^a-zA-z,\.\s]", "", x)).lower())

y_train = train.iloc[:, 1:]

x_train.head()
y_train.head()
x_train.apply(lambda x: len(x)).describe()
unique_symbols = Counter()



for _, message in x_train.iteritems():

    unique_symbols.update(message)

    

print("Unique symbols:", len(unique_symbols))
# some parameters

BATCH_SIZE = 512  # batch size for the network

EPOCH_NUMBER = 100  # number of epochs to train

SENTENCE_LEN = 1000
num_unique_symbols = len(unique_symbols)



tokenizer = Tokenizer(

    char_level=True,

    filters=None,

    lower=False,

    num_words=num_unique_symbols

)



tokenizer.fit_on_texts(x_train)

sequences = tokenizer.texts_to_sequences(x_train)



# Pad the input: I use the 500 lenght, just a bit over the median length.



padded_sequences = pad_sequences(sequences, maxlen=SENTENCE_LEN)



# I will take just a bit of the data as the validation set to see that the network converges:



x_train, x_test, y_train, y_test = train_test_split(padded_sequences, y_train, stratify=y_train['Sentiment'], test_size=0.2)



# So, let's define the model!

x_test = to_categorical(np.array(x_test), num_classes=num_unique_symbols)

x_train = to_categorical(x_train, num_classes=num_unique_symbols)

model = Sequential()

model.add(LSTM(150, input_shape=(SENTENCE_LEN, num_unique_symbols), activation="tanh", return_sequences=True))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(LSTM(100, input_shape=(SENTENCE_LEN, num_unique_symbols), activation="tanh", return_sequences=True))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(LSTM(30, input_shape=(SENTENCE_LEN, num_unique_symbols), activation="tanh"))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(100, activation="tanh"))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(50, activation="tanh"))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(1, activation="sigmoid"))



model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

filepath = "saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)

callbacks_list = [checkpoint, es]



history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs = EPOCH_NUMBER, validation_split = 0.1, callbacks=callbacks_list)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
scores = model.evaluate(x_test, y_test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("model.hdf5")