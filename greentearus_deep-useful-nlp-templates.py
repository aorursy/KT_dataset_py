import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import spacy
test = pd.read_csv("../input/imdb-sentiments/test.csv")

train = pd.read_csv("../input/imdb-sentiments/train.csv")



X = train['text']

y = train['sentiment']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=273, stratify=y)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence



word_index = 0



max_features = 1000

# cut texts after this number of words (among top max_features most common words)

maxlen = 80

batch_size = 500



#samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#samples = X_train



tokenizer = Tokenizer(num_words=max_features)



#build the vocab. :

tokenizer.fit_on_texts(X_train)



x_tn = tokenizer.texts_to_sequences(X_train)

x_tt = tokenizer.texts_to_sequences(X_test)



x_tn = sequence.pad_sequences(x_tn, maxlen=maxlen)

x_tt = sequence.pad_sequences(x_tt, maxlen=maxlen)



#convert list of strings to matrix encoding (binary == one-hot, count == BoW, tfidf == tfidf, freq == tf):

x_tn = tokenizer.texts_to_matrix(X_train, mode='binary') 

x_tt = tokenizer.texts_to_matrix(X_test, mode='binary') 

#< better use sequences coz more flexible padding options and consistent with common tutorials



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

#tokenizer.get_config()
np.shape(x_tn)
max_features = 1000

# cut texts after this number of words (among top max_features most common words)

maxlen = 80

batch_size = 20
from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import LSTM



model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(max_features,)))

model.add(Dense(16, activation='relu', input_shape=(max_features,)))

#model.add(Dense(16, activation='relu', input_shape=(max_features,)))

#model.add(Embedding(max_features, 128))

#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(x_tn, y_train,

          batch_size=batch_size,

          epochs=3,

          validation_data=(x_tt, y_test))
from sklearn.metrics import classification_report

y_pred = model.predict_classes(x_tt)

print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt





history_dict = history.history

history_dict.keys()



acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']



epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence



word_index = 0



max_features = 20000

# cut texts after this number of words (among top max_features most common words)

maxlen = 500

batch_size = 100



#samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#samples = X_train



tokenizer = Tokenizer(num_words=max_features)



#build the vocab. :

tokenizer.fit_on_texts(X_train)



x_tn = tokenizer.texts_to_sequences(X_train)

x_tt = tokenizer.texts_to_sequences(X_test)



x_tn = sequence.pad_sequences(x_tn, maxlen=maxlen)

x_tt = sequence.pad_sequences(x_tt, maxlen=maxlen)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
model = Sequential()

model.add(Embedding(max_features, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



history = model.fit(x_tn, y_train,

          batch_size=batch_size,

          epochs=3,

          validation_data=(x_tt, y_test))
from sklearn.metrics import classification_report

y_pred = model.predict_classes(x_tt)

print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt





history_dict = history.history

history_dict.keys()



acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']



epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence



word_index = 0



max_features = 20000

# cut texts after this number of words (among top max_features most common words)

maxlen = 500

batch_size = 100



tokenizer = Tokenizer(num_words=max_features)



#build the vocab. :

tokenizer.fit_on_texts(X_train)



x_tn = tokenizer.texts_to_sequences(X_train)

x_tt = tokenizer.texts_to_sequences(X_test)



x_tn = sequence.pad_sequences(x_tn, maxlen=maxlen)

x_tt = sequence.pad_sequences(x_tt, maxlen=maxlen)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.datasets import imdb



# set parameters:

max_features = 5000

maxlen = 400

batch_size = 32

embedding_dims = 50

filters = 250

kernel_size = 3

hidden_dims = 250

epochs = 2



tokenizer = Tokenizer(num_words=max_features)



#build the vocab. :

tokenizer.fit_on_texts(X_train)



x_tn = tokenizer.texts_to_sequences(X_train)

x_tt = tokenizer.texts_to_sequences(X_test)



x_tn = sequence.pad_sequences(x_tn, maxlen=maxlen)

x_tt = sequence.pad_sequences(x_tt, maxlen=maxlen)



model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(max_features,

                    embedding_dims,

                    input_length=maxlen))

model.add(Dropout(0.2))



# we add a Convolution1D, which will learn filters

# word group filters of size filter_length:

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

# we use max pooling:

model.add(GlobalMaxPooling1D())



# We add a vanilla hidden layer:

model.add(Dense(hidden_dims))

model.add(Dropout(0.2))

model.add(Activation('relu'))



# We project onto a single unit output layer, and squash it with a sigmoid:

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])





history = model.fit(x_tn, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_tt, y_test))
from sklearn.metrics import classification_report

y_pred = model.predict_classes(x_tt)

print(classification_report(y_test, y_pred))
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM, Bidirectional

from keras.layers import Conv1D, MaxPooling1D

from keras.datasets import imdb



# Embedding

max_features = 20000

maxlen = 100

embedding_size = 128



# Convolution

kernel_size = 5

filters = 64

pool_size = 4



# LSTM

lstm_output_size = 70



# Training

batch_size = 30

epochs = 2



'''

Note:

batch_size is highly sensitive.

Only 2 epochs are needed as the dataset is very small.

'''



tokenizer = Tokenizer(num_words=max_features)



#build the vocab. :

tokenizer.fit_on_texts(X_train)



x_tn = tokenizer.texts_to_sequences(X_train)

x_tt = tokenizer.texts_to_sequences(X_test)



x_tn = sequence.pad_sequences(x_tn, maxlen=maxlen)

x_tt = sequence.pad_sequences(x_tt, maxlen=maxlen)





model = Sequential()

model.add(Embedding(max_features, embedding_size, input_length=maxlen))

model.add(Dropout(0.25))

model.add(Conv1D(filters,

                 kernel_size,

                 padding='valid',

                 activation='relu',

                 strides=1))

model.add(MaxPooling1D(pool_size=pool_size))

model.add(LSTM(lstm_output_size))

#model.add(Bidirectional(LSTM(lstm_output_size)))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])





history = model.fit(x_tn, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_tt, y_test))
from sklearn.metrics import classification_report

y_pred = model.predict_classes(x_tt)

print(classification_report(y_test, y_pred))