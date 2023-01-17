# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/glove-6b"))



# Any results you write to the current directory are saved as output.
import os, sys



import numpy as np



from keras.models import Model



from keras.layers import Input, Dense, Flatten

from keras.layers import Conv1D, MaxPooling1D

from keras.layers import Embedding



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.utils import to_categorical
docs = []          # list of text samples

labels = []        # list of label ids

labels_Index = {}  # dictionary mapping label index to label name



PATH = "../input/imdbdatacorpus/aclimdb_v1/aclImdb_v1/aclImdb/train"



TEXT_DATA_DIR = os.path.join(PATH)
for name in ['neg','pos']:

    path = os.path.join(TEXT_DATA_DIR, name)

    if os.path.isdir(path):

        label_Id = len(labels_Index)

        labels_Index[label_Id] = name

        for fname in sorted(os.listdir(path)):

            fpath = os.path.join(path, fname)

            f = open(fpath, encoding = "ISO-8859-1")

            t = f.read()

            docs.append(t)

            f.close()

            labels.append(label_Id)



print('Found %s docs.' % len(docs))
docs_test = []          # list of text samples

labels_test = []        # list of label ids

labels_Index_test = {}  # dictionary mapping label index to label name
PATH1 = "../input/imdbdatacorpus/aclimdb_v1/aclImdb_v1/aclImdb/test"
for name1 in ['neg','pos']:

    path = os.path.join(PATH1,name1)

    if os.path.isdir(path):

        label_Id1 = len(labels_Index_test)

        labels_Index_test[label_Id1] = name1

        for fname in sorted(os.listdir(path)):

            fpath = os.path.join(path, fname)

            f = open(fpath, encoding = "ISO-8859-1")

            t = f.read()

            docs_test.append(t)

            f.close()

            labels_test.append(label_Id1)



print('Found %s docs.' % len(docs_test))
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# Prepare tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(docs)



word_Index = tokenizer.word_index



vocab_Size = len(word_Index) + 1

print('Found %s unique tokens.' % vocab_Size)
# Prepare tokenizer

tokenizer_1 = Tokenizer()

tokenizer_1.fit_on_texts(docs_test)



word_Index_1 = tokenizer_1.word_index



vocab_Size_1 = len(word_Index_1) + 1

print('Found %s unique tokens.' % vocab_Size_1)
# integer encode the documents

sequences = tokenizer.texts_to_sequences(docs)

print(docs[1], sequences[1])

print("------------------------#################-------------------------")

sequences_test = tokenizer.texts_to_sequences(docs_test)

print(docs_test[1], sequences_test[1])

#for i in sequences:

#    print (len(i))
MAX_SEQUENCE_LENGTH = 1000



data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



print('Shape of data tensor:', data.shape)

print('Shape of data tensor:', data_test.shape)
# split the data into a training set and a test set

X_train = data



X_test = data_test



y_train = labels



y_test = labels_test
Y_train = to_categorical(y_train)

Y_test = to_categorical(y_test)
PATH_glove = "../input/glove-6b"



embeddings_index = {}

f = open(os.path.join(PATH_glove, 'glove.6B.50d.txt'), encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_Matrix = np.zeros((vocab_Size, 50))

for word, i in word_Index.items():

    embedding_Vector = embeddings_index.get(word)

    if embedding_Vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_Matrix[i] = embedding_Vector



print (embedding_Matrix.shape)
embedding_layer = Embedding(vocab_Size,

                            50,

                            weights=[embedding_Matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(64, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(4)(x)

x = Conv1D(64, 5, activation='relu')(x)

x = MaxPooling1D(4)(x)

x = Conv1D(64, 5, activation='relu')(x)

x = MaxPooling1D(4)(x)  # global max pooling

x = Flatten()(x)

x = Dense(64, activation='relu')(x)

x = Dense(36, activation='relu')(x)

preds = Dense(len(labels_Index), activation='softmax')(x)



model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
# summarize the model

print(model.summary())
model.fit(X_train, Y_train, epochs=3)
Y_pred = model.predict(X_test)

print(Y_pred)
y_pred =[]

for i in Y_pred:

    y_pred.append(np.argmax(i))



print(y_pred)
# calculate accuracy of class predictions

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
# print the confusion matrix

metrics.confusion_matrix(y_test, y_pred)
from keras.datasets import imdb #A utility to load a dataset



from keras.models import Sequential



from keras.layers import Dense, LSTM, Dropout, Embedding

from keras.layers import Conv1D, MaxPooling1D



from keras.preprocessing import sequence #To convert a variable length sentence into a prespecified length

embedding_vector_length = 36



model_LSTM = Sequential()



model_LSTM.add(Embedding(vocab_Size, embedding_vector_length, input_length=MAX_SEQUENCE_LENGTH))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(LSTM(100))

model_LSTM.add(Dropout(0.2))

model_LSTM.add(Dense(50, activation='relu'))

model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_LSTM.summary())
model_LSTM.fit(X_train, y_train, epochs=3, batch_size=256)
scores =model_LSTM.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
Y_pred =model_LSTM.predict(X_test)

print(Y_pred)