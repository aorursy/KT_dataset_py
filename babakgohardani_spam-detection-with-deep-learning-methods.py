# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# we need to fit model with sequence of tokens with specific length

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

# normal LSTM/GRU and the Version with Cuda

from keras.layers import Dense, Embedding, GRU, LSTM, Dropout, Bidirectional

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam, rmsprop



# keras wrapper for k-fold cross-validation

from keras.wrappers.scikit_learn import KerasClassifier

# normsl cross validation

from sklearn.model_selection import cross_val_score, train_test_split

# cross validation for hyperparameter tuning

from sklearn.model_selection import GridSearchCV



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x_raw = []

y_raw = []



with open("../input/spam.csv", encoding = "ISO-8859-1") as f:

    for line in f:

        y_raw.append(line.split()[0])

        x_raw.append(' '.join(i for i in line.split()[1:]))
y = [1 if i=='ham' else 0 for i in y_raw]



print(max(len(s) for s in x_raw))

print(min(len(s)for s in x_raw))

sorted_X = sorted(len(s) for s in x_raw)

print(sorted_X[len(sorted_X) // 2])
tokenizer = Tokenizer()

tokenizer.fit_on_texts(x_raw)

sequences = tokenizer.texts_to_sequences(x_raw)



vocab_size = len(tokenizer.word_index)+1

print(vocab_size)
# divide sum of length of all sequences by number of all sequences to find averge length of each sequence

sum([len(x) for x in sequences]) // len(sequences)
pad = 'post' 

max_len = 25

embedding_size = 100

batch_size = 20

sequences = pad_sequences(sequences, maxlen=max_len, padding=pad, truncating=pad)

sequences.shape



X_train, X_test, y_train, y_test = train_test_split(sequences, y, test_size = 0.2, random_state= 0)
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

model.add(Dropout(0.8))

model.add(LSTM(140, return_sequences=False))

model.add(Dropout(0.8))

model.add(Dense(1, activation='sigmoid', name='Classification'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#save_best = ModelCheckpoint('SpamDetection.hdf', save_best_only=True, monitor='val_acc', mode='max')

# callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# Uses Automatic Verification Datasets (fastest option)

# model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[callback_early_stopping])

n_epochs = 10

results = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# model.load_weights(filepath='SpamDetection.hdf')

eval_ = model.evaluate(X_test, y_test)

print(eval_[0], eval_[1]) # loss / accuracy
def plot_model(result):

    acc = result.history['acc']

    val_acc = result.history['val_acc']

    loss = result.history['loss']

    val_loss = result.history['val_loss']

    x = range(1, len(acc)+1)

    

    plt.figure(figsize=(12, 5))

    plt.subplot(1,2,1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label= 'Validation acc')

    plt.legend()

    

    plt.subplot(1,2,2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='validation loss')

    plt.legend()

    

plot_model(results)
model1 = Sequential()

model1.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

model1.add(Dropout(0.8))

model1.add(GRU(140, return_sequences=False))

model1.add(Dropout(0.86))

model1.add(Dense(1, activation='sigmoid', name='Classification'))

model1.summary()



model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

results1 = model1.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2)



eval_ = model1.evaluate(X_test, y_test)

print(eval_[0], eval_[1]) # loss / accuracy



plot_model(results1)
model2 = Sequential()

model2.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

model2.add(Dropout(0.8))

model2.add(Bidirectional(LSTM(140, return_sequences=False)))

model2.add(Dropout(0.8))

model2.add(Dense(1, activation='sigmoid', name='Classification'))

model2.summary()



model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



results2 = model2.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2)



eval_ = model2.evaluate(X_test, y_test)

print(eval_[0], eval_[1]) # loss / accuracy



plot_model(results2)
from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D



n_epochs = 10

model3 = Sequential()

model3.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

model3.add(Conv1D(128, 3, activation='relu'))

model3.add(MaxPool1D(3))

model3.add(Dropout(0.2))

model3.add(Conv1D(128, 3, activation='relu'))

model3.add(GlobalMaxPooling1D())

model3.add(Dropout(0.2))

model3.add(Dense(64, activation='relu'))

model3.add(Dropout(0.2))

model3.add(Dense(32, activation='relu'))

model3.add(Dropout(0.2))

model3.summary()

model3.add(Dense(1, activation='sigmoid'))





model3.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

result3 = model3.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, validation_split=0.2, verbose=1)

eval_ = model3.evaluate(X_test, y_test)

print(eval_[0], eval_[1]) # loss / accuracy



plot_model(result3)
# word2vec = {}

# f = open('glove.6B/glove.6B.100d.txt', encoding="utf-8")

# for line in f:

#     values = line.split()

#     word = values[0]

#     vec = np.asarray(values[1:], dtype='float32')

#     word2vec[word] = vec

    

# print(len(word2vec))



# embedding_matrix = np.zeros((vocab_size, embedding_size))



# for word,i in  tokenizer.word_index.items():

#     if i < vocab_size:

#         embedding_vector = word2vec.get(word)

#         if embedding_vector is not None:

#          # words not in the glove will be set to zero   

#             embedding_matrix[i] = embedding_vector

            



# n_epochs = 20

# model4 = Sequential()

# model4.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,weights = [embedding_matrix], input_length=max_len, trainable=True))

# model4.add(Conv1D(128, 3, activation='relu'))

# model4.add(MaxPool1D(3))

# model4.add(Dropout(0.2))

# model4.add(Conv1D(128, 3, activation='relu'))

# model4.add(GlobalMaxPooling1D())

# model4.add(Dropout(0.2))

# model4.add(Dense(64, activation='relu'))

# model4.add(Dropout(0.2))

# model4.add(Dense(32, activation='relu'))

# model4.add(Dropout(0.2))

# model4.add(Dense(16, activation='relu'))

# model4.add(Dropout(0.2))

# model4.summary()

# model4.add(Dense(1, activation='sigmoid'))





# model4.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# save_best = ModelCheckpoint('SMS.hdf', save_best_only=True, monitor='val_acc', mode='max')

# result4 = model4.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, validation_split=0.2, verbose=1,  callbacks=[save_best])

# eval_ = model4.evaluate(X_test, y_test)

# print(eval_[0], eval_[1]) # loss / accuracy



# plot_model(result4)