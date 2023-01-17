# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras as keras

from keras.models import Sequential, load_model

from keras.layers import LSTM, GRU,SimpleRNN

from keras.layers import Dense, Embedding, Bidirectional, Dropout, Flatten

from keras.optimizers import Adam, SGD

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
test.head()
print('Training Dataset contain {} samples'.format(train.shape[0]))

print('Testing Dataset contain {} samples'.format(test.shape[0]))
train = train.drop(['id', 'keyword', 'location'], axis=1)

test = test.drop(['id', 'keyword', 'location'], axis=1)
y_train =  train['target'].values

X_train = train.drop(['target'], axis=1).values.reshape(len(train),)

X_test = test['text'].values.reshape(len(test),)
print(X_test)
print(y_train)

print(X_train)
total_tweets = np.concatenate((X_train, X_test))

print('Total tweets : ', len(total_tweets))
total_tweets


tokenizer = Tokenizer()

tokenizer.fit_on_texts(total_tweets)



# Vocbvulary Size

#　trian、testのツイートの総単語数（単語の次元数）を取得

vocab_size = len(tokenizer.word_index) + 1

print('Size of Vocabulary : ', vocab_size)
# Maximum length for padding sequence

maxlen = max(len(x.split()) for x in total_tweets)

print('Maximum length of tweet : ', maxlen)
X_train_token = tokenizer.texts_to_sequences(X_train)

X_test_token = tokenizer.texts_to_sequences(X_test)



print('Text before tokenized')

print(X_train[0])

print('\nText after tokenized')

print(X_train_token[0])
X_train_pad = pad_sequences(X_train_token, maxlen=maxlen, padding='post')

X_test_pad = pad_sequences(X_test_token, maxlen=maxlen, padding='post')



print('Tokenized text before padding')

print(X_train_token[0])

print('\nTokenized text after padding')

print(X_train_pad[0])


embed_units=100

hidden_units=128



model=Sequential()

model.add(Embedding(vocab_size, embed_units, input_length = maxlen))

model.add(SimpleRNN(hidden_units))

model.add(Dropout(0.2))

#model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(1, activation='sigmoid'))



model.summary()

learning_rate = 0.0001



model.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
batch_size_1= 150

batch_size_2= 300

batch_size_3= 700

num_itr = 5

es_cb=keras.callbacks.EarlyStopping( patience=0, verbose=0)

model_history = model.fit(X_train_pad, y_train, 

                          batch_size=batch_size_1, 

                          epochs=num_itr, 

                          validation_split=0.3,

                          callbacks=[es_cb])




model_BRNN=Sequential()

model_BRNN.add(Embedding(vocab_size, embed_units, input_length = maxlen))

model_BRNN.add(Bidirectional(SimpleRNN(hidden_units)))

model_BRNN.add(Dropout(0.2))

#model.add(Flatten())

model_BRNN.add(Dense(256, activation='relu'))

model_BRNN.add(Dropout(0.2))

model_BRNN.add(Dense(1, activation='sigmoid'))



model_BRNN.summary()

learning_rate = 0.0001



model_BRNN.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
model_BRNN_history = model_BRNN.fit(X_train_pad, y_train, 

                          batch_size=batch_size_3, 

                          epochs=num_itr, 

                          validation_split=0.2,

                          callbacks=[es_cb])




model_LS = Sequential()

model_LS.add(Embedding(vocab_size, embed_units, input_length = maxlen))

model_LS.add(Bidirectional(LSTM(hidden_units)))

model_LS.add(Dropout(0.2))

#model.add(Flatten())

model_LS.add(Dense(256, activation='relu'))

model_LS.add(Dropout(0.2))

model_LS.add(Dense(1, activation='sigmoid'))



model_LS.summary()
learning_rate = 0.0001



model_LS.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
model_LS_history = model_LS.fit(X_train_pad, y_train, 

                          batch_size=batch_size_3, 

                          epochs=num_itr, 

                          validation_split=0.4,

                          callbacks=[es_cb])


model_BLS=Sequential()

model_BLS.add(Embedding(vocab_size, embed_units, input_length = maxlen))

model_BLS.add(Bidirectional(LSTM(hidden_units)))

model_BLS.add(Dropout(0.2))

#model.add(Flatten())

model_BLS.add(Dense(256, activation='relu'))

model_BLS.add(Dropout(0.2))

model_BLS.add(Dense(1, activation='sigmoid'))



model_BLS.summary()

learning_rate = 0.0001



model_BLS.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])


model_BLS_history = model_LS.fit(X_train_pad, y_train, 

                          batch_size=batch_size_3, 

                          epochs=num_itr, 

                          validation_split=0.9,

                          callbacks=[es_cb])
pred = model_LS.predict(X_test_pad)
sub = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sub["target"] = pred

sub["target"] = sub["target"].apply(lambda x : 0 if x<=.5 else 1)
sub
sub.to_csv("submit_5.csv", index=False)