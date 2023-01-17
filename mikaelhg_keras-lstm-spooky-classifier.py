import numpy as np

import pandas as pd
NUM_WORDS = 10000

N = 128

MAX_LEN = 50

NUM_CLASSES = 3
from keras.layers import Embedding, LSTM, Dense, Flatten

from keras.models import Sequential



model = Sequential()

model.add(Embedding(NUM_WORDS, N, input_length=MAX_LEN))

model.add(LSTM(N, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(Flatten())

model.add(Dense(NUM_CLASSES, activation='softmax'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.sample(10)
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from sklearn import preprocessing



X = train['text']

Y = train['author']



tokenizer = Tokenizer(num_words=NUM_WORDS)

tokenizer.fit_on_texts(X)



train_x = tokenizer.texts_to_sequences(X)

train_x = pad_sequences(train_x, maxlen=MAX_LEN)



lb = preprocessing.LabelBinarizer()

lb.fit(Y)



train_y = lb.transform(Y)
model.fit(train_x, train_y, validation_split=0.2, batch_size=1024, epochs=8, verbose=2)
score = model.evaluate(train_x, train_y, batch_size=1024, verbose=2)

print(score)
p = model.predict(pad_sequences(tokenizer.texts_to_sequences(test['text']), maxlen=MAX_LEN),

                  batch_size=1024)



for i in range(10):

    row = p[i]

    print(TX[i])

    for j in range(len(lb.classes_)):

        print('{0:>5} {1:02.2f}'.format(lb.classes_[j], row[j]))

    print()
import pickle, h5py

model.save('spooky_model.hdf5')

with open('tokenizer.pickle', 'wb') as f:

    pickle.dump(tokenizer, f)

with open('binarizer.pickle', 'wb') as f:

    pickle.dump(lb, f)