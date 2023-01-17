!pip show keras
!pip freeze > requirements.txt
import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# keras

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, BatchNormalization

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

# plt

import matplotlib.pyplot as plt

#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# seed values

SEED = 42

random.seed = SEED

np.random.seed(seed=SEED)
# MODEL

BATCH_SIZE  = 128

EPOCH       = 10

VAL_SPLIT   = 0.15  #15%



# TOKENIZER

# The maximum number of words to be used. (most frequent)

MAX_WORDS = 20000

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 150



DATA_PATH = '/kaggle/input/sf-dl-movie-genre-classification/'
train_df = pd.read_csv(DATA_PATH+'train.csv',)
train_df.head()
train_df.info()
train_df.genre.value_counts().plot(kind='bar',figsize=(12,4),fontsize=10)

plt.xticks(rotation=60)

plt.xlabel("Genres",fontsize=10)

plt.ylabel("Counts",fontsize=10)
test_df = pd.read_csv(DATA_PATH+'test.csv')

test_df.head()
Y = pd.get_dummies(train_df.genre)

CLASS_NUM = Y.shape[1]

print('Shape of label tensor:', Y.shape)
Y.head()
# данные у нас и так достаточно чистые
# для построения словаря мы используем весь текст

all_text = train_df.text.append(test_df.text, ignore_index=True)
%%time

tokenize = Tokenizer(num_words=MAX_WORDS)

tokenize.fit_on_texts(all_text)
%%time

sequences = tokenize.texts_to_sequences(train_df.text)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

print(sequences_matrix.shape)
# вот так теперь выглядит наш текст

print(train_df.text[1])

print(sequences_matrix[1])
def RNN():

    inputs = Input(name='inputs',shape=[MAX_SEQUENCE_LENGTH])

    layer = Embedding(MAX_WORDS,50,input_length=MAX_SEQUENCE_LENGTH)(inputs)

    layer = LSTM(100)(layer)

    layer = Dense(256, activation='relu', name='FC1')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(CLASS_NUM, activation='sigmoid', name='out_layer')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(sequences_matrix,Y,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCH,

                    validation_split=VAL_SPLIT)
model.save('keras_nlp_lstm.h5')
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.show();
plt.title('Accuracy')

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.show();
test_sequences = tokenize.texts_to_sequences(test_df.text)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH)
%%time

predict_proba = model.predict(test_sequences_matrix)
# на соревнованиях всегда сохраняйте predict_proba, чтоб потом можно было построить ансамбль решений

predict_proba = pd.DataFrame(predict_proba, columns=Y.columns)

predict_proba.to_csv('predict_proba.csv', index=False)

predict_proba.head()
predict_genre = Y.columns[np.argmax(predict_proba.values, axis=1)]
submission = pd.DataFrame({'id':range(1, len(predict_genre)+1), 

                           'genre':predict_genre}, 

                          columns=['id', 'genre'])



submission.to_csv('submission.csv', index=False)

submission.head()