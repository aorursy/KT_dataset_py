import numpy as np

import pandas as pd

import os

import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text
#clean train and test data

train_df['comment_text'] = train_df.comment_text.apply(lambda x: clean_text(x))

test_df['comment_text'] = test_df.comment_text.apply(lambda x: clean_text(x))
#Setting values for model to training using Essay Text

X = train_df['comment_text']

y = train_df['target']
MAX_VOCAB_SIZE = 20000

MAX_SEQUENCE_LENGTH = 40

EMBEDDING_DIM = 300
print('Loading word vectors...')



word2vec = {}

with open(os.path.join('../input/glove6b/glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf8") as f:

  # is just a space-separated text file in the format:

  # word vec[0] vec[1] vec[2] ...

  for line in f:

    values = line.split()

    word = values[0]

    vec = np.asarray(values[1:], dtype='float32')

    word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(X)

list_tokenized_train = tokenizer.texts_to_sequences(X)

X_t = pad_sequences(list_tokenized_train, maxlen=MAX_SEQUENCE_LENGTH)
# prepare embedding matrix

word2idx = tokenizer.word_index

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():

  if i < MAX_VOCAB_SIZE:

    embedding_vector = word2vec.get(word)

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

      embedding_matrix[i] = embedding_vector
train_df[train_df['target']>.80][['target','comment_text']].count()
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.utils import to_categorical



model = Sequential()

model.add(Embedding(

  num_words,

  EMBEDDING_DIM,

  weights=[embedding_matrix],

  input_length=MAX_SEQUENCE_LENGTH,

  trainable=False

))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(1, activation="sigmoid"))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 512

epochs = 1

model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
list_tokenized_test = tokenizer.texts_to_sequences(test_df['comment_text'].values)

X_te = pad_sequences(list_tokenized_test, maxlen=MAX_SEQUENCE_LENGTH)

prediction = model.predict(X_te)

prediction=pd.DataFrame(prediction)

prediction.columns=['prediction']

test_df=pd.concat([test_df,prediction],axis=1)
my_submission = pd.DataFrame({'id': test_df.id,'prediction':test_df.prediction})

my_submission.to_csv('submission.csv', index=False)