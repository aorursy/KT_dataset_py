import re

import numpy as np

import string

from nltk.corpus import stopwords

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM
def clean_doc(doc):

  tokens = doc.split()

  re_punc = re.compile('[%s]' % re.escape(string.punctuation))

  tokens = [re_punc.sub("", w) for w in tokens]

  tokens = [word for word in tokens if word.isalpha()]

  stop_words = stopwords.words('english')

  tokens = [word for word in tokens if word not in stop_words]

  tokens = [word for word in tokens if len(word) > 1]

  tokens = " ".join(tokens)

  return tokens
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head()
train_docs = list(map(clean_doc, train_docs))

test_docs = list(map(clean_doc, test_docs))
tokenizer = Tokenizer(filters = "", lower = False)

tokenizer.fit_on_texts(train_docs)



vocab_size = len(tokenizer.word_index) + 1
train_sequence = tokenizer.texts_to_sequences(train_docs)

test_sequence = tokenizer.texts_to_sequences(test_docs)
max_length = max([len(i) for i in train_sequence])

print(max_length)



train_sequence = pad_sequences(train_sequence, maxlen = max_length)

test_sequence = pad_sequences(test_sequence, maxlen = max_length)



print(train_sequence.shape)

print(test_sequence.shape)
# TODO: Design your model

model = Sequential()

model.add(Embedding(vocab_size, 32, input_length = 23))

model.add(LSTM(100))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())
model.fit(train_sequence, train_df.target.tolist(), batch_size = 32, epochs = 4, verbose = 1, validation_split = 0.25)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
sample_submission["target"] = model.predict(test_sequence)

sample_submission['target'] = sample_submission['target'].apply(lambda x: 1 if x >= 0.5 else 0)
sample_submission.to_csv("submission.csv", index=False)