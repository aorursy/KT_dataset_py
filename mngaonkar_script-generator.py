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
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, TimeDistributed, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_WORDS = 50000
INPUT_LENGTH = 19
# s1 = open('/kaggle/input/scifi-stories-text-corpus/internet_archive_scifi_v3.txt', 'rt').read()
# s1 = s1.replace('\n', ' ')
# s1 = s1.split('.')

# print(s1[100])
# read dialogs
s1 = []
for index in range(1,2):
  df = pd.read_json('/kaggle/input/game-of-thrones-srt/season%d.json'%(index))

  filter = '@'

  for episode in range(len(df.columns)):
      e = df[df.columns[episode]].dropna().sort_index()
      dialogs = list(e.values)
      dialogs = [x.replace('.', ' .') for x in dialogs]
      dialogs = [x.replace('?', ' ?') for x in dialogs]
      dialogs = [x.replace(',', ' ,') for x in dialogs]
      dialogs = [x.replace(':', ' :') for x in dialogs]
      s1 = s1 + dialogs

print("total lines = ", len(s1))
print(s1)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(s1)
seq = tokenizer.texts_to_sequences(s1)
print(seq[:10])
# print(tokenizer.word_index)

# free up memory
s1 = None
corpus = [subitem for item in seq for subitem in item]
print("corpus word length = ", len(corpus))
vocab_size = len(tokenizer.word_index)
print('vocab size = ', vocab_size)
sentence_len = 20
prediction_len = 1
train_len = sentence_len - prediction_len

train_seq = []
for item in range(len(corpus) - sentence_len):
    train_seq.append(corpus[item:item + sentence_len])
    
# free up corpus
corpus = None
trainX = []
trainy = []
for i in train_seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])

# free up train sequence data
train_seq = None
model = Sequential([
    Embedding(vocab_size + 1, 50, input_length=train_len),
    LSTM(128),
    # Dense(150, activation='relu'),
    Dense(3942),
    Activation('softmax')
])
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.asarray(trainX), pd.get_dummies(np.asarray(trainy)), batch_size=64, epochs=50, validation_split=0.2)
model.save('/kaggle/working/model_weights.hdf5')
model.load_weights('/kaggle/working/model_weights.hdf5')
token_to_word_map = dict(map(reversed, tokenizer.word_index.items()))

def generate_text(input_text, prediction_length):
    tokens = tokenizer.texts_to_sequences([input_text])

    while len(tokens[0]) < prediction_length:
        if len(tokens[0]) <= INPUT_LENGTH:
            padded_tokens = pad_sequences(tokens[-INPUT_LENGTH:], maxlen=INPUT_LENGTH)
        else:
            padded_tokens = [tokens[0][-INPUT_LENGTH:]]

        prediction = model.predict(np.asarray(padded_tokens).reshape(1,-1))
        tokens[0].append(prediction.argmax())
        
    tokens[0] = [134 if x==0 else x for x in tokens[0]]

    generated_text = " ".join(map(lambda x : token_to_word_map[x], tokens[0]))
    generated_text = generated_text.replace(' .', '.')

    return generated_text
print(generate_text("king in the north", 200))
