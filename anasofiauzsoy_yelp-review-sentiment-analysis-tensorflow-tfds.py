# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tensorflow-datasets
import tensorflow as tf
import tensorflow_datasets as tfds
data = tfds.load('yelp_polarity_reviews', split='train', shuffle_files=True)
reviews = []
polarity = []

for i in data.take(20000):
    reviews.append((i['text'].numpy().decode("utf-8")))
    polarity.append(int(i['label']))
reviews[3]
def clean_text(review):
    cleaned = review.replace("\\n", " ")
    cleaned = cleaned.replace("\'", "'")
    cleaned = cleaned.replace("\\r", " ")
    cleaned = cleaned.replace("\\""", " ")
    return cleaned
reviews = [clean_text(review) for review in reviews]
reviews[3]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, max_length, padding = 'post')
padded_sequences
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(padded_sequences, polarity)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y)
len(train_x), len(train_y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length-1))
model.add(Bidirectional(LSTM(20, return_sequences = True)))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(np.array(train_x), np.array(train_y), epochs = 3, verbose = 1, 
          validation_data = (np.array(val_x), np.array(val_y)))
print("Accuracy: ", model.evaluate(np.array(test_x), np.array(test_y))[1])
tokenizer.sequences_to_texts([test_x[0]])
np.round(max(model.predict(test_x[0])))
