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

import numpy as np

from keras.preprocessing.text import Tokenizer
# Load Data

df = pd.read_csv('/kaggle/input/phish-reviews/Reviews.csv')

data = df.Reviews.tolist()
# Text Preprocessing helper funcs + format reviews

import re

def format_review(review):

    """Add spaces around punctuation and remove references to images/citations."""



    # Add spaces around punctuation

    review = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', review)



    # Remove references to figures

    review = re.sub(r'\((\d+)\)', r'', review)



    # Remove double spaces

    review = re.sub(r'\s\s', ' ', review)

    return review





def remove_spaces(patent):

    """Remove spaces around punctuation"""

    patent = re.sub(r'\s+([.,;?])', r'\1', patent)



    return patent



formatted_data = list(map(format_review, data))
def generate_tokenized_sequences(texts, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):

    # Fit tokenizer

    tk = Tokenizer(lower=lower, filters=filters)

    tk.fit_on_texts(texts)

    

    # Save Index

    index = tk.index_word

    

    # Generate Sequences

    sequences = tk.texts_to_sequences(texts)

    

    return sequences, index
sequences, index = generate_tokenized_sequences(formatted_data)

num_words = len(index) + 1
df['Sequences'] = sequences

df.drop('Reviews', inplace=True, axis=1)
def find_max_sequence_length(seqs):

    lengths = []

    for s in seqs:

        lengths.append(len(s))

    return max(lengths)



def pad_sequences(seqs, max_length):

    output = []

    for slist in seqs:

        arr = np.zeros(max_length)

        for i, element in np.ndenumerate(slist):

            arr[i] = element

        output.append(arr)

    return output
max_length = find_max_sequence_length(sequences)

padded_sequences = pad_sequences(sequences, max_length)

df.Sequences = pd.Series(padded_sequences)
# Train-Test Partition

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(df.Sequences, df.Score, test_size=0.15)

X_train = X_train.values

Y_train = Y_train.apply(float).values
n_samples = len(Y_train)

sample_size = max_length

n_channels = 1



def reshape_X(arr):

    arr = np.array([element for element in arr])

    return np.expand_dims(arr, axis=2)



X_train = reshape_X(X_train)

X_test = reshape_X(X_test)
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten

from keras.models import Sequential
# Y_train engineering

Y_train = Y_train * 100
# Define Model



EMBEDDING_DIM = 32

FILTERS = 200

POOL_SIZE = 3

DROPOUT_RATE = 0.2



Model = Sequential()

Model.add(Conv1D(input_shape=(sample_size, 1),kernel_size=13, filters=FILTERS, activation='relu'))

Model.add(MaxPooling1D(POOL_SIZE))

Model.add(Dense(128, activation='relu'))

Model.add(Dropout(0.2))

Model.add(Dense(64, activation='relu'))

Model.add(Dropout(0.2))

Model.add(Flatten())

Model.add(Dense(1, activation='linear'))

Model.summary()



Model.compile(optimizer='Adam', loss=['mse'], metrics=['mae', 'mse'])
Model.fit(X_train, Y_train, epochs=5, validation_split=0.1, shuffle=True)
# Make Predictions

preds = (Model.predict(X_test, verbose=1)).flatten() / 100

obs = Y_test.values
# Model Error

errors = obs - preds

print(np.sqrt(np.mean(errors*errors)))

print(np.mean(np.absolute(errors/obs)))
#Benchmark Error

p = np.mean(Y_train.flatten() / 100)

bench_pred = np.array([p for element in obs])

bench_error = obs - bench_pred

print(np.sqrt(np.mean(bench_error*bench_error)))

print(np.mean(np.absolute(bench_error/obs)))