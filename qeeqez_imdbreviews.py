import pyprind

import pandas as pd

from string import punctuation

import re

import numpy as np

df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv', encoding='utf-8')
from collections import Counter

counts = Counter()

pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrences')

for i,review in enumerate(df['review']):

    text=''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()

    df.loc[i,'review'] = text

    pbar.update()

    counts.update(text.split())



## Create a mapping

## Map each unique word to an integer

word_counts = sorted(counts, key=counts.get, reverse=True)

print(word_counts[:5])

word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}



mapped_reviews = []

pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')

for review in df['review']:

    mapped_reviews.append([word_to_int[word] for word in review.split()])

    pbar.update()
## Define same-length sequences

## if sequence length < 200: left-pad with zeros

## if sequence length > 200: use the last 200 elements

sequence_length = 200

sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):

    review_arr = np.array(row)

    sequences[i, -len(row):] = review_arr[-sequence_length:]
X_train = sequences[:25000,:]

y_train = df.loc[:24999, 'sentiment'].values

y_train[y_train=='positive'] = 1

y_train[y_train=='negative'] = 0

X_test = sequences[25000:,:]

y_test = df.loc[25000:, 'sentiment'].values

y_test[y_test=='positive'] = 1

y_test[y_test=='negative'] = 0
print(f"Train: {len(X_train)}; Test: {len(X_test)}")
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, Conv2D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers

n_words = max(list(word_to_int.values())) + 1

mbed_size = 200



def create_model():

    model = Sequential()

    model.add(Embedding(n_words, mbed_size))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation="relu"))

    model.add(Dropout(0.05))

    model.add(Dense(1, activation="sigmoid"))

    

    model.compile(

        loss='binary_crossentropy', 

        optimizer='adam', 

        metrics=['accuracy']

    )

    

    return model



model = create_model()

model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model, show_shapes=True, show_layer_names=False, 

                 rankdir='LR').create(prog='dot', format='svg'))
batch_size = 100

epochs = 3



model.fit(

    X_train,

    y_train, 

    batch_size=batch_size, 

    epochs=epochs, 

    validation_split=0.3

)
import string

print("punctuation:",string.punctuation)

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

print("stopwords:",set(stopwords.words("english")))

lines = df["review"].values.tolist()

stop_words = set(stopwords.words("english"))

reviews = list()

for line in lines:

    tokens = word_tokenize(line)

    tokens = [w.lower() for w in tokens]

    table = str.maketrans("","",string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    words = [w for w in stripped if w.isalpha()]

    words = [w for w in words if w not in stop_words]

    reviews.append(words)

len(reviews)    
import gensim

model = gensim.models.Word2Vec(

    sentences=reviews,

    size=mbed_size,

    window=5,

    workers=4,

    min_count=1

)
words=list(model.wv.vocab)

print("vocabulary size:",len(words))

model.wv.get_vector(words[1])
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer()

tokenizer.fit_on_texts(reviews)

seqs = tokenizer.texts_to_sequences(reviews)

review_pad = pad_sequences(seqs,padding="post")

word_index = tokenizer.word_index

sentiments = df["sentiment"].values



num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words,mbed_size))

for word,i in word_index.items():

    vector = model.wv.get_vector(word)

    if vector is not None:

        embedding_matrix[i] = vector
def create_model():

    model = Sequential()

    model.add(Embedding(len(embedding_matrix), mbed_size))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation="relu"))

    model.add(Dropout(0.05))

    model.add(Dense(1, activation="sigmoid"))

    

    model.layers[0].set_weights([embedding_matrix])

    

    model.compile(

        loss='binary_crossentropy', 

        optimizer='adam', 

        metrics=['accuracy']

    )

    

    return model



model = create_model()

model.summary()
batch_size = 100

epochs = 3



model.fit(

    review_pad,

    sentiments,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.3

)
def create_model():

    model = Sequential()

    model.add(Embedding(len(embedding_matrix), mbed_size))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation="relu"))

    model.add(Dropout(0.05))

    model.add(Dense(1, activation="sigmoid"))

    

    model.layers[0].set_weights([embedding_matrix])

    

    model.compile(

        loss='binary_crossentropy', 

        optimizer='adam',  

        metrics=['accuracy']

    )

    

    return model



model = create_model()

model.summary()
batch_size = 100

epochs = 3



model.fit(

    review_pad,

    sentiments,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.3

)
def create_model():

    model = Sequential()

    model.add(Embedding(len(embedding_matrix), mbed_size))

    model.add(Bidirectional(LSTM(32, return_sequences = True)))

    model.add(Bidirectional(LSTM(64, return_sequences = True)))

    model.add(GlobalMaxPool1D())

    model.add(Dense(20, activation="relu"))

    model.add(Dropout(0.05))

    model.add(Dense(1, activation="sigmoid"))

    

    model.layers[0].set_weights([embedding_matrix])

    

    model.compile(

        loss='binary_crossentropy', 

        optimizer='adam',  

        metrics=['accuracy']

    )

    

    return model



model = create_model()

model.summary()
batch_size = 100

epochs = 1



model.fit(

    review_pad,

    sentiments,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=0.3

)