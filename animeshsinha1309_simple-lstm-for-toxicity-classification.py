# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import keras

from matplotlib import pyplot as plt



INPUT_DIR = "../input/jigsaw-unintended-bias-in-toxicity-classification"

GLOVE_DIR = "../input/glove-global-vectors-for-word-representation"



print(os.listdir(INPUT_DIR))

print(os.listdir(GLOVE_DIR))
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))

train.head()
with open(os.path.join(INPUT_DIR, 'sample_submission.csv')) as sample_submission:

    for x in range(5):

        print(next(sample_submission), end='')
with open(os.path.join(INPUT_DIR, 'test.csv')) as sample_submission:

    for x in range(10):

        print(next(sample_submission), end='')
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as glove_file:

    for x in range(5):

        print(next(glove_file))
embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))

print('Embeddings_index is a map of the words to a', len(embeddings_index['the']), 'dimentional vector.')
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# Generating the Text corpus in the form of a Numpy list

corpus = train['comment_text'].tolist()

print("Some sample comments we train the Tokenizer on:\n", corpus[:3])



# Fitting the tokenizer on the corpus, 

tokenizer = Tokenizer(num_words=1000000)

tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



# Padding to convert Jagged array into uniform length 2-D time series data

data = pad_sequences(sequences)
labels = train['target'].as_matrix()

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)



# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]



# Split into Train and Validation Sets

VALIDATION_SPLIT = 0.25 # Percentage of sample going to the Validation set

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]
EMBEDDING_DIM = 100

MAX_SEQUENCE_LENGTH = len(data[0])



embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,

                                            EMBEDDING_DIM,

                                            weights=[embedding_matrix],

                                            input_length=MAX_SEQUENCE_LENGTH,

                                            trainable=False)
def build_model():

    words = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,))

    x = embedding_layer(words)

    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True))(x)



    hidden = tf.keras.layers.concatenate([

        tf.keras.layers.GlobalMaxPooling1D()(x),

        tf.keras.layers.GlobalAveragePooling1D()(x),

    ])

    hidden = tf.keras.layers.add([hidden, tf.keras.layers.Dense(512, activation='relu')(hidden)])

    hidden = tf.keras.layers.add([hidden, tf.keras.layers.Dense(512, activation='relu')(hidden)])

    result = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)

    

    model = tf.keras.models.Model(inputs=words, outputs=result)

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
model = build_model()

history = model.fit(x = x_train, y = y_train, validation_data=(x_val, y_val), epochs = 2)
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model.png')
print(history.history.keys())

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('The Loss Function')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

test.head()
questions = test['comment_text'].tolist()

q_data = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=MAX_SEQUENCE_LENGTH)

print(q_data.shape)
result = model.predict(q_data)

ids = test['id'].tolist()
assert len(result) == len(ids)

with open('submission.csv', 'w') as file:

    file.write('id,prediction\n')

    for item in range(len(ids)):

        file.write(str(ids[item]) + ',' + str(result[item][0]) + '\n')
with open('submission.csv') as sample_submission:

    for x in range(5):

        print(next(sample_submission), end='')