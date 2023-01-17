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
from __future__ import absolute_import, division, print_function, unicode_literals

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

sns.set(rc={'figure.figsize' : (12, 6)})

sns.set_style("darkgrid", {'axes.grid' : True})

import skimage



# Импортируем TensorFlow и tf.keras

import tensorflow as tf

from tensorflow import keras
data = pd.read_csv('../input/IMDB Dataset.csv')

data.head()
data.shape
data.info()
# Number of poitive and negative reviews

data.sentiment.value_counts()
# Lets encode labels: each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

data.head()
# Now, let's see the average number of words per sample

plt.figure(figsize=(10, 6))

plt.hist([len(sample) for sample in list(data['review'])], 50)

plt.xlabel('Length of samples')

plt.ylabel('Number of samples')

plt.title('Sample length distribution')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

# So, we get such structure:

#        | word1  | word2  |  word3 | word4

# text1  |   1    |    1   |   1    |   0

# text2  |   0    |    1   |   1    |   0

# text3  |   2    |    1   |   0    |   0

# text4  |   0    |    0   |   0    |   1

vect_texts = vectorizer.fit_transform(list(data['review']))

# ['word1', 'word2', 'word3', 'word4']

all_ngrams = vectorizer.get_feature_names()

num_ngrams = min(50, len(all_ngrams))

all_counts = vect_texts.sum(axis=0).tolist()[0]



all_ngrams, all_counts = zip(*[(n, c) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)])

ngrams = all_ngrams[:num_ngrams]

counts = all_counts[:num_ngrams]



idx = np.arange(num_ngrams)



# Let's now plot a frequency distribution plot of the most seen words in the corpus.

plt.figure(figsize=(30, 30))

plt.bar(idx, counts, width=0.8)

plt.xlabel('N-grams')

plt.ylabel('Frequencies')

plt.title('Frequency distribution of ngrams')

plt.xticks(idx, ngrams, rotation=45)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 2)

TOP_K = 20000

TOKEN_MODE = 'word'

MIN_DOC_FREQ = 2



def ngram_vectorize(texts, labels):

    kwargs = {

        'ngram_range' : NGRAM_RANGE,

        'dtype' : 'int32',

        'strip_accents' : 'unicode',

        'decode_error' : 'replace',

        'analyzer' : TOKEN_MODE,

        'min_df' : MIN_DOC_FREQ,

    }

    # Learn Vocab from train texts and vectorize train and val sets

    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    transformed_texts = tfidf_vectorizer.fit_transform(texts)

    

    # Select best k features, with feature importance measured by f_classif

    # Set k as 20000 or (if number of ngrams is less) number of ngrams   

    selector = SelectKBest(f_classif, k=min(TOP_K, transformed_texts.shape[1]))

    selector.fit(transformed_texts, labels)

    transformed_texts = selector.transform(transformed_texts).astype('float32')

    return transformed_texts

# Vectorize the data

vect_data = ngram_vectorize(data['review'], data['sentiment'])
vect_data.shape
tfidf = TfidfVectorizer()

tr_texts = tfidf.fit_transform(data['review'])

tr_texts.shape
from sklearn.model_selection import train_test_split



# Split data to target (y) and features (X)

X = vect_data.toarray()

y = (np.array(data['sentiment']))



# Here we split data to training and testing parts

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print("Train dataset shape: {0}, \nTest dataset shape: {1}".format(X_train.shape, X_test.shape))
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.layers import Dropout
# First, let's create a function that returns the appropriate number of units and the activation for the last layer.

def get_last_layer_units_and_activation(num_classes):

    if num_classes == 2:

        activation = 'sigmoid'

        units = 1

    else:

        activation = 'softmax'

        units = num_classes

    return units, activation
# input shape is the vocabulary count used for the movie reviews (10,000 words)

DROPOUT_RATE = 0.2

UNITS = 64

NUM_CLASSES = 2

LAYERS = 2

input_shape = X_train.shape[1:]



op_units, op_activation = get_last_layer_units_and_activation(NUM_CLASSES)



model = keras.Sequential()

# Applies Dropout to the input

model.add(Dropout(rate=DROPOUT_RATE, input_shape=input_shape))

for _ in range(LAYERS-1):

    model.add(Dense(units=UNITS, activation='relu'))

    model.add(Dropout(rate=DROPOUT_RATE))

    

model.add(Dense(units=op_units, activation=op_activation))

model.summary()
LEARNING_RATE = 1e-3



# Compile model with parameters

if NUM_CLASSES == 2:

    loss = 'binary_crossentropy'

else:

    loss = 'sparse_categorical_crossentropy'

optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
EPOCHS = 100

BATCH_SIZE = 128



# Create callback for early stopping on validation loss. If the loss does

# not decrease on two consecutive tries, stop training

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]



# Train and validate model

# To start training, call the model.fit method—the model is "fit" to the training data.

# Note that fit() will return a History object which we can use to plot training vs. validation accuracy and loss.

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), verbose=1, batch_size=BATCH_SIZE, callbacks=callbacks)
# Next, compare how the model performs on the test dataset:

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print('Test loss:', test_loss)

print('Test accuracy:', test_acc)
# Let's plot training and validation accuracy as well as loss.

def plot_history(history):

    accuracy = history.history['acc']

    val_accuracy = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    epochs = range(1,len(accuracy) + 1)

    

    # Plot accuracy  

    plt.figure(1)

    plt.plot(epochs, accuracy, 'b', label='Training accuracy')

    plt.plot(epochs, val_accuracy, 'g', label='Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    

    # Plot loss

    plt.figure(2)

    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'g', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()



plot_history(history)
 # Save model

model.save('IMDB_model_dropout_nn.h5')