#import libraries

import numpy as np

import matplotlib.pyplot as plt

import os

import tensorflow as tf

import random

import pandas as pd

import os

print(os.listdir("../input"))
data_path = os.path.join("../input", 'train.tsv')

test_data_path = os.path.join("../input", 'test.tsv')

data = pd.read_csv(data_path, sep='\t')

test_data = pd.read_csv(test_data_path, sep='\t')

data.describe()
data.head()
from sklearn.model_selection import train_test_split

train_texts = data['Phrase']

train_labels = np.array(data['Sentiment'])



#train_labels = pd.get_dummies(train_labels)



test_texts = test_data['Phrase']

X_train, X_test, y_train, y_test = train_test_split(train_texts, train_labels, test_size=0.33, random_state=42)

y_temp = y_train

y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)



data = ((X_train, np.array(y_train)),(X_test, np.array(y_test)))
from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



NGRAM_RANGE = (1,3)

TOP_K = 9000

TOKEN_MODE = 'word'

MIN_DOCUMENT_FREQUENCY = 5
def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):

    """Vectorizes texts as n-gram vectors.

    # Arguments

        train_texts: list, training text strings.

        train_labels: np.ndarray, training labels.

        val_texts: list, validation text strings.



    # Returns

        x_train, x_val: vectorized training and validation texts

    """

    # Create keyword arguments to pass to the 'tf-idf' vectorizer.

    

        # Create keyword arguments to pass to the 'tf-idf' vectorizer.

    stop_words_lst = text.ENGLISH_STOP_WORDS.union(["\'s"])

    kwargs = {

            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.

            'dtype': 'int32',

            'strip_accents': 'unicode',

            'decode_error': 'replace',

            'analyzer': TOKEN_MODE,  # Split text into word tokens.

            'min_df': MIN_DOCUMENT_FREQUENCY,

            'stop_words' : stop_words_lst

    }

    vectorizer = TfidfVectorizer(**kwargs)



    # Learn vocabulary from training texts and vectorize training texts.

    x_train = vectorizer.fit_transform(train_texts)



    # Vectorize validation texts.

    x_val = vectorizer.transform(val_texts)

    

    # Vectorize test texts.

    x_test = vectorizer.transform(test_texts)

    

    # Select top 'k' of the vectorized features.

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))

    selector.fit(x_train, y_temp)

    

    

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    x_test = selector.transform(x_test).astype('float32')

    return x_train, x_val, x_test
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.layers import Dropout
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):

    """Creates an instance of a multi-layer perceptron model.



    # Arguments

        layers: int, number of `Dense` layers in the model.

        units: int, output dimension of the layers.

        dropout_rate: float, percentage of input to drop at Dropout layers.

        input_shape: tuple, shape of input to the model.

        num_classes: int, number of output classes.



    # Returns

        An MLP model instance.

    """

    print('input shape : ', input_shape)

    model = models.Sequential()

    model.add(Dropout(dropout_rate, input_shape=input_shape))

    i=0

    for _ in range(layers-1):

        model.add(Dense(units=units, activation='relu'))

        model.add(Dropout(rate=dropout_rate))

        pass

    

    model.add(Dense(units=64, activation='relu'))

    model.add(Dropout(rate=dropout_rate))

        

    model.add(Dense(units=num_classes, activation='softmax', name='d2'))

    return model
def train_ngram_model(X_train, 

                      X_label,

                      X_validate,

                      val_label,

                      learning_rate=1e-3,

                      epochs=50,

                      batch_size=150,

                      layers=3,

                      units=128,

                      dropout_rate=0.2):

    """Trains n-gram model on the given dataset.



    # Arguments

        data: tuples of training and test texts and labels.

        learning_rate: float, learning rate for training model.

        epochs: int, number of epochs.

        batch_size: int, number of samples per batch.

        layers: int, number of `Dense` layers in the model.

        units: int, output dimension of Dense layers in the model.

        dropout_rate: float: percentage of input to drop at Dropout layers.

    """

    num_classes = 5

    

    x_train, x_val = X_train, X_validate

    val_labels = val_label

    # Create model instance.

    model = mlp_model(layers=layers,

                      units=units,

                      dropout_rate=dropout_rate,

                      input_shape=x_train.shape[1:],

                      num_classes=num_classes)

    loss = 'categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

   

    callbacks = [tf.keras.callbacks.EarlyStopping(

        monitor='val_loss', patience=6)]

    # Train and validate model.

    history = model.fit(

            x_train,

            X_label,

            epochs=epochs,

            callbacks=callbacks,

            validation_data=(x_val, val_labels),

            verbose=2,  # Logs once per epoch.

            batch_size=batch_size)

    # Print results.

    history = history.history

    print('Validation accuracy: {acc}, loss: {loss}'.format(

            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    

    return model
(train_texts, train_labels), (val_texts, val_labels) = data

x_train, x_val, x_test = ngram_vectorize(train_texts, train_labels, val_texts, test_texts)

model = train_ngram_model(x_train, train_labels, x_val, val_labels)
#make prediction

y_test = model.predict(x_test)

y_class = np.argmax(y_test, axis=1)



#write output

my_submission = pd.DataFrame({'PhraseId': test_data.PhraseId, 'Sentiment': y_class})

my_submission.to_csv('submission.csv', index=False)