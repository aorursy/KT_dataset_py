import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

import random as rn



from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.layers import Dropout

from tensorflow import set_random_seed

import torch as tc



path = './../input/aclimdb/'

np.random.seed(1)

set_random_seed(1)

tc.manual_seed(1)

rn.seed(1)
def shuffle(X, y):

    perm = np.random.permutation(len(X))

    X = X[perm]

    y = y[perm]

    return X, y
def load_imdb_dataset(path):

    imdb_path = os.path.join(path, 'aclImdb')



    

    train_texts = []

    train_labels = []

    test_texts = []

    test_labels = []

    for dset in ['train', 'test']:

        for cat in ['pos', 'neg']:

            dset_path = os.path.join(imdb_path, dset, cat)

            for fname in sorted(os.listdir(dset_path)):

                if fname.endswith('.txt'):

                    with open(os.path.join(dset_path, fname)) as f:

                        if dset == 'train': train_texts.append(f.read())

                        else: test_texts.append(f.read())

                    label = 0 if cat == 'neg' else 1

                    if dset == 'train': train_labels.append(label)

                    else: test_labels.append(label)



    train_texts = np.array(train_texts)

    train_labels = np.array(train_labels)

    test_texts = np.array(test_texts)

    test_labels = np.array(test_labels)



    train_texts, train_labels = shuffle(train_texts, train_labels)

    test_texts, test_labels = shuffle(test_texts, test_labels)



    return train_texts, train_labels, test_texts, test_labels
trX, trY, ttX, ttY = load_imdb_dataset(path)



print ('Train samples shape :', trX.shape)

print ('Train labels shape  :', trY.shape)

print ('Test samples shape  :', ttX.shape)

print ('Test labels shape   :', ttY.shape)
NGRAM_RANGE = (1, 2)

TOP_K = 25000

TOKEN_MODE = 'word'

MIN_DOC_FREQ = 2



def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {

        'ngram_range' : NGRAM_RANGE,

        'dtype' : 'int32',

        'strip_accents' : 'unicode',

        'decode_error' : 'replace',

        'analyzer' : TOKEN_MODE,

        'min_df' : MIN_DOC_FREQ,

    }

    

    tfidf_vectorizer = TfidfVectorizer(**kwargs)

    x_train = tfidf_vectorizer.fit_transform(train_texts)

    x_val = tfidf_vectorizer.transform(val_texts)

    

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))

    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val
def get_last_layer_units_and_activation(num_classes):

    activation = 'softmax'

    units = num_classes

    return units, activation
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):

    op_units, op_activation = get_last_layer_units_and_activation(num_classes)

    model = models.Sequential()

    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    

    for _ in range(layers-1):

        model.add(Dense(units=units, activation='relu'))

        model.add(Dropout(rate=dropout_rate))

        

    model.add(Dense(units=op_units, activation=op_activation))

    return model
def train_ngram_model(data, learning_rate=1e-3, epochs=1000, batch_size=128, layers=2, units=64, 

                      dropout_rate=0.2):

    

    num_classes = 25000

    

    

    trX, trY, ttX, ttY = data

    

    x_train, x_val = ngram_vectorize(trX, trY, ttX)

    

    

    model = mlp_model(layers, units=units, dropout_rate=dropout_rate,

                      input_shape=x_train.shape[1:], num_classes=num_classes)

    

    

   

    loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    

    

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    



    history = model.fit(x_train, trY, epochs=epochs, validation_data=(x_val, ttY),

                        verbose=2, batch_size=batch_size, callbacks=callbacks)

    



    history = history.history

    val_acc = history['val_acc'][-1]

    val_loss = history['val_loss'][-1]

    print ('Validation accuracy: {acc}, loss: {loss}'.format(

            acc=val_acc, loss=val_loss))

    

    model.save('IMDB_mlp_model_' + str(val_acc) + '_' + str(loss) + '.h5')

    return val_acc, val_loss
with tf.device('/device:GPU:2'):

    results = train_ngram_model((trX, trY, ttX, ttY))



print ('With lr=1e-3 | val_acc={results[0]} | val_loss={results[1]}'.format(results=results))

print ('===========================================================================================')