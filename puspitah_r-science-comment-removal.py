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
import os # accessing directory structure

import pandas as pd



reddit_200k_test = pd.read_csv("../input/rscience-popular-comment-removal/reddit_200k_test.csv", delimiter=',', encoding = 'ISO-8859-1')

reddit_200k_train = pd.read_csv("../input/rscience-popular-comment-removal/reddit_200k_train.csv", delimiter=',', encoding = 'ISO-8859-1')

reddit_test = pd.read_csv("../input/rscience-popular-comment-removal/reddit_test.csv", delimiter=',', encoding = 'ISO-8859-1')

reddit_train = pd.read_csv("../input/rscience-popular-comment-removal/reddit_train.csv", delimiter=',', encoding = 'ISO-8859-1')



print(reddit_test.head())

reddit_200k_test.head()

print('\t reddit_200k_test')        # 55843 entries, 8 columns

reddit_200k_test.info()

print('\t reddit_200k_train')       # 167529 entries, 8 columns

reddit_200k_train.info()

print('\t reddit_test')

reddit_test.info()                      # 7111 entries, 4 columns

print('\t reddit_train')

reddit_train.info()                      #21336 entries, 4 columns

# Clean up unnecessary columns

reddit_train = reddit_train.drop(columns="Unnamed: 0")

reddit_train = reddit_train.drop(columns="X")

reddit_test = reddit_test.drop(columns="Unnamed: 0")

reddit_test = reddit_test.drop(columns="X")

# Check composition of YES & NO

gap_reddit_test= len(reddit_test[reddit_test['REMOVED'] == 0])- len(reddit_test[reddit_test['REMOVED'] == 1])

print("reddit_test :  0 vs 1")

print(len(reddit_test[reddit_test['REMOVED'] == 0])," vs ", len(reddit_test[reddit_test['REMOVED'] == 1]), " = ",gap_reddit_test)

print("")



gap_reddit_train= len(reddit_train[reddit_train['REMOVED'] == 0])- len(reddit_train[reddit_train['REMOVED'] == 1])

print("reddit_train : 0 vs 1")

print(len(reddit_train[reddit_train['REMOVED'] == 0])," vs ", len(reddit_train[reddit_train['REMOVED'] == 1])," = ", gap_reddit_train)

# Delete rows to get balanced dataset

reddit_train = reddit_train.sort_values(by=['REMOVED'])

reddit_train = reddit_train.iloc[gap_reddit_train:,]

print("reddit_train : 0 vs 1")

print(len(reddit_train[reddit_train['REMOVED'] == 0])," vs ", len(reddit_train[reddit_train['REMOVED'] == 1]))



reddit_test = reddit_test.sort_values(by=['REMOVED'])

reddit_test = reddit_test.iloc[gap_reddit_test:,]

print("reddit_test : 0 vs 1")

print(len(reddit_test[reddit_test['REMOVED'] == 0])," vs ", len(reddit_test[reddit_test['REMOVED'] == 1]))

# Convert Pandas to Numpy

reddit_test_numpy=reddit_test.to_numpy()

print("reddit_test_numpy.shape= ",reddit_test_numpy.shape)

reddit_train_numpy=reddit_train.to_numpy()

print("reddit_train_numpy.shape= ",reddit_train_numpy.shape)
#Collect Key Metrics in the training dataset



import numpy as np

import matplotlib.pyplot as plt



sample_texts = reddit_train_numpy[:,0]



num_words = [len(s.split()) for s in sample_texts]



print("max = ",np.max(num_words))

print("min = ",np.min(num_words))

print("median = ",np.median(num_words)) # we can also get this from explore_data.py
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from shutil import copyfile

copyfile(src = "../input/utilities/util_text_classification.py", dst = "../working/util_text_classification.py")



from util_text_classification import *

plot_frequency_distribution_of_ngrams(sample_texts, ngram_range=(1, 2), num_ngrams=50)

plot_sample_length_distribution(sample_texts)

#Calculate ratio for model selection (no of samples/no words per sample)



ratio = reddit_train_numpy.shape[0]/np.median(num_words)

print(ratio)   # 320
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

import category_encoders as ce

from sklearn.preprocessing import OneHotEncoder



X_train = reddit_train_numpy[:,0]

X_test = reddit_test_numpy[:,0]

y_train=reddit_train_numpy[:,1]

y_test=reddit_test_numpy[:,1]



X= np.concatenate(( reddit_train_numpy[:,0],  reddit_test_numpy[:,0]))

print(X_train.shape, "+", X_test.shape, " = ", X.shape)



X_train_ngram, X_test_ngram = ngram_vectorize(X_train, y_train, X_test)

print("X_train_ngram.shape= ", X_train_ngram.shape)

print("X_test_ngram.shape= ", X_test_ngram.shape)



#Categorical encoding is done after vectorization to avoid error on the method 'ngram_vectorize'

# 0 = 1 0

# 1 = 0 1

ohe =  ce.OneHotEncoder(handle_unknown='ignore')

y_train=ohe.fit_transform(reddit_train_numpy[:,1])

y_test=ohe.fit_transform(reddit_test_numpy[:,1])

print(y_train)

print("transformed into :")

print(reddit_train_numpy[0:4,1])

print(reddit_train_numpy[13709:13713,1])
# Rename input & labels

print(type(X_train_ngram))



X_train_ngram = X_train_ngram.todense()



print(type(X_train_ngram))

print(X_train_ngram.shape)



input_size=X_train_ngram.shape[1]
#Shallow NN

from keras import layers, models, optimizers, losses

from keras.models import Sequential

from keras.layers import Dense

from keras.metrics import categorical_accuracy

# ---------------------------------





#Due to big input dimension, do not use lots of nodes & layers to avoid overfitting

net = Sequential()

net.add(Dense(1, input_dim=input_size, activation='relu'))

net.add(Dense(2, activation='softmax' ))



net.compile(loss='categorical_crossentropy' , optimizer=optimizers.Adam(), metrics=['accuracy'] )



net.summary()

net.fit(X_train_ngram, y_train, epochs=10, verbose=1)

print("training loss, acc: " + str(net.evaluate(X_train_ngram, y_train, verbose=0)))

print("test loss, acc: " + str(net.evaluate(X_test_ngram, y_test, verbose=0)))

import numpy as numpy

from sklearn.metrics import accuracy_score



# load the pre-trained word-embedding vectors 

embeddings_index = {}

for i, line in enumerate(open('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')):

    values = line.split()

    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')



# create a tokenizer 

token = text.Tokenizer()

#token.fit_on_texts(trainDF['text'])    # change from pandas to numpy

token.fit_on_texts(X_train)   # should be X_train + X_test

word_index = token.word_index



# convert text to sequence of tokens and pad them to ensure equal length vectors 

X_train_seq = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=70)

X_test_seq = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=70)

print("X_train_seq.shape", X_train_seq.shape)

print("X_test_seq.shape", X_test_seq.shape)



# create token-embedding mapping

embedding_matrix = numpy.zeros((len(word_index) + 1, 300))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

        

def create_rnn_lstm():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the LSTM Layer

    lstm_layer = layers.LSTM(100)(embedding_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(70, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    #output_layer12 = layers.Dense(70, activation="relu")(output_layer1)

    output_layer2 = layers.Dense(2, activation="softmax")(output_layer1)



    # Compile the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



LSTM_classifier = create_rnn_lstm()

LSTM_classifier.fit(X_train_seq, y_train, batch_size=128, epochs = 10, shuffle=True) # batch_size=64

LSTM_classifier.summary()

predictions = LSTM_classifier.predict(X_test_seq)

predictions= np.argmax(predictions, axis=1)   # change from 2D array to 1D array



y_test_int64=reddit_test_numpy[:,1].astype(int)  # change from numpy object to int64 as required by sklearn.accuracy

print ("RNN-LSTM, Word Embeddings accuracy (Eout) = ",  accuracy_score(predictions, y_test_int64))



predictions2 = LSTM_classifier.predict(X_train_seq)

predictions2= np.argmax(predictions2, axis=1)   # change from 2D array to 1D array



y_train_int64=reddit_train_numpy[:,1].astype(int)  # change from numpy object to int64 as required by sklearn.accuracy

print ("RNN-LSTM, Word Embeddings accuracy (Ein)= ", accuracy_score(predictions2, y_train_int64))



#Result will be bettter if number of epoch is increased but then started to overfit
def create_rnn_gru():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the GRU Layer

    lstm_layer = layers.GRU(100)(embedding_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(2, activation="softmax")(output_layer1)



    # Compile the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



GRU_classifier = create_rnn_gru()

GRU_classifier.fit(X_train_seq, y_train, batch_size=128, epochs = 10, shuffle=True)

GRU_classifier.summary()



predictions = GRU_classifier.predict(X_test_seq)

predictions= np.argmax(predictions, axis=1)   # change from 2D array to 1D array



predictions2 = GRU_classifier.predict(X_train_seq)

predictions2= np.argmax(predictions2, axis=1)   # change from 2D array to 1D array



print ("RNN-GRU, Word Embeddings accuracy (Ein)= ", accuracy_score(predictions2, y_train_int64))

print ("RNN-GRU, Word Embeddings accuracy (Eout) = ",  accuracy_score(predictions, y_test_int64))
def create_bidirectional_rnn():

    # Add an Input Layer

    input_layer = layers.Input((70, ))



    # Add the word embedding Layer

    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)

    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)



    # Add the LSTM Layer

    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)



    # Add the output Layers

    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)

    output_layer1 = layers.Dropout(0.25)(output_layer1)

    output_layer2 = layers.Dense(2, activation="sigmoid")(output_layer1)



    # Compile the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)

    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    

    return model



#-------------------------------------------------

Bidirectional_classifier = create_bidirectional_rnn()

Bidirectional_classifier.fit(X_train_seq, y_train, batch_size=2000, epochs = 10, shuffle=True)

Bidirectional_classifier.summary()



predictions = Bidirectional_classifier.predict(X_test_seq)

predictions= np.argmax(predictions, axis=1)   # change from 2D array to 1D array



predictions2 = Bidirectional_classifier.predict(X_train_seq)

predictions2= np.argmax(predictions2, axis=1)   # change from 2D array to 1D array



print ("RNN-Bidirectional, Word Embeddings accuracy (Ein)= ", accuracy_score(predictions2, y_train_int64))

print ("RNN-Bidirectional, Word Embeddings accuracy (Eout) = ",  accuracy_score(predictions, y_test_int64))