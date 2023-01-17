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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from sklearn import preprocessing

from keras.preprocessing import sequence



from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding, SpatialDropout1D, MaxPooling1D, Embedding, Conv1D, Flatten, Dropout

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM

from keras.optimizers import Adam

from sklearn.utils import shuffle

from sklearn.metrics import f1_score, confusion_matrix





# load data

input_file = "../input/imdb-review-dataset/imdb_master.csv"



# comma delimited is the default

data = pd.read_csv(input_file, header = 0, encoding='ISO-8859-1', engine='python')


# show data

data.head()
# we will divide data on train and test sets



indexes_train = np.where(data['type'] == 'train')

indexes_test = np.where(data['type'] == 'test')



X_train = data['review'][indexes_train[0]].values

Y_train = data['label'][indexes_train[0]].values



X_test = data['review'][indexes_test[0]].values

Y_test = data['label'][indexes_test[0]].values
# certain indices of positive and negative reviews



index_unsup = np.where(Y_train == 'unsup')

Y_train = np.delete(Y_train, index_unsup)

X_train = np.delete(X_train, index_unsup)
# example review

data['review'][55000]
# Analysis of the test and training sample



def buil_hist(data):

    lenf_data = []

    for i in data:

        lenf_data.append(i)

    return lenf_data

    

lenf_train = buil_hist(Y_train)

lenf_test = buil_hist(Y_test)   



plt.subplot(1, 2, 1)

plt.title('Train')

plt.hist(lenf_train)

plt.subplot(1, 2, 2)

plt.hist(lenf_test)

plt.title('Test')

plt.show()
# transform catecorical labels in num labels

le = preprocessing.LabelEncoder()

le.fit(Y_train)



print('Lisr unique labels: {}'.format(le.classes_))



Y_train_encod = le.transform(Y_train) 

Y_test_encod = le.transform(Y_test) 



# inverse transform

# list(le.inverse_transform(Y_test_encod))



# shufle train data set

X_train, Y_train_encod = shuffle(X_train, Y_train_encod)



# check a shuffle

print(Y_train_encod[10], '\n', X_train[10])
# HYPERPARAMETERS

max_features = 10000

maxlen = 100

embedding_dimenssion = 100



VALIDATION_SPLIT = 0.1

CLASSES = 1

NB_EPOCH = 20

BATCH_SIZE = 64

OPTIMIZER = Adam(lr=0.001)



# Tokenization and encoding text corpus

tk = Tokenizer(num_words=max_features)

tk.fit_on_texts(X_train)

X_train_en = tk.texts_to_sequences(X_train)

X_test_en = tk.texts_to_sequences(X_test)



# dictionaries

word2index = tk.word_index

index2word = tk.index_word
# check the correctness of the encoding

print('Orginal \n{}'.format(X_train[2]))



print('\nDecoding')

for index in X_train_en[2]:

    x = index2word.get(index)

    print(x, end=' ')

print('\nVerify coding fidelity. Click continue to continue.')
# Аnalysis of the length of each review

lenf_reviews = list(map(len, X_train_en))



plt.hist(lenf_reviews)

plt.title('lenf_reviews')

plt.show()
# we give feedback to the same dimension

X_train_new = sequence.pad_sequences(X_train_en, maxlen=maxlen)

X_test_new = sequence.pad_sequences(X_test_en, maxlen=maxlen)

print('\nLed examples of texts to the general dimension. Click continue to continue')



# path to the pre-trained word vectors or download the link

# https://nlp.stanford.edu/projects/glove/

glove_dir = ''.join(['../input/glove-vectors/glove.6B.', str(embedding_dimenssion),'d.txt']) # This is the folder with the dataset



embeddings_index = {} # We create a dictionary of word -> embedding



with open(glove_dir, encoding='utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0] # The first value is the word, the rest are the values of the embedding

        embedding = np.asarray(values[1:], dtype='float32') # Load embedding

        embeddings_index[word] = embedding # Add embedding to our embedding dictionary



print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

print('\nLoaded the pre-trained word vectors in English: Click continue to continue')
embedding_matrix = np.zeros((max_features, embedding_dimenssion))



# The vectors need to be in the same position as their index.

# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on



# Loop over all words in the word index

for word, i in word2index.items():

    # If we are above the amount of words we want to use we do nothing

    if i >= max_features:

        break

    # Get the embedding vector for the word

    embedding_vector = embeddings_index.get(word)

    # If there is an embedding vector, put it in the embedding matrix

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

print('Creating your own dictionary of word vectors from Glove is over. Click continue.')
# checking the dimension of arrays

print(embedding_matrix.shape, X_train_new.shape, Y_train_encod.shape, X_test_new.shape, Y_test_encod.shape)
print('Click continue to learn the model')



model = Sequential()



# LSTM version 1

model.add(Embedding(max_features, embedding_dimenssion, input_length=maxlen,

                    weights=[embedding_matrix], trainable=False))

model.add(LSTM(125, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation="sigmoid"))

model.summary()





# compile the model



model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])



# train model

model.fit(X_train_new, Y_train_encod, batch_size=BATCH_SIZE, epochs=10, validation_split=VALIDATION_SPLIT, verbose=1)



# evaluate the quality of the system using accuracy

scores = model.evaluate(X_test_new, Y_test_encod)

print('losses: {}'.format(scores[0]))

print('TEST accuracy: {}'.format(scores[1]))
# predicted labels on test

Y_predicted_test = model.predict_classes(X_test_new)
# evaluate the quality of the system using f1-score and confusion matrix

print('F1-score: {0}'.format(f1_score(Y_predicted_test, Y_test_encod)))

print('Confusion matrix:')

confusion_matrix(Y_predicted_test, Y_test_encod)