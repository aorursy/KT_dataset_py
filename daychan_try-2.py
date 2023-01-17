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
#coding:utf-8

import keras

import os

import pdb

import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = "0"



VECTOR_DIR = 'vectors.bin'



MAX_SEQUENCE_LENGTH = 200

EMBEDDING_DIM = 200

VALIDATION_SPLIT = 0.16

TEST_SPLIT = 0.2

MAX_FEATURES = 70000



def load_doc2vec():

    from keras.utils import to_categorical

    train_docs = open('train_contents.txt').read().split('\n')

    train_labels = open('train_labels.txt').read().split('\n')

    test_docs = open('test_contents.txt').read().split('\n')

    test_labels = open('test_labels.txt').read().split('\n')

    data = train_docs + test_docs

    print('(1) training doc2vec model...')

    # train_d2v_model()

    print('(2) load doc2vec model...')

    import gensim

    model = gensim.models.Doc2Vec.load('doc2vec.model')

    x_train = []

    x_test = []

    y_train = train_labels

    y_test = test_labels

    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))

    p2 = int(len(data) * (1 - TEST_SPLIT))

    i = 0

    for idx, docvec in enumerate(model.docvecs):

        if idx < p2:

            x_train.append(docvec)

        else:

            x_test.append(docvec)

            try:

                model.docvecs[idx + 1]

            except KeyError:

                break



    y_train = to_categorical(np.asarray(y_train))

    y_test = to_categorical(np.asarray(y_test))

    x_train = np.asarray(x_train)

    x_test = np.asarray(x_test)

    x_val = x_train[p1:p2]

    y_val = y_train[p1:p2]

    x_train = x_train[:p1]

    y_train = y_train[:p1]



    print('train doc shape: ' + str(len(x_train)) + ' , ' + str(len(x_train[0])))

    print('test doc shape: ' + str(len(x_test)) + ' , ' + str(len(x_test[0])))

    return x_train, y_train, x_val, y_val, x_test, y_test, data



def load_normal():

    print("Step 1")

    train_texts = open('../input/train_contents.txt').read().split('\n')

    train_labels = open('../input/train_labels.txt').read().split('\n')

    test_texts = open('../input/test_contents.txt').read().split('\n')

    test_labels = open('../input/test_labels.txt').read().split('\n')

    all_texts = train_texts + test_texts

    all_labels = train_labels + test_labels



    print("Step 2")

    from keras.preprocessing.text import Tokenizer

    from keras.preprocessing.sequence import pad_sequences

    from keras.utils import to_categorical

    



    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(all_texts)

    sequences = tokenizer.texts_to_sequences(all_texts)

    word_index = tokenizer.word_index



    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(all_labels))

    print('Shape of data tensor:', data.shape)

    print('Shape of label tensor:', labels.shape)



    # import random

    # a = random.Random()

    # a.seed(10)

    # a.shuffle(data)

    # a.shuffle(labels)



    print("Step 3")

    # split the data into training set, validation set, and test set

    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))

    p2 = int(len(data) * (1 - TEST_SPLIT))

    x_train = data[:p1]

    y_train = labels[:p1]

    x_val = data[p1:p2]

    y_val = labels[p1:p2]

    x_test = data[p2:]

    y_test = labels[p2:]

    print('train docs: ' + str(len(x_train)))

    print('val docs: ' + str(len(x_val)))

    print('test docs: ' + str(len(x_test)))

    return x_train, y_train, x_val, y_val, x_test, y_test, word_index



x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_normal()

print("Step 4: training model...")

from keras.layers import Dense, Input, Flatten, Dropout

from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, CuDNNLSTM, GlobalAveragePooling1D

from keras.models import Sequential

from keras import optimizers, metrics

# from keras.utils import plot_model





def load_word2vec():

    print('(4) load word2vec as embedding...')

    import gensim

    from keras.utils import plot_model

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)

    # 读取word2vec模型

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    # shpae = (65605, 200)



    not_in_model = 0

    in_model = 0

    for word, i in word_index.items():

        if word in w2v_model:

            in_model += 1

            embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')

        else:

            not_in_model += 1

    # 在word_index字典中取word，到w2v_model中查多少词没有, 将有的词加入embedding_matrix，对于没有的词，weight = 0　(np.zeros　in line 70)



    print(str(not_in_model)+' words not in w2v model')

    # 结果：　13822 words not in w2v model



    from keras.layers import Embedding

    embedding_layer = Embedding(len(word_index) + 1,

                                EMBEDDING_DIM,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=False)

    # 将embedding层的参数设置为所得的embedding_matrix，并锁层不进行训练 (trainable=False)

    return embedding_layer





# pdb.set_trace()

model = Sequential()



# we start off with an efficient embedding layer which maps

# our vocab indices into embedding_dims dimensions

model.add(Embedding(len(word_index) + 1,

                    EMBEDDING_DIM,

                    input_length=MAX_SEQUENCE_LENGTH))



model.add(Conv1D(int(EMBEDDING_DIM / 4), 10, padding='same', activation='relu', strides=10))

# model.add(Dropout(0.2))

model.add(CuDNNLSTM(int(EMBEDDING_DIM / 4)))



model.add(Dense(12, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])



model.fit(x_train, y_train,

          epochs=5,

          validation_data=(x_val, y_val))

model.save('cnn.h5')



print("Step 5: testing model...")

print(model.metrics_names)

# model.metrics

print(model.evaluate(x_test, y_test))
!ls ../input