# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import pandas as pd

from gensim.models import Word2Vec, Doc2Vec

from gensim.models.doc2vec import TaggedDocument

from gensim.test.utils import common_texts, get_tmpfile

import matplotlib.pyplot as plt

import numpy as np

import pickle



from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import confusion_matrix, f1_score, classification_report



import keras

from keras import losses, models

from keras.models import Sequential, Model

from keras.layers import (Input, Conv2D, Conv1D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Embedding, GlobalMaxPooling1D,

                          SeparableConv1D, MaxPooling1D, GlobalMaxPooling2D, MaxPooling2D, concatenate, Activation, ZeroPadding2D, Dropout,

                          add, Concatenate, Reshape, MaxPool2D, MaxPool1D, Lambda, Reshape, ReLU, SpatialDropout1D, Add)

from keras.callbacks import (LearningRateScheduler, ReduceLROnPlateau)

from keras.optimizers import Adam, SGD

from keras.regularizers import l2



from tensorflow.python.keras.preprocessing import sequence

from tensorflow.python.keras.preprocessing import text
# download link for output files

from IPython.display import HTML

import pandas as pd

import numpy as np



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
# Tokenize titles



# Limit on the length of text sequences. Sequences longer than this will be truncated.

MAX_SEQUENCE_LENGTH = 20



# load train and test titles

train = pd.read_csv('../input/train.csv')

train_titles = train['title'].tolist()

test = pd.read_csv('../input/test.csv')

test_titles = test['title'].tolist()

train_and_test_titles = train_titles

train_and_test_titles.extend(test_titles)



# tokenize titles, pad shorter titles with zeros

tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(train_and_test_titles)

sequences = tokenizer.texts_to_sequences(train_and_test_titles)

sequences = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index

word_index['END'] = 0

print('Found %s unique tokens.' % len(word_index))



# get labels

labels = np.asarray(train['Category'].tolist())

labels = keras.utils.to_categorical(labels)



# resplit train and submission data

sequences_submission = sequences[len(train):]

sequences = sequences[:len(train)]
# Pre-train word vectors using gensim's word2vec



EMBED_SIZE = 300



# split titles into their individual words

tokenized_titles = []

for title in train_and_test_titles:

    tokenized_titles.append(title.split())



# train word vectors

model = Word2Vec(tokenized_titles, size=EMBED_SIZE, window=5, min_count=1, workers=4)



# get word vectors in a dict format

w2v_dict = dict(zip(model.wv.index2word, model.wv.vectors))



# create embedding matrix for keras Embedding layer

embedding_matrix = np.zeros((len(word_index), EMBED_SIZE))

for word, i in word_index.items():

    if word == 'END':

        continue

    embedding_vector = w2v_dict.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

print('embedding matrix shape:', embedding_matrix.shape)
# learning rate scheduler

def step_decay(epoch):

    initial_lrate = 1e-4

    drop = 0.1

    epochs_drop = 20.0

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate



# NN params

maxlen = sequences.shape[1]

max_features = embedding_matrix.shape[0]

embed_size = embedding_matrix.shape[1]

filter_nr = 512

filter_size = 2

dense_nr = 4096

spatial_dropout = 0.2

dense_dropout = 0.5

train_embed = True



# define the model (Functional API)

inputs = Input(shape=(maxlen,), dtype='int32')

embed = Embedding(input_dim=max_features, output_dim=embed_size, weights=[embedding_matrix], trainable=train_embed)(inputs)

embed = SpatialDropout1D(spatial_dropout)(embed)



block1 = BatchNormalization()(embed)

block1 = ReLU()(block1)

block1 = Conv1D(filters=filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)

block1 = GlobalMaxPooling1D()(block1)



block2 = BatchNormalization()(embed)

block2 = ReLU()(block2)

block2 = Conv1D(filters=filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)

block2 = GlobalMaxPooling1D()(block2)



block3 = BatchNormalization()(embed)

block3 = ReLU()(block3)

block3 = Conv1D(filters=filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)

block3 = GlobalMaxPooling1D()(block3)



block4 = BatchNormalization()(embed)

block4 = ReLU()(block4)

block4 = Conv1D(filters=filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)

block4 = GlobalMaxPooling1D()(block4)



output = Concatenate()([block1, block2, block3, block4])

output = Dense(dense_nr, activation='relu')(output)

output = Dropout(dense_dropout)(output)

output = Dense(58, activation='softmax')(output)



# build, summarize, and fit model

model = Model(inputs=inputs, outputs=output)

lrate = LearningRateScheduler(step_decay, verbose=1)

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model_history = model.fit(sequences, labels, batch_size=512, epochs=60, verbose=1, callbacks=[lrate])



# evaluate model on current fold of training and validation data, as well as dedicated test data

train_acc = model_history.history['acc'][-1]

train_loss = model_history.history['loss'][-1]



# summarize history for accuracy

plt.plot(model_history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(model_history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# predict and cast as dataframe

y_submit = model.predict(sequences_submission, batch_size=128)

y_submit_int = np.argmax(y_submit, axis=1)

y_submit_int_df = pd.DataFrame({'Category':y_submit_int})



# join dataframes to arrive at submission dataframe

submit = test.drop(columns=['title', 'image_path'])

submit = submit.join(y_submit_int_df)

submit.to_csv('submission_WideCNN_Word2Vec.csv', index=False)

create_download_link(filename='submission_WideCNN_Word2Vec.csv')