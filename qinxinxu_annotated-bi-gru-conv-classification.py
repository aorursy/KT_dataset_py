# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU

from keras.callbacks import Callback

from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from keras.preprocessing import text, sequence

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.models import Model

from keras.optimizers import Adam



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# all global parameters

conf = {

    'EMBEDDING_FILE' : '/kaggle/input/glove840b300dtxt/glove.840B.300d.txt',

    'train_file': '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv',

    'test_file': '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv',

    'test_label': '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv',

    

    'max_features':10000,

    'max_len': 150,

    'embed_size' : 300,



    'batch_size': 128,

    'epochs': 4,

    'train_size': 0.9

}
train = pd.read_csv(conf['train_file'])

test = pd.read_csv(conf['test_file'])



train.comment_text.fillna(' ')

test.comment_text.fillna(' ')



tags = list(train.columns)[2:]



X_train = train.comment_text.str.lower()

y_train = train[tags].values



X_test = test.comment_text.str.lower()
y_train
"""

  Abstract base class used to build new callbacks.

  Attributes:

      params: dict. Training parameters

          (eg. verbosity, batch size, number of epochs...).

      model: instance of `keras.models.Model`.

          Reference of the model being trained.

  The `logs` dictionary that callback methods

  take as argument will contain keys for quantities relevant to

  the current batch or epoch.

  

In Keras, Callback is a python class meant to be subclassed to provide specific functionality, 

with a set of methods called at various stages of training (including batch/epoch start and ends), 

testing, and predicting. 

Callbacks are useful to get a view on internal states and statistics of the model during training.

"""

class RocAucEvaluation(Callback):

    def __init__(self, validation_data = (), internal = 1):

        super(Callback, self).__init__()

        # ?

        self.internal = internal

        self.X_val, self.y_val = validation_data

        

    def on_epoch_end(self, epoch, logs={}):

        """

        Called at the end of an epoch.

        Arguments:

            epoch: integer, index of epoch.

            logs: dictionary of logs.

        """

        if epoch % self.internal == 0:

            # verbose: Integer. 0, 1, or 2. Verbosity mode.

            # 0 = silent, 1 = progress bar, 2 = one line per epoch.

#             Returns:

#         A `History` object. Its `History.history` attribute is

#         a record of training loss values and metrics values

#         at successive epochs, as well as validation loss values

#         and validation metrics values (if applicable).

            y_pred = self.model.predict(self.X_val, verbose = 0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC_AUC - epoch: {:d} - scoe: {:.6f}".format(epoch+1, score))
# num_words: the maximum number of words to keep, based on word frequency. 

# Only the most common num_words-1 words will be kept.



tok = text.Tokenizer(num_words = conf['max_features'])

# with default filter

# These sequences are then split into lists of tokens. They will then be indexed or vectorized.



tok.fit_on_texts(list(X_train) + list(X_test))

# Updates internal vocabulary based on a list of texts.

# Required before using texts_to_sequences or texts_to_matrix.



X_train = tok.texts_to_sequences(X_train)

X_test = tok.texts_to_sequences(X_test)

# Transforms each text in texts to a sequence of integers.

# Only top num_words-1 most frequent words will be taken into account. 

# Only words known by the tokenizer will be taken into account.



x_train = sequence.pad_sequences(X_train, maxlen = conf['max_len'])

x_test = sequence.pad_sequences(X_test, maxlen = conf['max_len'])

# Pads sequences to the same length.
# Split a sentence into a list of words.

# keras.preprocessing.text.text_to_word_sequence(text, 

#    filters=base_filter(), lower=True, split=" ")

x_train[1]
 # {word : embedding}

embedding_index = {}

with open(conf['EMBEDDING_FILE'], encoding = 'utf8') as f:

    for line in f:

        values = line.rstrip().rsplit(' ')

        # rsplit: except for spliting from right, same as split

        

        word = values[0]

        vec = np.asarray(values[1:], dtype = 'float32')

        embedding_index [word] = vec

        
# dictionary: {words : rank/index}. Only set after fit_on_texts was called.

word_index = tok.word_index



#  initializeing embedding matrix

num_words = min(conf['max_features'], len(word_index) + 1)

embedding_matrix = np.zeros((num_words,conf['embed_size']))



# what we want is index: embedding

for word, i in word_index.items():

    if i >= conf['max_features']:

        continue

    embedding_vec = embedding_index.get(word)

    if embedding_vec is not None:

        embedding_matrix[i] = embedding_vec

        

# add embedding matrix into para dict

conf['embedding_matrix'] = embedding_matrix



del embedding_index
# build model



inp = Input(shape = (conf['max_len'],))

# this returns a tensor

# a layer instance is callable on a tensor, and returns a tensor



x = Embedding(num_words, conf['embed_size'], weights = [conf['embedding_matrix']], trainable = False)(inp)

# Turns positive integers (indexes) into dense vectors of fixed size.

# This embedding layer will encode the input sequence into a sequence of dense embed_size vectors



x = SpatialDropout1D(0.2)(x)

# spatial drop out, drop entire 1d feature maps instead of individual elements.

# promote independence between feature maps



x = Bidirectional(GRU(128, return_sequences = True, dropout = 0.1 ))(x)

# bidirectional wrapper of rnn layer

# gated recurrent unit



x = Conv1D(64, kernel_size = 3, padding = 'valid', kernel_initializer = 'glorot_uniform')(x)

# 1d convolution layer 



avg_pool  = GlobalAveragePooling1D()(x)

# global averaging pooling

max_pool = GlobalMaxPooling1D()(x)

# global max pooling

x = concatenate([avg_pool, max_pool]) 



preds = Dense(6, activation="sigmoid")(x) # six classes

# Dense implements the operation: output = activation(dot(input, kernel) + bias)



model = Model(inp, preds)

# Model groups layers into an object with training and inference features.



model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])

# loss: what else?

    

# Optimizer that implements the Adam algorithm, set the learing rate

# metrics: what else?
# use a validation set to train the model

# should add a cross validation instead of just one model

conf['train_size'] = 0.9

conf['random_state'] = 0

X_tr, X_val, y_tr, y_val = train_test_split(x_train, y_train, train_size = conf['train_size'], random_state = conf['random_state'])
filepath="weights_base.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Save the best model after every epoch.



early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

# Stop training when a monitored quantity has stopped improving.



ra_val = RocAucEvaluation(validation_data=(X_val, y_val), internal = 1)

# use the validation set(from training set) to check the model



callbacks_list = [ra_val,checkpoint, early]

# print out when 1 epoch ends
model.fit(X_tr, y_tr, batch_size = conf['batch_size'], epochs = conf['epochs'], 

          validation_data = (X_val, y_val), callbacks = callbacks_list, verbose = 1)

model.load_weights(filepath)

print('predicting...')

y_pred = model.predict(x_test, batch_size = 1024, verbose = 1)