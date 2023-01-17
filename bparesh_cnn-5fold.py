## This kernel is using CNN model approach 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import os

import gc

import logging

import datetime

import warnings

import pickle

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import time
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from tensorflow.compat.v1.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Flatten

from tensorflow.compat.v1.keras.layers import Conv1D, MaxPooling1D

from tensorflow.compat.v1.keras.preprocessing import text, sequence

from tensorflow.compat.v1.keras.losses import binary_crossentropy

from tensorflow.compat.v1.keras import backend as K

import tensorflow.compat.v1.keras.layers as L

from tensorflow.compat.v1.keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.compat.v1.keras.layers import Layer

from tensorflow.compat.v1.keras.models import Model

from tensorflow.compat.v1.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer

from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences

from tensorflow.compat.v1.keras.utils import plot_model

from tensorflow.compat.v1.keras.callbacks import TensorBoard
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)
def build_embedding_matrix(word_index, path):

    '''

     credits to: https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold

    '''

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

        except:

            embedding_matrix[i] = embeddings_index["unknown"]

            

    del embedding_index

    gc.collect()

    return embedding_matrix
def tokenizer_text(train, test):

    '''

        credits go to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ 

    '''



    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    punct += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'

    tokenizer = Tokenizer(filters=punct) 

    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))

    word_index = tokenizer.word_index

    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))

    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))

    X_train = pad_sequences(X_train, maxlen=MAX_LEN)

    X_test = pad_sequences(X_test, maxlen=MAX_LEN)

    

    return X_train, X_test, word_index
def build_embeddings(word_index):

    embedding_matrix = np.concatenate(

        [build_embedding_matrix(word_index, f) for f in EMB_PATHS], axis=-1) 

    return embedding_matrix
def load_data():

    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')

    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')

    y_train = np.where(train['target'] >= 0.5, True, False) * 1

    X_train, X_test, word_index = tokenizer_text(train, test)

    embedding_matrix = build_embeddings(word_index)

    del train,test

    gc.collect()

    return X_train,y_train, X_test, word_index, embedding_matrix
EMB_MAX_FEAT = 300

MAX_LEN = 220

BATCH_SIZE = 512

NUM_EPOCHS = 10

COMMENT_TEXT_COL = 'comment_text'

EMB_PATHS = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
def build_model(embedding_matrix, fold_n=0):

    file_path = f"best_model_fold_{fold_n}.hdf5"

    check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,save_best_only=True, mode="min")

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    

    inp = Input(shape = (MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)

    x = SpatialDropout1D(0.2)(x)

    # add conv layers

    x = Conv1D(128, 2, activation='relu', padding='same')(x)

    x = MaxPooling1D(5, padding='same')(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)

    x = MaxPooling1D(5, padding='same')(x)

    x = Conv1D(128, 4, activation='relu', padding='same')(x)

    x = MaxPooling1D(5, padding='same')(x)

    x = Conv1D(128, 5, activation='relu', padding='same')(x)

    x = MaxPooling1D(5, padding='same')(x)    

    x = Flatten()(x)

    x = Dropout(0.1)(Dense(128, activation='relu') (x))

    result = Dense(1, activation="sigmoid")(x)

    

    model = Model(inputs=inp, outputs=result)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model
def run_model(X, y,X_test, embedding_matrix, word_index):



    predictions = np.zeros((len(X_test), 1))

    n_fold = 5

    tensorboard_callback = TensorBoard("logs")

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        # separate train and validation data

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        # build model

        model = build_model(embedding_matrix, fold_n)

        # training



        model.fit(

            X_train,y_train,

            batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2,

            validation_data=(X_valid, y_valid),

            callbacks=[tensorboard_callback] 

        )



        predictions+=model.predict(X_test, batch_size=2048)

        print(predictions)

        del model

        gc.collect()    

       

    preds = predictions/n_fold

    print(preds)

    return preds
def submit(sub_preds):

    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

    submission['prediction'] = sub_preds

    submission.reset_index(drop=False, inplace=True)

    submission.to_csv('submission.csv', index=False)
# import data

X_train, y_train,X_test, word_index,embedding_matrix = load_data()

model = build_model(embedding_matrix)

model.summary()
# plot model

plot_model(model, to_file='model_plot.png',show_layer_names=True)
del model

gc.collect()
sub_preds = run_model(X_train, y_train,X_test, embedding_matrix, word_index)

submit(sub_preds)
# Load the extension and start TensorBoard



%load_ext tensorboard

%tensorboard --logdir logs