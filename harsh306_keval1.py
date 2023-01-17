import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import sys, os, re

np.random.seed(32)

os.environ["OMP_NUM_THREADS"] = "4"



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv", 

                    usecols= ['id','target','comment_text'] )

test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")



#train['target'] = train.target.apply(lambda x: 1 if x > 0.5 else 0) # slow

train['target'] = np.where(train['target'] > 0.5, 1, 0) #faster

train['comment_text'] = train.comment_text.apply(lambda x: x.lower())
y = train.target.values

X_train, X_valid, Y_train, Y_valid = train_test_split(train[['comment_text']], y, test_size = 0.1)
embed_size = 300

max_features = 200000

max_len = 220
tk = Tokenizer(num_words=max_features, lower = True)

tk.fit_on_texts(X_train['comment_text'].values)

X_train["comment_seq"] = tk.texts_to_sequences(X_train['comment_text'].values)

X_valid["comment_seq"] = tk.texts_to_sequences(X_valid['comment_text'].values)

test["comment_seq"] = tk.texts_to_sequences(test['comment_text'].values)
%%time

X_train_p = pad_sequences(X_train.comment_seq, maxlen = max_len)

X_valid_p = pad_sequences(X_valid.comment_seq, maxlen = max_len)

test_p = pad_sequences(test.comment_seq, maxlen = max_len)

del X_train, X_valid, test, train
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, LeakyReLU

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, CuDNNGRU,CuDNNLSTM

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D



import logging

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback



import warnings

warnings.filterwarnings('ignore')
file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,save_best_only = True)
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape = (max_len,))

    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    #x = SpatialDropout1D(dr)(x)

    x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)  

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x) 

    #att = AttentionWeightedAverage()(x)

    conc = concatenate([avg_pool, max_pool])

    output = Dropout(0.75)(conc)

    output = Dense(units=110)(output)

    output = LeakyReLU(alpha=0.3)(output)

    output = Dense(units=55)(output)

    output = Activation('relu')(output)

    prediction = Dense(1, activation = "sigmoid")(output)

    model = Model(inputs = inp, outputs = prediction)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train_p, Y_train, batch_size = 1024, epochs = 6, validation_data = (X_valid_p, Y_valid), 

                        verbose = 1 , callbacks = [check_point])

    model = load_model(file_path)

    return model
model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.3)
pred = model.predict(test_p, batch_size = 1024, verbose = 1)
submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")

submission['prediction'] = pred

submission.to_csv("submission.csv", index = False)

submission.head(10)