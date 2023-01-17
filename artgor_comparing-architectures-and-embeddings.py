import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from nltk.tokenize import TweetTokenizer

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, cross_val_score

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

pd.set_option('max_colwidth',400)



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, add

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

import os

print(os.listdir("../input"))
# I'll load preprocessed data from my dataset

train = pd.read_csv('../input/jigsaw-public-files/train.csv')

test = pd.read_csv('../input/jigsaw-public-files/test.csv')

# after processing some of the texts are emply

train['comment_text'] = train['comment_text'].fillna('')

test['comment_text'] = test['comment_text'].fillna('')

sub = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
full_text = list(train['comment_text'].values) + list(test['comment_text'].values)
%%time

tk = Tokenizer(lower = True, filters='', num_words=120000)

tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train['comment_text'])

test_tokenized = tk.texts_to_sequences(test['comment_text'])
max_len = 230

X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_test = pad_sequences(test_tokenized, maxlen = max_len)
# y = train['target']

y = np.where(train['target'] >= 0.5, True, False) * 1

from keras.utils import to_categorical

y_binary = to_categorical(y)
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embed_size = 300

max_features = 120000
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
# score is 0.92393

def build_model(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    

    inp = Input(shape = (max_len,))

    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x1 = SpatialDropout1D(spatial_dr)(x)



    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)

    

    x1 = Conv1D(int(units/2), kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)

    

    y = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)

    y = Conv1D(int(units/2), kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)

    

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    

    avg_pool2 = GlobalAveragePooling1D()(y)

    max_pool2 = GlobalMaxPooling1D()(y)

       

    

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    x = Dense(2, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_binary, batch_size = 128, epochs = 3, validation_split=0.1, 

                        verbose = 2, callbacks = [check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_model(lr = 1e-3, lr_d = 0, units = 128, spatial_dr = 0.1)

pred = model.predict(X_test, batch_size = 1024, verbose = 1)[:, 1]
plt.hist(pred);

plt.title('Distribution of predictions');
# sub['prediction'] = pred

# sub.to_csv('submission.csv', index=False)
# https://www.kaggle.com/tunguz/bi-gru-cnn-poolings-gpu-kernel-version/data

def build_model1(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)



    inp = Input(shape = (max_len,))



    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x1 = SpatialDropout1D(dr)(x)

    

    x = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)

    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)



    y = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)

    y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)



    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)



    avg_pool2 = GlobalAveragePooling1D()(y)

    max_pool2 = GlobalMaxPooling1D()(y)



    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    x = Dense(2, activation = "sigmoid")(x)



    model = Model(inputs = inp, outputs = x)



    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])



    history = model.fit(X_train, y_binary, batch_size = 128, epochs = 3, validation_split=0.1, 



                        verbose = 2, callbacks = [check_point, early_stop])



    model = load_model(file_path)



    return model
# model = build_model1(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

# pred = model.predict(X_test, batch_size = 1024, verbose = 1)[:, 1]

# sub['prediction'] = pred

# sub.to_csv('submission.csv', index=False)
def build_model2(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)



    inp = Input(shape = (max_len,))



    x = Embedding(max_features + 1, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x = SpatialDropout1D(dr)(x)

    

    x = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x)

    x = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    

    hidden = add([hidden, Dense(units * 4, activation='relu')(hidden)])

    hidden = add([hidden, Dense(units * 4, activation='relu')(hidden)])



    result = Dense(2, activation='sigmoid')(hidden)



    model = Model(inputs = inp, outputs = result)



    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    

    history = model.fit(X_train, y_binary, batch_size = 128, epochs = 3, validation_split=0.1, 



                        verbose = 2, callbacks = [check_point, early_stop])



    model = load_model(file_path)



    return model
# model = build_model2(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

# pred = model.predict(X_test, batch_size = 1024, verbose = 1)[:, 1]

# sub['prediction'] = pred

# sub.to_csv('submission.csv', index=False)