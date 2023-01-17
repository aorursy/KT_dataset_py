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

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
df_train = pd.read_csv("../input/data111/SADATA2 - SADATA.tsv", header=0, delimiter="\t", quoting=3,encoding='utf8', engine='python')

df_train.head()
maping = {'NOT': 1, 'HOF': 0}

df_train['Task A'] = df_train['Task A'].map(maping)
tk = Tokenizer(lower = True, filters='')

tk.fit_on_texts(df_train['Tweet'])
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( df_train['Tweet'], df_train['Task A'], test_size=0.2, random_state=42)
y = np.array(y_train)
y
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

y_ohe = ohe.fit_transform(y.reshape(-1, 1))
y_ohe
train_tokenized = tk.texts_to_sequences(X_train)

test_tokenized = tk.texts_to_sequences(X_test)
max_len = 50

X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_test = pad_sequences(test_tokenized, maxlen = max_len)
embed_size = 300

max_features = 30000
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path,encoding='utf-8'))



word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    

    inp = Input(shape = (max_len,))

    x = Embedding(9837, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x1 = SpatialDropout1D(spatial_dr)(x)



    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)

    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)

    avg_pool1_gru = GlobalAveragePooling1D()(x1)

    max_pool1_gru = GlobalMaxPooling1D()(x1)

    

    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)

    avg_pool3_gru = GlobalAveragePooling1D()(x3)

    max_pool3_gru = GlobalMaxPooling1D()(x3)

    

    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)

    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)

    avg_pool1_lstm = GlobalAveragePooling1D()(x1)

    max_pool1_lstm = GlobalMaxPooling1D()(x1)

    

    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)

    avg_pool3_lstm = GlobalAveragePooling1D()(x3)

    max_pool3_lstm = GlobalMaxPooling1D()(x3)

    

    

    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,

                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])

    x = BatchNormalization()(x)

    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))

    x = BatchNormalization()(x)

    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))

    x = Dense(2, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.1, 

                        verbose = 1, callbacks = [check_point, early_stop])

    model = load_model(file_path)

    return model
model1 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 64, spatial_dr = 0.3, kernel_size1=3, kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)
model1.summary()
pred1 = model1.predict(X_test, batch_size = 1024, verbose = 1)

pred = pred1
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
accuracy_score(y_test, predictions)