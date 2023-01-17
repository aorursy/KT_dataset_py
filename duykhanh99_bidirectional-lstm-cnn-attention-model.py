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
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")

test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")

sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")
train.head(15)
test.head()
train.loc[train.SentenceId == 2]
print('Average count of phrases per sentence in train is {0:.0f}.'.format(train.groupby('SentenceId')['Phrase'].count().mean()))

print('Average count of phrases per sentence in test is {0:.0f}.'.format(test.groupby('SentenceId')['Phrase'].count().mean()))
print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0], len(train.SentenceId.unique())))

print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0], len(test.SentenceId.unique())))
print('Average word length of phrases in train is {0:.0f}.'.format(np.mean(train['Phrase'].apply(lambda x: len(x.split())))))

print('Average word length of phrases in test is {0:.0f}.'.format(np.mean(test['Phrase'].apply(lambda x: len(x.split())))))
text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)

text_trigrams = [i for i in ngrams(text.split(), 3)]
Counter(text_trigrams).most_common(30)
text = ' '.join(train.loc[train.Sentiment == 4, 'Phrase'].values)

text = [i for i in text.split() if i not in stopwords.words('english')]

text_trigrams = [i for i in ngrams(text, 3)]

Counter(text_trigrams).most_common(30)
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

vectorizer.fit(full_text)

train_vectorized = vectorizer.transform(train['Phrase'])

test_vectorized = vectorizer.transform(test['Phrase'])
y = train['Sentiment']
logreg = LogisticRegression()

ovr = OneVsRestClassifier(logreg)
%%time

ovr.fit(train_vectorized, y)
scores = cross_val_score(ovr, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)

print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
%%time

svc = LinearSVC(dual=False)

scores = cross_val_score(svc, train_vectorized, y, scoring='accuracy', n_jobs=-1, cv=3)

print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))
ovr.fit(train_vectorized, y)

svc.fit(train_vectorized, y)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU 

from keras.layers import CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
tk = Tokenizer(lower = True, filters = '')

tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train['Phrase'])

test_tokenized = tk.texts_to_sequences(test['Phrase'])
max_len = 50

X_train = pad_sequences(train_tokenized, maxlen = max_len)

X_test = pad_sequences(test_tokenized, maxlen = max_len)
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
embed_size = 300

max_features = 30000


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, 

                 dense_units=128, dr=0.0,conv_size=32):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

    

    inp = Input(shape = (max_len,))

    x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)

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

    x = Dense(5, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.1, 

                        verbose = 1, callbacks = [check_point, early_stop])

    model = load_model(file_path)

    return model
model1 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 64, spatial_dr = 0.3, kernel_size1=3, 

                      kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)
model2 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 128, 

                      spatial_dr = 0.5, kernel_size1=3, kernel_size2=3, dense_units=64, dr=0.2, conv_size=32)
def build_model2(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                                  save_best_only = True, mode = "min")

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)



    inp = Input(shape = (max_len,))

    x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x1 = SpatialDropout1D(spatial_dr)(x)



    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)

    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)

    

    x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)

    avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)

    max_pool1_gru = GlobalMaxPooling1D()(x_conv1)

    

    x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)

    avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)

    max_pool2_gru = GlobalMaxPooling1D()(x_conv2)

    

    

    x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)

    avg_pool1_lstm = GlobalAveragePooling1D()(x_conv3)

    max_pool1_lstm = GlobalMaxPooling1D()(x_conv3)

    

    x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)

    avg_pool2_lstm = GlobalAveragePooling1D()(x_conv4)

    max_pool2_lstm = GlobalMaxPooling1D()(x_conv4)

    

    

    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru,

                    avg_pool1_lstm, max_pool1_lstm, avg_pool2_lstm, max_pool2_lstm])

    x = BatchNormalization()(x)

    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))

    x = BatchNormalization()(x)

    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))

    x = Dense(5, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.1, 

                        verbose = 1, callbacks = [check_point, early_stop])

    model = load_model(file_path)

    return model
model3 = build_model2(lr = 1e-4, lr_d = 0, units = 64, spatial_dr = 0.5, kernel_size1=4, 

                      kernel_size2=3, dense_units=32, dr=0.1, conv_size=32)
model4 = build_model2(lr = 1e-3, lr_d = 0, units = 64, spatial_dr = 0.5, 

                      kernel_size1=3, kernel_size2=3, dense_units=64, dr=0.3, conv_size=32)
model5 = build_model2(lr = 1e-3, lr_d = 1e-7, units = 64, spatial_dr = 0.3, 

                      kernel_size1=3, kernel_size2=3, dense_units=64, dr=0.4, conv_size=64)
pred1 = model1.predict(X_test, batch_size = 1024, verbose = 1)

pred = pred1

pred2 = model2.predict(X_test, batch_size = 1024, verbose = 1)

pred += pred2

pred3 = model3.predict(X_test, batch_size = 1024, verbose = 1)

pred += pred3

pred4 = model4.predict(X_test, batch_size = 1024, verbose = 1)

pred += pred4

pred5 = model5.predict(X_test, batch_size = 1024, verbose = 1)

pred += pred5

#predictions = np.round(np.argmax(pred, axis=1)).astype(int)

#sub['Sentiment'] = predictions

#sub.to_csv("blend.csv", index=False)
class Attention(Layer):

    """

    Keras Layer that implements an Attention mechanism for temporal data.

    Supports Masking.

    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

    # Input shape

        3D tensor with shape: `(samples, steps, features)`.

    # Output shape

        2D tensor with shape: `(samples, features)`.

    :param kwargs:

    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

    The dimensions are inferred based on the output shape of the RNN.

    Example:

        model.add(LSTM(64, return_sequences=True))

        model.add(Attention())

    """

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None

        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:

            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:

            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim
def build_model(maxlen, max_features, embed_size, embedding_matrix):

    input_words = Input((max_len, ))

    x_words = Embedding(19479,

                        embed_size,

                        weights=[embedding_matrix],

                        mask_zero=True,

                        trainable=False)(input_words)

    x_words = SpatialDropout1D(0.2)(x_words)

    x_words = Bidirectional(LSTM(128, return_sequences=True))(x_words)

    x_words = Bidirectional(LSTM(128, return_sequences=True))(x_words)

    

    x = Attention(maxlen)(x_words)

    #x = GlobalMaxPooling1D()(x)

    #x = GlobalAveragePooling1D()(x)

    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)

    x = Dropout(0.2)(x)

    pred = Dense(5, activation='softmax')(x)



    model = Model(inputs=input_words, outputs=pred)

    return model



model = build_model(max_len, max_features, embed_size, embedding_matrix)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
save_file = 'model_by_tyk.h5'

#history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.2, 

                        #verbose = 1, callbacks = [check_point, early_stop])

history = model.fit(X_train, y_ohe,

                    epochs=20, verbose=1,

                    batch_size=512, shuffle=True)
pred = model.predict(X_test, batch_size = 1024, verbose = 1)
sub['Sentiment'] = pred

sub.to_csv("blend.csv", index=False)