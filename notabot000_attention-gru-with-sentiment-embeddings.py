# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import textblob
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/synthessence2k18/train.csv')
test = pd.read_csv('../input/synthessence2k18/test.csv')
train.head()
train = train.drop(['title', 'date', 'restaurant_average_prices', 'restaurant_cuisine', 'restaurant_features', 'restaurant_good_for', 'restaurant_meals', 'restaurant_open_hours_friday', 'restaurant_open_hours_monday', 'restaurant_open_hours_saturday', 'restaurant_open_hours_sunday', 'restaurant_open_hours_thursday', 'restaurant_open_hours_tuesday', 'restaurant_open_hours_wednesday'], axis=1)
test = test.drop(['title', 'date', 'restaurant_average_prices', 'restaurant_cuisine', 'restaurant_features', 'restaurant_good_for', 'restaurant_meals', 'restaurant_open_hours_friday', 'restaurant_open_hours_monday', 'restaurant_open_hours_saturday', 'restaurant_open_hours_sunday', 'restaurant_open_hours_thursday', 'restaurant_open_hours_tuesday', 'restaurant_open_hours_wednesday'], axis=1)
def fill_nans(df, cols):
    for col in cols:
        df[col].fillna(df[col].mean(), inplace=True)
cols = ['restaurant_rating_atmosphere', 'restaurant_rating_food', 'restaurant_rating_service', 'restaurant_rating_value']
fill_nans(train, cols)
fill_nans(test, cols)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
enc.fit(train["ratingValue_target"].values.reshape(-1, 1))
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer()
cv1.fit(train["text"])
cv2 = CountVectorizer()
cv2.fit(test["text"])
NUM_FOLDS = 8

train["id"] = train.index
train["fold_id"] = train['id'].apply(lambda x: x%NUM_FOLDS)
EMBEDDING_FILE = "../input/glove-stanford/glove.42B.300d.txt"
EMBEDDING_DIM = 300

all_words = set(cv1.vocabulary_.keys()).union(set(cv2.vocabulary_.keys()))

def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in all_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index

embeddings_index = get_embedding()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 60

tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.append(train["text"].values, test["text"].values))
word_index = tokenizer.word_index

nb_words = len(word_index) + 1
embedding_matrix = np.random.rand(nb_words, EMBEDDING_DIM + 2)

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    sent = textblob.TextBlob(word).sentiment
    if embedding_vector is not None:
        embedding_matrix[i] = np.append(embedding_vector, [sent.polarity, sent.subjectivity])
    else:
        embedding_matrix[i, -2:] = [sent.polarity, sent.subjectivity]
        
seq = pad_sequences(tokenizer.texts_to_sequences(train["text"]), maxlen=MAX_SEQUENCE_LENGTH)
test_seq = pad_sequences(tokenizer.texts_to_sequences(test["text"]), maxlen=MAX_SEQUENCE_LENGTH)
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


class Attention(Layer):
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
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping

def build_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM + 2,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    dropout = SpatialDropout1D(0.2)
    gru_layer = Bidirectional(GRU(64, return_sequences=True))
    attention = Attention(MAX_SEQUENCE_LENGTH)
    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    dense_input = Input(shape=(len(cols),))
    
    dense_vector = BatchNormalization()(dense_input)
    
    phrase_vector = gru_layer(dropout(embedding_layer(seq_input)))
    phrase_vector = attention(phrase_vector)
    feature_vector = concatenate([phrase_vector, dense_vector])
    feature_vector = Dense(100, activation="relu")(feature_vector)
    feature_vector = Dense(20, activation="relu")(feature_vector)
    
    output = Dense(5, activation="softmax")(feature_vector)
    
    model = Model(inputs=[seq_input, dense_input], outputs=output)
    return model
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
test_preds = np.zeros((test.shape[0], 5))

for i in range(NUM_FOLDS):
    print("FOLD", i+1)
    
    train_seq, val_seq = seq[train["fold_id"] != i], seq[train["fold_id"] == i]
    train_dense, val_dense = train[train["fold_id"] != i][cols], train[train["fold_id"] == i][cols]
    y_train = enc.transform(train[train["fold_id"] != i]["ratingValue_target"].values.reshape(-1, 1))
    y_val = enc.transform(train[train["fold_id"] == i]["ratingValue_target"].values.reshape(-1, 1))
    
    model = build_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[f1])
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, verbose=1)
    
    model.fit([train_seq, train_dense], y_train, validation_data=([val_seq, val_dense], y_val),
              epochs=15, batch_size=1024, shuffle=True, callbacks=[early_stopping], verbose=1)
    
    test_preds += model.predict([test_seq, test[cols]], batch_size=1024, verbose=1)
    print()
    
test_preds /= NUM_FOLDS
test["pred"] = test_preds.argmax(axis=1)

test["ratingValue_target"] = test["pred"].astype(int).apply(lambda x: x + 1)
test["id"] = test.index
test[["id", "ratingValue_target"]].to_csv("submission.csv", index=False)
