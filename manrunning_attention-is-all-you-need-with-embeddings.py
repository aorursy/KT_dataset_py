# import some libaries
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from scikitplot.metrics import plot_confusion_matrix

from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Add, Bidirectional, CuDNNLSTM, Dense, Input, Embedding,merge, BatchNormalization, Reshape
from keras.models import Model
# load train and test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train datasets shape:", train.shape)
print("Test datasets shape:", test.shape)
# Here I will split the data to train and validation data
train_data, validation_data = train_test_split(train, test_size=.1, random_state=1234)

# Here I will use Tokenizer to extract the keyword vector as baseline
# I will use train data to fit the Tokenizer, then use this Tokenizer to extract the validation data
max_length = 100
max_features = 50000
token = Tokenizer(num_words=max_features)
token.fit_on_texts(list(np.asarray(train_data.question_text)))
xtrain = token.texts_to_sequences(np.asarray(train_data.question_text))
xvalidate = token.texts_to_sequences(np.asarray(validation_data.question_text))
xtest = token.texts_to_sequences(np.asarray(test.question_text))

# Because Tokenizer will split the sentence, for some sentence are smaller,
# so we have to pad the missing position
xtrain = pad_sequences(xtrain, maxlen=max_length)
xvalidate = pad_sequences(xvalidate, maxlen=max_length)
xtest = pad_sequences(xtest, maxlen=max_length)

ytrain = train_data.target
yvaliate = validation_data.target
# Here I write a helper function to evaluate model
def evaluate(y, pred):
    f1_list = list()
    thre_list = np.arange(0.1, 0.501, 0.01)
    for thresh in thre_list:
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(y, (pred>thresh).astype(int))
        f1_list.append(f1)
        print("F1 score at threshold {0} is {1}".format(thresh, f1))
    #return f1_list
    plot_confusion_matrix(y, np.array(pd.Series(pred.reshape(-1,)).map(lambda x:1 if x>thre_list[np.argmax(f1_list)] else 0)))
    print('Best Threshold: ',thre_list[np.argmax(f1_list)])
    return thre_list[np.argmax(f1_list)]
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
em_file = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*d.split(' ')) for d in open(em_file))

all_embs = np.stack(embedding_index.values())
em_mean, em_std = all_embs.mean(), all_embs.std()
em_size = all_embs.shape[1]

word_index = token.word_index
nb_words = min(max_features, len(word_index))
em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# loop for every word
for word, i in word_index.items():
    if i >= max_features: continue
    em_v = embedding_index.get(word)
    if em_v is not None:
        em_matrix[i] = em_v
    
"""Here is attention model parameters"""
# You can tune attention layers numbers and parallism attention
atten_layers = 1    # How many attention block to be used.
num_att = 10        # How many parallism attention layer to be used.
lstm_units = 128    # How many LSTM units to be used.
### Here I will build a Attention model
def attention_model(em_matrix, atten_layers=atten_layers, num_att=num_att, lstm_units=lstm_units):
    inp = Input(shape=(max_length, ))
    x = Embedding(max_features, em_matrix.shape[1], weights=[em_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True))(x)

    # Here I build a bisic Attention block
    def attention_block(layer, atten_len=max_length, num_multi=num_att, dense_units=lstm_units*2):
        atten_list = []
        for _ in range(num_multi):
            atten_list.append(Attention(atten_len)(layer))
        add_layer = Add()(atten_list)
        add_layer = BatchNormalization()(add_layer)
        dense_layer = Dense(dense_units, activation='relu')(add_layer)
        return Add()([add_layer, dense_layer])   # Residual add layer

    for j in range(atten_layers):
        if j ==0:
            x = attention_block(x)
        else:
            x = Reshape((lstm_units*2, 1))(x)
            x = attention_block(x, atten_len=lstm_units*2)
    
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)

    model.summary()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model
# I build this model based on wide&deep model structure idea
from keras.layers import concatenate
def att_new_model(em_matrix, atten_layers=atten_layers, num_att=num_att, lstm_units=lstm_units):
    inp = Input(shape=(max_length, ))
    x = Embedding(max_features, em_matrix.shape[1], weights=[em_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True))(x)

    # Here I build a bisic Attention block
    def attention_block(layer, atten_len=max_length, num_multi=num_att, dense_units=lstm_units*2):
        add_layer_list = []
        for j in range(num_att):
            atten_list = list()
            for _ in range(num_multi):
                atten_list.append(Attention(atten_len)(layer))
            add_layer_list.append(Add()(atten_list))
        add_layer = concatenate(add_layer_list)
        add_layer = BatchNormalization()(add_layer)
        dense_layer = Dense(dense_units, activation='relu')(add_layer)
        return dense_layer
        # return Add()([add_layer, dense_layer])   # Residual add layer

#     for j in range(atten_layers):
#         if j ==0:
#             x = attention_block(x)
#         else:
#             x = Reshape((lstm_units*2, 1))(x)
#             x = attention_block(x, atten_len=lstm_units*2)
    x = attention_block(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)

    model.summary()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model
# model = att_new_model(em_matrix=em_matrix, atten_layers=10, lstm_units=64, num_att=10)
# model.fit(xtrain, ytrain, epochs=2, batch_size=512, validation_data=(xvalidate, yvaliate), verbose=1)

# pred_vali_glove = model.predict(xvalidate, batch_size=1024)
# best_thre = evaluate(yvaliate, pred_vali_glove)

# pred_test_glove = model.predict(xtest)
model = attention_model(em_matrix=em_matrix, atten_layers=1, lstm_units=64, num_att=10)
model.fit(xtrain, ytrain, epochs=2, batch_size=512, validation_data=(xvalidate, yvaliate), verbose=1)

pred_vali_glove = model.predict(xvalidate, batch_size=1024)
best_thre = evaluate(yvaliate, pred_vali_glove)

pred_test_glove = model.predict(xtest)
del embedding_index, all_embs, word_index, em_matrix
import gc
gc.collect()
time.sleep(10)
# em_file = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
# def get_coefs(word, *arr):
#     return word, np.asarray(arr, dtype='float32')
# embedding_index = dict(get_coefs(*o.split(" ")) for o in open(em_file) if len(o)>100)

# all_embs = np.stack(embedding_index.values())
# em_mean, em_std = all_embs.mean(), all_embs.std()
# em_size = all_embs.shape[1]

# word_index = token.word_index
# nb_words = min(max_features, len(word_index))
# em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# # loop for every word
# for word, i in word_index.items():
#     if i >= max_features: continue
#     em_v = embedding_index.get(word)
#     if em_v is not None:
#         em_matrix[i] = em_v
# model = attention_model(em_matrix=em_matrix)
# model.fit(xtrain, ytrain, epochs=2, batch_size=512, validation_data=(xvalidate, yvaliate), verbose=1)

# pred_vali_wiki = model.predict(xvalidate, batch_size=1024)
# best_thre = evaluate(yvaliate, pred_vali_wiki)

# pred_test_wiki = model.predict(xtest)
# del embedding_index, all_embs, word_index, em_matrix
# gc.collect()
# time.sleep(10)
em_file = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(em_file, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embedding_index.values())
em_mean, em_std = all_embs.mean(), all_embs.std()
em_size = all_embs.shape[1]

word_index = token.word_index
nb_words = min(max_features, len(word_index))
em_matrix = np.random.normal(em_mean, em_std, (nb_words, em_size))
# loop for every word
for word, i in word_index.items():
    if i >= max_features: continue
    em_v = embedding_index.get(word)
    if em_v is not None:
        em_matrix[i] = em_v
    

model = attention_model(em_matrix=em_matrix)
model.fit(xtrain, ytrain, epochs=2, batch_size=512, validation_data=(xvalidate, yvaliate), verbose=1)

pred_vali_para = model.predict(xvalidate, batch_size=1024)
best_thre = evaluate(yvaliate, pred_vali_para)

pred_test_para = model.predict(xtest)
del embedding_index, all_embs, word_index, em_matrix
gc.collect()
time.sleep(10)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

model_num = 2

pred_data = np.empty([len(pred_vali_para), model_num])
pred_data[:, 0] = pred_vali_glove.reshape(-1, )
#pred_data[:, 1] = pred_vali_wiki.reshape(-1, )
pred_data[:, 1] = pred_vali_para.reshape(-1, )

lr.fit(pred_data, yvaliate)

weights = lr.coef_

sub_pred_weighted = np.sum([pred_data[:, i]*weights[i] for i in range(model_num)], axis=0)
best_thre = evaluate(yvaliate ,sub_pred_weighted)
# GET different model prediction result on test datasets
sub_data = np.empty([len(xtest), model_num])
sub_data[:, 0] = pred_test_glove.reshape(-1, )
# sub_data[:, 1] = pred_test_wiki.reshape(-1, )
sub_data[:, 1] = pred_test_para.reshape(-1, )

#sub_pred = 0.1 * pred_lstm_glove + 0.2*pred_bidi_lstm_glove + 0.1*pred_lstm_wiki + 0.2*pred_bidi_lstm_wiki+0.1*pred_lstm_para+0.3*pred_lstm_para
# According to Linear Regression model result with different weights multiply with prediction.
sub_pred = np.sum([sub_data[:, i]*weights[i] for i in range(model_num)], axis=0)
sub_pred = (sub_pred > best_thre).astype(int)

sub_df = pd.DataFrame({'qid':test.qid.values})
sub_df['prediction'] = sub_pred
sub_df.to_csv('submission.csv', index=False)
