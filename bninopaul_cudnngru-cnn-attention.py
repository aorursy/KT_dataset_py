# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate
from keras.layers import ConvRNN2D, SpatialDropout1D, Reshape, MaxPool2D, Concatenate, Flatten, Conv2D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
%matplotlib inline

from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer

#GPU configs
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction = 1)
config = K.tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
K.set_session(K.tf.Session(config = config))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Source: https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
print([dev.name for dev in device_lib.list_local_devices()])
%%time
base_path = "../input/"
train_df = pd.read_csv(base_path+"train.csv")
test_df = pd.read_csv(base_path+"test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

print("Train: ", train_X.shape, train_y.shape)
print("Validation: ", val_X.shape, val_y.shape)
print("Test :", test_X.shape)
%%time
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

base_path = "../input/embeddings/"
files = {"glove": "glove.840B.300d/glove.840B.300d.txt",
         "wiki_news": "wiki-news-300d-1M/wiki-news-300d-1M.vec",
         "paragram": "paragram_300_sl999/paragram_300_sl999.txt"}

embedding_matrices = {}

for emb in files:
    if emb=="glove":
        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb]))
    elif emb=="wiki_news":
        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb]) \
                         if len(o)>100)
    elif emb=="paragram":
        emb_index = dict(get_coefs(*o.split(" ")) for o in open(base_path+files[emb],
                                                             encoding="utf8", 
                                                             errors='ignore') \
                         if len(o)>100)
    all_embs = np.stack(emb_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = emb_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    embedding_matrices[emb] = embedding_matrix
%%time
inp = Input(shape=(maxlen,))

inp_glove = Embedding(max_features, embed_size, weights=[embedding_matrices["glove"]])(inp)
inp_glove = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_glove)
inp_glove = GlobalMaxPool1D()(inp_glove)

inp_wiki = Embedding(max_features, embed_size, weights=[embedding_matrices["wiki_news"]])(inp)
inp_wiki = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_wiki)
inp_wiki = GlobalMaxPool1D()(inp_wiki)

inp_paragram = Embedding(max_features, embed_size, weights=[embedding_matrices["paragram"]])(inp)
inp_paragram = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_paragram)
inp_paragram = GlobalMaxPool1D()(inp_paragram)

merged  = concatenate([inp_glove, inp_wiki, inp_paragram])
x = Dense(32, activation="relu")(merged)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#train
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

#model performance 
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
f1_scores = []
threshs = np.arange(0.1, 0.9, 0.01)
for thresh in threshs:
    thresh = np.round(thresh, 2)
    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    f1_scores.append(f1_score)
    
plt.plot(threshs, f1_scores)
max_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)
max_thresh = np.round(threshs[np.argmax(f1_scores)], 3)
plt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))
thresh = max_thresh

#predictions on the validation set
pred_val_y_1 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_1 = (pred_val_y_1>=thresh).astype(int)

#predictions on the test set
pred_test_y_1 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_1 = (pred_test_y_1>=thresh).astype(int)
%%time
filter_sizes = [1, 2, 3, 5]
num_filt = 36

inp = Input(shape=(maxlen,))

embeddings = [Embedding(max_features, embed_size, weights=[embedding_matrices[emb]])(inp) \
              for emb in embedding_matrices]

embed_maxpools = []
for embed in embeddings:
    embed = SpatialDropout1D(0.1)(embed)
    embed = Reshape((maxlen, embed_size, 1))(embed)
    
    maxpools = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filt, kernel_size=(filter_sizes[i], embed_size),
                  kernel_initializer="he_normal", activation="elu")(embed)
        maxpools.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))
    merged = Concatenate(axis=1)(maxpools)
    merged = Flatten()(merged)
    merged = Dropout(0.2)(merged)
    
    embed_maxpools.append(merged)

embed_maxpools = Concatenate(axis=1)(embed_maxpools)
   
x = Dense(32, activation="relu")(embed_maxpools)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#train
model.fit(train_X, train_y, batch_size=256, epochs=2, validation_data=(val_X, val_y))

#model performance 
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
f1_scores = []
threshs = np.arange(0.1, 0.9, 0.01)
for thresh in threshs:
    thresh = np.round(thresh, 2)
    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    f1_scores.append(f1_score)
    
plt.plot(threshs, f1_scores)
max_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)
max_thresh = np.round(threshs[np.argmax(f1_scores)], 3)
plt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))
thresh = max_thresh

#predictions on the validation set
pred_val_y_2 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_2 = (pred_val_y_2>=thresh).astype(int)

#predictions on the test set
pred_test_y_2 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_2 = (pred_test_y_2>=thresh).astype(int)
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
%%time
inp = Input(shape=(maxlen,))

inp_glove = Embedding(max_features, embed_size, weights=[embedding_matrices["glove"]])(inp)
inp_glove = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_glove)
inp_glove = Attention(maxlen)(inp_glove)

inp_wiki = Embedding(max_features, embed_size, weights=[embedding_matrices["wiki_news"]])(inp)
inp_wiki = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_wiki)
inp_wiki = Attention(maxlen)(inp_wiki)

inp_paragram = Embedding(max_features, embed_size, weights=[embedding_matrices["paragram"]])(inp)
inp_paragram = Bidirectional(CuDNNGRU(128, return_sequences=True))(inp_paragram)
inp_paragram = Attention(maxlen)(inp_paragram)

merged  = concatenate([inp_glove, inp_wiki, inp_paragram])
x = Dense(32, activation="relu")(merged)
x = Dropout(0.1)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#train
model.fit(train_X, train_y, batch_size=256, epochs=2, validation_data=(val_X, val_y))

#model performance 
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
f1_scores = []
threshs = np.arange(0.1, 0.9, 0.01)
for thresh in threshs:
    thresh = np.round(thresh, 2)
    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    f1_scores.append(f1_score)
    
plt.plot(threshs, f1_scores)
max_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)
max_thresh = np.round(threshs[np.argmax(f1_scores)], 3)
plt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))
thresh = max_thresh

#predictions on the validation set
pred_val_y_3 = model.predict([val_X], batch_size=1024, verbose=1)
#pred_val_y_3 = (pred_val_y_3>=thresh).astype(int)

#predictions on the test set
pred_test_y_3 = model.predict([test_X], batch_size=1024, verbose=1)
#pred_test_y_3 = (pred_test_y_3>=thresh).astype(int)
pred_val_y = 0.5*pred_val_y_1 + 0.3*pred_val_y_2 + 0.2*pred_val_y_3
f1_scores = []
threshs = np.arange(0.1, 0.9, 0.01)
for thresh in threshs:
    thresh = np.round(thresh, 2)
    f1_score = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    f1_scores.append(f1_score)
    
plt.plot(threshs, f1_scores)
max_fscore = np.round(f1_scores[np.argmax(f1_scores)], 3)
max_thresh = np.round(threshs[np.argmax(f1_scores)], 3)
plt.title("F scores at different values of thresholds | Max: {} | Thresh {}".format(max_fscore, max_thresh))
pred_test_y = 0.5*pred_test_y_1 + 0.3*pred_test_y_2 + 0.2*pred_test_y_3
pred_test_y = (pred_test_y>=max_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
