#import Libraries
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,LSTM,GRU
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.optimizers import Adam, Adadelta
from keras.initializers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import gensim.models.keyedvectors as word2vec

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
# PREPROCESSING PART
fill = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
       "can't've": "cannot have", "'cause": "because", "could've": "could have", 
       "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
       "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
       "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
       "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
       "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
       "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
       "I'll've": "I will have","I'm": "I am", "I've": "I have", 
       "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
       "i'll've": "i will have","i'm": "i am", "i've": "i have", 
       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
       "it'll": "it will", "it'll've": "it will have","it's": "it is", 
       "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
       "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
       "she's": "she is", "should've": "should have", "shouldn't": "should not", 
       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
       "this's": "this is",
       "that'd": "that would", "that'd've": "that would have","that's": "that is", 
       "there'd": "there would", "there'd've": "there would have","there's": "there is", 
       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
       "they'll've": "they will have", "they're": "they are", "they've": "they have", 
       "to've": "to have", "wasn't": "was not", "we'd": "we would", 
       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
       "we're": "we are", "we've": "we have", "weren't": "were not", 
       "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
       "what's": "what is", "what've": "what have", "when's": "when is", 
       "when've": "when have", "where'd": "where did", "where's": "where is", 
       "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
       "who's": "who is", "who've": "who have", "why's": "why is", 
       "why've": "why have", "will've": "will have", "won't": "will not", 
       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
       "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

import re, string
from nltk.tokenize import TweetTokenizer    
from nltk.tokenize import word_tokenize
tokenizer=TweetTokenizer()
def clean_text(text):    
    #fixing apostrope
    text = text.replace("’", "'")
    #to lower
    text = text.lower()
    #remove \n
    text = re.sub("\\n","",text)

    # remove leaky elements like ip,user
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",text)
    
    
    #Split the sentences into words
    words = tokenizer.tokenize(text)
    # (')aphostophe  replacement (ie)   you're --> you are  
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words = [fill[word] if word in fill else word for word in words]
    #words = [lem.lemmatize(word, "v") for word in words]
    #words = [i for i in text.split() if i not in eng_stopwords]
    text = " ".join(words)
    return text
# clean the comment_text in train_df
cleaned_train_comment = []
for i in range(0,len(train)):
    cleaned_comment = clean_text(train['question_text'][i])
    cleaned_train_comment.append(cleaned_comment)
train['question_text'] = pd.Series(cleaned_train_comment).astype(str)
# clean the comment_text in train_df
cleaned_test_comment = []
for i in range(0,len(test)):
    cleaned_comment = clean_text(test['question_text'][i])
    cleaned_test_comment.append(cleaned_comment)
test['question_text'] = pd.Series(cleaned_test_comment).astype(str)
X_train = train["question_text"].fillna("fillna").values
y_train = train["target"].values
X_test = test["question_text"].fillna("fillna").values
## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
tok = text.Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tok.word_index
print('Found %s unique tokens.' % len(word_index))
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
print("Embedding matrix Shape : ",embedding_matrix.shape)
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95,random_state=123)
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
filter_sizes = [1,2,3,5]
num_filters = 36

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Reshape((maxlen, embed_size, 1))(x)

maxpool_pool = []
for i in range(len(filter_sizes)):
    conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                 kernel_initializer='he_normal', activation='elu')(x)
    maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

z = Concatenate(axis=1)(maxpool_pool)   
z = Flatten()(z)
z = Dropout(0.1)(z)

outp = Dense(1, activation="sigmoid")(z)

model = Model(inputs=inp, outputs=outp)
# compile the model
#Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Adadelta_opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
"""
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
"""
""" 
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
            """
""" 
file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,save_best_only = True, mode = "min")
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
"""
history = model.fit(X_tra, y_tra, batch_size = 512, epochs = 2, validation_data = (X_val, y_val), 
                    verbose = 1)#callbacks = [ra_val, check_point, early_stop])
pred_cnn_val_y = model.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_cnn_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)
pred_cnn_test_y = model.predict([x_test], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
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


EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tok.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        

inp =Input(shape=(maxlen, ))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(CuDNNGRU(128, kernel_initializer=glorot_normal(seed=12300),return_sequences=True))(x)
x = Bidirectional(CuDNNGRU(64, kernel_initializer=glorot_normal(seed=12300),return_sequences=True))(x)
x = Conv1D(128, kernel_size = 3, activation='relu', padding = "valid")(x)
x = Conv1D(128, kernel_size = 3, activation='relu', padding = "valid")(x)
x = Attention(66)(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation="sigmoid")(x)
# compile the model
model = Model(inputs=inp, outputs=x)
#Adam_opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#Adadelta_opt = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
"""
file_path1 = "best_model1.hdf5"
check_point = ModelCheckpoint(file_path1, monitor = "val_loss", verbose = 1,save_best_only = True, mode = "min")
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
"""
history = model.fit(X_tra, y_tra, batch_size = 1024, epochs = 4, validation_data = (X_val, y_val), 
                    verbose = 1)#callbacks = [ra_val, check_point, early_stop])
pred_glove_val_y = model.predict([X_val], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_glove_val_y>thresh).astype(int))))
pred_glove_test_y = model.predict([x_test], batch_size=1024, verbose=1)
pred_val_y = (3 * pred_glove_val_y  + 2 * pred_cnn_val_y) / 5.0

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)
pred_test_y = (3 * pred_glove_test_y + 2 * pred_cnn_test_y) / 5.0
pred_test_y = (pred_test_y > best_thresh).astype(int)
submission['prediction'] = pred_test_y
submission.to_csv('submission.csv', index=False)

