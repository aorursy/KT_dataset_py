# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

SEED = 2018

np.random.seed(SEED)
tf.set_random_seed(SEED)

from tqdm import tqdm
tqdm.pandas()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

import gc, re
from sklearn import metrics
from sklearn.model_selection import train_test_split
# https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing/notebook
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }    

def clean_text(x):
    for dic in [contraction_mapping, mispell_dict, punct_mapping]:
        for word in dic.keys():
            x = x.replace(word, dic[word])
    return x
def load_and_preprocess_data(max_features=50000, maxlen=70):
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

    train_df['question_text'] = train_df['question_text'].fillna("").apply(lambda x: clean_text(x))
    test_df['question_text'] = test_df['question_text'].fillna("").apply(lambda x: clean_text(x))
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values
    
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X) + list(test_X))
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
    
    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))

    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]    
    
    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
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

        self.W = self.add_weight('{}_W'.format(self.name), shape=(input_shape[-1].value,),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name), shape=(input_shape[1].value,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint
                                    )
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
# def cnn1d(embedding_matrix, maxlen=70, max_features=50000, units=512):
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp)
    
#     layer_conv3 = tf.keras.layers.Conv1D(units, 3, activation="relu")(x)
#     layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)

#     layer_conv4 = tf.keras.layers.Conv1D(units, 2, activation="relu")(x)
#     layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)

#     layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
#     layer = tf.keras.layers.BatchNormalization()(layer)
#     layer = tf.keras.layers.Dropout(0.1)(layer)

#     output = tf.keras.layers.Dense(1, activation="sigmoid")(layer)
    
#     model = Model(inputs=inp, outputs=output)
#     model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#     return model
def model_lstm_atten(embedding_matrix, maxlen=70, max_features=50000, units=64):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(units*2, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(units, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
def model_lstm_du(embedding_matrix, maxlen=70, max_features=50000, units=64):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    attn = Attention(maxlen)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool, attn])
    conc = Dense(units, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def model_gru_atten_3(embedding_matrix, maxlen=70, max_features=50000, units=64):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(units*2, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(units//2, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
def model_bilstm_2dcnn(embedding_matrix, maxlen=70, max_features=50000, units=64):
    conv_filters = 32
    inp = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(max_features, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Reshape((2 * maxlen, units, 1))(x)
    x = Conv2D(conv_filters, (3, 3))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    outp= Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model
def train_pred(model, train_X, train_y, val_X, val_y, epochs=2):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    print('='*100)
    return pred_val_y, pred_test_y, best_score
embed_size = 300
max_features = 95000
maxlen = 40
import time
start_time = time.time()

train_X, val_X, test_X, train_y, val_y, word_index = load_and_preprocess_data(max_features=max_features, maxlen=maxlen)

total_time = (time.time() - start_time)/60.0
print("Took {0} minutes".format(total_time))
# all_len = []

# for x in train_X:
#     l = 0
#     for dx in x:
#         if dx > 0:
#             l += 1
#     all_len.append(l)
# max_tokens = np.mean(all_len)+2 * np.std(all_len)
# print(max_tokens)
# p = np.sum(all_len < max_tokens)/len(all_len)
# print(p)
import time
start_time = time.time()

embedding_matrix_1 = load_glove(word_index)
embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)

total_time = (time.time() - start_time)/60.0
print("Took {0} minutes".format(total_time))
#embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)
embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3), axis=1)
print(np.shape(embedding_matrix))

del embedding_matrix_2
outputs = []
# model = cnn1d(embedding_matrix, maxlen=maxlen, max_features=max_features, units=128)
# pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 2)
# outputs.append([pred_val_y, pred_test_y, best_score, 'CNN 1D'])
model = model_lstm_atten(embedding_matrix, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM ATTN'])
model = model_lstm_atten(embedding_matrix_3, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM ATTN PARA'])
model = model_lstm_du(embedding_matrix, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 1)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU'])
model = model_lstm_du(embedding_matrix_1, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 1)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU GLOVE'])
model = model_lstm_du(embedding_matrix_3, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'LSTM DU PARA'])
model = model_gru_atten_3(embedding_matrix, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 2)
outputs.append([pred_val_y, pred_test_y, best_score, 'GRU ATTN'])
model = model_gru_atten_3(embedding_matrix_1, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, 'GRU ATTN GLOVE'])
model = model_bilstm_2dcnn(embedding_matrix, maxlen=maxlen, max_features=max_features, units=64)
pred_val_y, pred_test_y, best_score = train_pred(model, train_X, train_y, val_X, val_y, epochs = 3)
outputs.append([pred_val_y, pred_test_y, best_score, 'BILSTM CNN2D'])
weights = [i for i in range(1, len(outputs) + 1)]
weights = [float(i) / sum(weights) for i in weights] 
print(weights)
for output in outputs:
    print(output[2], output[3])
from sklearn.linear_model import LinearRegression
X = np.asarray([outputs[i][0] for i in range(len(outputs))])
X = X[...,0]
reg = LinearRegression().fit(X.T, val_y)
print(reg.score(X.T, val_y),reg.coef_)
#Weights
pred_val_y = np.sum([outputs[i][0] * weights[i] for i in range(0, len(outputs))], axis = 0)

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
print("Best threshold: ", thresholds[0][0])
# Mean
pred_val_y = np.mean([outputs[i][0] for i in range(len(outputs))], axis = 0)

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
thresholds.sort(key=lambda x: x[1], reverse=True)
print("Best threshold: ", thresholds[0][0])
# Regression Coeffecient
pred_val_y = np.sum([outputs[i][0] * reg.coef_[i] for i in range(0, len(outputs))], axis = 0)

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(val_y, (pred_val_y > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)
pred_test_y = np.sum([outputs[i][1] * reg.coef_[i] for i in range(0, len(outputs))], axis = 0)
#pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = (pred_test_y > best_thresh).astype(int)
test_df = pd.read_csv("../input/test.csv", usecols=["qid"])
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(out_df)
