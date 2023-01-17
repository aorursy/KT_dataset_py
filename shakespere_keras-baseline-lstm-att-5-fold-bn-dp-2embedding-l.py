import numpy as np 

import pandas as pd 

import os

from tqdm import tqdm

tqdm.pandas()
os.listdir('../input/')
import random

def set_seed(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



set_seed(2411)

SEED = 42

import psutil

from multiprocessing import Pool

import multiprocessing



num_partitions = 10  # number of partitions to split dataframe

num_cores = psutil.cpu_count()  # number of cores on your machine



print('number of cores:', num_cores)



def df_parallelize_run(df, func):

    

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    

    return df
TEXT_COL = 'comment_text'

EMB_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

glove_path = '../input/glove840b300dtxt/glove.840B.300d.txt'

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', index_col='id')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')

# From Quora kaggle Comp's (latest one)

import re

# remove space

spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']

def remove_space(text):

    """

    remove extra spaces and ending space if any

    """

    for space in spaces:

        text = text.replace(space, ' ')

    text = text.strip()

    text = re.sub('\s+', ' ', text)

    return text



# replace strange punctuations and raplace diacritics

from unicodedata import category, name, normalize



def remove_diacritics(s):

    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))

                  if category(c) != 'Mn')



special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',

                         "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',

                         '…': ' ... ', '\ufeff': ''}

def clean_special_punctuations(text):

    for punc in special_punc_mappings:

        if punc in text:

            text = text.replace(punc, special_punc_mappings[punc])

    text = remove_diacritics(text)

    return text



# clean numbers

def clean_number(text):

    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text) # digits followed by a single alphabet...

    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text) #1st, 2nd, 3rd, 4th...

    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)

    return text



import string

regular_punct = list(string.punctuation)

extra_punct = [

    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',

    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',

    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',

    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',

    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',

    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',

    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',

    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤',

    ':)', ': )', ':-)', '(:', '( :', '(-:', ':\')',

    ':D', ': D', ':-D', 'xD', 'x-D', 'XD', 'X-D',

    '<3', ':*',

    ';-)', ';)', ';-D', ';D', '(;',  '(-;',

    ':-(', ': (', ':(', '\'):', ')-:',

    '-- :','(', ':\'(', ':"(\'',]



def handle_emojis(text):

    # Smile -- :), : ), :-), (:, ( :, (-:, :')

    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)

    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D

    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)

    # Love -- <3, :*

    text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)

    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;

    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)

    # Sad -- :-(, : (, :(, ):, )-:

    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)

    # Cry -- :,(, :'(, :"(

    text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)

    return text



def stop(text):

    

    from nltk.corpus import stopwords

    

    text = " ".join([w.lower() for w in text.split()])

    stop_words = stopwords.words('english')

    

    words = [w for w in text.split() if not w in stop_words]

    return " ".join(words)



all_punct = list(set(regular_punct + extra_punct))

# do not spacing - and .

all_punct.remove('-')

all_punct.remove('.')



# clean repeated letters

def clean_repeat_words(text):

    

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)

    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)

    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)

    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)

    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)

    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)

    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)

    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)

    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)

    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)

    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)

    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)

    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)

    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)

    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)

    text = re.sub(r"(Q|q)(Q|q)+", "q", text)

    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)

    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)

    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)

    text = re.sub(r"(V|v)(V|v)+", "v", text)

    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)

    text = re.sub(r"plzz+", "please", text)

    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)

    text = re.sub(r"(-+|\.+)", " ", text) #new haha #this adds a space token so we need to remove xtra spaces

    return text



def spacing_punctuation(text):

    """

    add space before and after punctuation and symbols

    """

    for punc in all_punct:

        if punc in text:

            text = text.replace(punc, f' {punc} ')

    return text



def preprocess(text):

    """

    preprocess text main steps

    """

    text = remove_space(text)

    text = clean_special_punctuations(text)

    text = handle_emojis(text)

    text = clean_number(text)

    text = spacing_punctuation(text)

    text = clean_repeat_words(text)

    text = remove_space(text)

    #text = stop(text)# if changing this, then chnage the dims 

    #(not to be done yet as its effecting the embeddings..,we might be

    #loosing words)...

    return text



mispell_dict = {'😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',}

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x



def correct_contraction(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
from tqdm import tqdm

tqdm.pandas()



def text_clean_wrapper(df):

    

    df["comment_text"] = df["comment_text"].astype('str').transform(preprocess)

    df['comment_text'] = df['comment_text'].transform(lambda x: correct_spelling(x, mispell_dict))

    df['comment_text'] = df['comment_text'].transform(lambda x: correct_contraction(x, contraction_mapping))

    

    return df



#fast!

train = df_parallelize_run(train, text_clean_wrapper)

test  = df_parallelize_run(test, text_clean_wrapper)



import gc

gc.enable()

del mispell_dict, all_punct, special_punc_mappings, regular_punct, extra_punct

gc.collect()
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def load_embeddings(embed_dir=EMB_PATH):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

    return embedding_index



def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

    embedding_matrix = np.zeros((max_features, 300))

    for word, i in tqdm(word_index.items(),disable = not verbose):

        if lower:

            word = word.lower()

        if i >= max_features: continue

        try:

            embedding_vector = embeddings_index[word]

        except:

            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def build_matrix(word_index, embeddings_index):

    embedding_matrix = np.zeros((len(word_index) + 1,300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embeddings_index[word]

        except:

            embedding_matrix[i] = embeddings_index["unknown"]

    return embedding_matrix
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import gc



maxlen = 220

max_features = 100000

embed_size = 300

tokenizer = Tokenizer(num_words=max_features, lower=True) #filters = ''

#tokenizer = text.Tokenizer(num_words=max_features)

print('fitting tokenizer')

tokenizer.fit_on_texts(list(train[TEXT_COL]) + list(test[TEXT_COL]))

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(list(train[TEXT_COL]))

y_train = train['target'].values

X_test = tokenizer.texts_to_sequences(list(test[TEXT_COL]))



X_train = pad_sequences(X_train, maxlen=maxlen)

X_test = pad_sequences(X_test, maxlen=maxlen)



del tokenizer

gc.collect()
embeddings_index1 = load_embeddings()

embedding_matrix1 = build_matrix(word_index, embeddings_index1)

del embeddings_index1

gc.collect()
embeddings_index2 = load_embeddings(glove_path)

embedding_matrix2 = build_matrix(word_index, embeddings_index2)

del embeddings_index2

gc.collect()
embedding_matrix = embedding_matrix1*0.6 + embedding_matrix2*0.4
from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



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
gc.collect()
import keras.layers as L

from keras.models import Model

from keras.optimizers import Adam
def LSTM_ATT_BN(verbose = False, compile = True):

    sequence_input = L.Input(shape=(maxlen,), dtype='int32')

    embedding_layer = L.Embedding(len(word_index) + 1,

                                300,

                                weights=[embedding_matrix],

                                input_length=maxlen,

                                trainable=False)

    x = embedding_layer(sequence_input)

    x = L.SpatialDropout1D(0.2)(x)

    #x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)

    x = L.Bidirectional(L.CuDNNGRU(64, return_sequences=True))(x)



    #CuDNNGRU

    att = Attention(maxlen)(x)

    avg_pool1 = L.GlobalAveragePooling1D()(x)

    max_pool1 = L.GlobalMaxPooling1D()(x)

   

    x = L.concatenate([att,avg_pool1, max_pool1])

    x = L.Dense(128,activation='relu')(x)

    x = L.BatchNormalization()(x)

    preds = L.Dense(1, activation='sigmoid')(x)

    

    model = Model(sequence_input, preds)

    if verbose:

        model.summary()

    if compile:

        model.compile(loss='binary_crossentropy',optimizer=Adam(0.005),metrics=['acc'])

    return model

# # https://www.kaggle.com/yekenot/2dcnn-textclassifier

# def model_cnn():

#     filter_sizes = [1,2,3,5]

#     num_filters = 36



#     inp = L.Input(shape=(maxlen,))

#     x = L.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix])(inp)

#     x = L.Reshape((maxlen, 300, 1))(x)



#     maxpool_pool = []

#     for i in range(len(filter_sizes)):

#         conv = L.Conv2D(num_filters, kernel_size=(filter_sizes[i], 300),

#                                      kernel_initializer='he_normal', activation='elu')(x)

#         maxpool_pool.append(L.MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))



#     z = L.Concatenate(axis=1)(maxpool_pool)   

#     z = L.Flatten()(z)

#     z = L.Dropout(0.1)(z)



#     outp = L.Dense(1, activation="sigmoid")(z)



#     model = Model(inputs=inp, outputs=outp)

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

#     return model
# def model_lstm_atten():

#     inp = L.Input(shape=(maxlen,))

#     x = L.Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp)

#     x = L.Bidirectional(L.CuDNNLSTM(128, return_sequences=True))(x)

#     x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)

#     x = Attention(maxlen)(x)

#     x = L.Dense(64, activation="relu")(x)

#     x = L.Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inp, outputs=x)

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

#     return model
# def model_gru_srk_atten():

#     inp = L.Input(shape=(maxlen,))

#     x = L.Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)

#     x = L.Bidirectional(L.CuDNNGRU(64, return_sequences=True))(x)

#     x = Attention(maxlen)(x) # New

#     x = L.Dense(16, activation="relu")(x)

#     x = L.Dropout(0.1)(x)

#     x = L.Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inp, outputs=x)

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

#     return model 
# def model_lstm_du():

#     inp = L.Input(shape=(maxlen,))

#     x = L.Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix])(inp)

#     x = L.Bidirectional(L.CuDNNGRU(64, return_sequences=True))(x)

#     avg_pool = L.GlobalAveragePooling1D()(x)

#     max_pool = L.GlobalMaxPooling1D()(x)

#     conc = L.concatenate([avg_pool, max_pool])

#     conc = L.Dense(64, activation="relu")(conc)

#     conc = L.Dropout(0.1)(conc)

#     outp = L.Dense(1, activation="sigmoid")(conc)

    

#     model = Model(inputs=inp, outputs=outp)

#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model
def model_gru_atten_3():

    inp = L.Input(shape=(maxlen,))

    x = L.Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = L.Bidirectional(L.CuDNNGRU(128, return_sequences=True))(x)

    x = L.Bidirectional(L.CuDNNGRU(100, return_sequences=True))(x)

    x = L.Bidirectional(L.CuDNNGRU(64, return_sequences=True))(x)

    x = Attention(maxlen)(x)

    x = L.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
# # https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code



# from keras.callbacks import *

# class CyclicLR(Callback):

#     """This callback implements a cyclical learning rate policy (CLR).

#     The method cycles the learning rate between two boundaries with

#     some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

#     The amplitude of the cycle can be scaled on a per-iteration or 

#     per-cycle basis.

#     This class has three built-in policies, as put forth in the paper.

#     "triangular":

#         A basic triangular cycle w/ no amplitude scaling.

#     "triangular2":

#         A basic triangular cycle that scales initial amplitude by half each cycle.

#     "exp_range":

#         A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

#         cycle iteration.

#     For more detail, please see paper.

    

#     # Example

#         ```python

#             clr = CyclicLR(base_lr=0.001, max_lr=0.006,

#                                 step_size=2000., mode='triangular')

#             model.fit(X_train, Y_train, callbacks=[clr])

#         ```

    

#     Class also supports custom scaling functions:

#         ```python

#             clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

#             clr = CyclicLR(base_lr=0.001, max_lr=0.006,

#                                 step_size=2000., scale_fn=clr_fn,

#                                 scale_mode='cycle')

#             model.fit(X_train, Y_train, callbacks=[clr])

#         ```    

#     # Arguments

#         base_lr: initial learning rate which is the

#             lower boundary in the cycle.

#         max_lr: upper boundary in the cycle. Functionally,

#             it defines the cycle amplitude (max_lr - base_lr).

#             The lr at any cycle is the sum of base_lr

#             and some scaling of the amplitude; therefore 

#             max_lr may not actually be reached depending on

#             scaling function.

#         step_size: number of training iterations per

#             half cycle. Authors suggest setting step_size

#             2-8 x training iterations in epoch.

#         mode: one of {triangular, triangular2, exp_range}.

#             Default 'triangular'.

#             Values correspond to policies detailed above.

#             If scale_fn is not None, this argument is ignored.

#         gamma: constant in 'exp_range' scaling function:

#             gamma**(cycle iterations)

#         scale_fn: Custom scaling policy defined by a single

#             argument lambda function, where 

#             0 <= scale_fn(x) <= 1 for all x >= 0.

#             mode paramater is ignored 

#         scale_mode: {'cycle', 'iterations'}.

#             Defines whether scale_fn is evaluated on 

#             cycle number or cycle iterations (training

#             iterations since start of cycle). Default is 'cycle'.

#     """



#     def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

#                  gamma=1., scale_fn=None, scale_mode='cycle'):

#         super(CyclicLR, self).__init__()



#         self.base_lr = base_lr

#         self.max_lr = max_lr

#         self.step_size = step_size

#         self.mode = mode

#         self.gamma = gamma

#         if scale_fn == None:

#             if self.mode == 'triangular':

#                 self.scale_fn = lambda x: 1.

#                 self.scale_mode = 'cycle'

#             elif self.mode == 'triangular2':

#                 self.scale_fn = lambda x: 1/(2.**(x-1))

#                 self.scale_mode = 'cycle'

#             elif self.mode == 'exp_range':

#                 self.scale_fn = lambda x: gamma**(x)

#                 self.scale_mode = 'iterations'

#         else:

#             self.scale_fn = scale_fn

#             self.scale_mode = scale_mode

#         self.clr_iterations = 0.

#         self.trn_iterations = 0.

#         self.history = {}



#         self._reset()



#     def _reset(self, new_base_lr=None, new_max_lr=None,

#                new_step_size=None):

#         """Resets cycle iterations.

#         Optional boundary/step size adjustment.

#         """

#         if new_base_lr != None:

#             self.base_lr = new_base_lr

#         if new_max_lr != None:

#             self.max_lr = new_max_lr

#         if new_step_size != None:

#             self.step_size = new_step_size

#         self.clr_iterations = 0.

        

#     def clr(self):

#         cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

#         x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

#         if self.scale_mode == 'cycle':

#             return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

#         else:

#             return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

#     def on_train_begin(self, logs={}):

#         logs = logs or {}



#         if self.clr_iterations == 0:

#             K.set_value(self.model.optimizer.lr, self.base_lr)

#         else:

#             K.set_value(self.model.optimizer.lr, self.clr())        

            

#     def on_batch_end(self, epoch, logs=None):

        

#         logs = logs or {}

#         self.trn_iterations += 1

#         self.clr_iterations += 1



#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

#         self.history.setdefault('iterations', []).append(self.trn_iterations)



#         for k, v in logs.items():

#             self.history.setdefault(k, []).append(v)

        

#         K.set_value(self.model.optimizer.lr, self.clr())
# clr = CyclicLR(base_lr=0.001, max_lr=0.002,

#                step_size=300., mode='exp_range',

#                gamma=0.99994)
from sklearn.model_selection import KFold

splits = list(KFold(n_splits=5).split(X_train,y_train))

from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend as K

import numpy as np

BATCH_SIZE = 2048

NUM_EPOCHS = 10
outputs = []
# oof_preds = np.zeros((X_train.shape[0]))

# test_preds = np.zeros((X_test.shape[0]))

# for fold in [0,1]:

#     K.clear_session()

#     tr_ind, val_ind = splits[fold]

# #     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#     model = LSTM_ATT_BN()

#     model.fit(X_train[tr_ind],

#         y_train[tr_ind]>0.5,

#         batch_size=BATCH_SIZE,

#         epochs=NUM_EPOCHS,

#         validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

#         callbacks = [es])



#     oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

#     test_preds += model.predict(X_test)[:,0]

# test_preds /= 2

# outputs.append([test_preds, 'LSTM_ATT_BN'])

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train>0.5,oof_preds)
oof_preds = np.zeros((X_train.shape[0]))

test_preds = np.zeros((X_test.shape[0]))

for fold in [0,1,2,3,4]:

    K.clear_session()

    tr_ind, val_ind = splits[fold]

#     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

    model = model_gru_atten_3()

    model.fit(X_train[tr_ind],

        y_train[tr_ind]>0.5,

        batch_size=BATCH_SIZE,

        epochs=NUM_EPOCHS,

        validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

        callbacks = [es])



    oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

    test_preds += model.predict(X_test)[:,0]

test_preds /= 5

outputs.append([test_preds, '3 GRU w/ atten'])

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train>0.5,oof_preds)
# oof_preds = np.zeros((X_train.shape[0]))

# test_preds = np.zeros((X_test.shape[0]))

# for fold in [0,1]:

#     K.clear_session()

#     tr_ind, val_ind = splits[fold]

# #     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#     model = model_cnn()

#     model.fit(X_train[tr_ind],

#         y_train[tr_ind]>0.5,

#         batch_size=BATCH_SIZE,

#         epochs=NUM_EPOCHS,

#         validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

#         callbacks = [es])



#     oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

#     test_preds += model.predict(X_test)[:,0]

# test_preds /= 2

# outputs.append([test_preds, 'model_cnn'])

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train>0.5,oof_preds)
# oof_preds = np.zeros((X_train.shape[0]))

# test_preds = np.zeros((X_test.shape[0]))

# for fold in [0,1]:

#     K.clear_session()

#     tr_ind, val_ind = splits[fold]

# #     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#     model = model_lstm_atten()

#     model.fit(X_train[tr_ind],

#         y_train[tr_ind]>0.5,

#         batch_size=BATCH_SIZE,

#         epochs=NUM_EPOCHS,

#         validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

#         callbacks = [es])



#     oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

#     test_preds += model.predict(X_test)[:,0]

# test_preds /= 2

# outputs.append([test_preds, 'model_lstm_atten'])

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train>0.5,oof_preds)
# oof_preds = np.zeros((X_train.shape[0]))

# test_preds = np.zeros((X_test.shape[0]))

# for fold in [0,1]:

#     K.clear_session()

#     tr_ind, val_ind = splits[fold]

#     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#     model = model_gru_srk_atten()

#     model.fit(X_train[tr_ind],

#         y_train[tr_ind]>0.5,

#         batch_size=BATCH_SIZE,

#         epochs=NUM_EPOCHS,

#         validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

#         callbacks = [es,ckpt])



#     oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

#     test_preds += model.predict(X_test)[:,0]

# test_preds /= 2

# outputs.append([test_preds, 'model_gru_srk_atten'])

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train>0.5,oof_preds)
# oof_preds = np.zeros((X_train.shape[0]))

# test_preds = np.zeros((X_test.shape[0]))

# for fold in [0,1]:

#     K.clear_session()

#     tr_ind, val_ind = splits[fold]

#     ckpt = ModelCheckpoint(f'gru_{fold}.hdf5', save_best_only = True)

#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

#     model = model_lstm_du()

#     model.fit(X_train[tr_ind],

#         y_train[tr_ind]>0.5,

#         batch_size=BATCH_SIZE,

#         epochs=NUM_EPOCHS,

#         validation_data=(X_train[val_ind], y_train[val_ind]>0.5),

#         callbacks = [es,ckpt])



#     oof_preds[val_ind] += model.predict(X_train[val_ind])[:,0]

#     test_preds += model.predict(X_test)[:,0]

# test_preds /= 2

# outputs.append([test_preds, 'model_lstm_du'])

# from sklearn.metrics import roc_auc_score

# roc_auc_score(y_train>0.5,oof_preds)
pred_test= np.mean([outputs[i][0] for i in range(len(outputs))], axis = 0)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

submission['prediction'] = pred_test

submission.reset_index(drop=False, inplace=True)

submission.head()

#%%
submission.to_csv('submission.csv', index=False)