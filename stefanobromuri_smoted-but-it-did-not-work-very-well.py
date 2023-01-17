import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import concatenate

from keras.callbacks import *



iterations_learning = 24

batch_size_learning= 4096

importance = 0.9999

splits_kfold = 4



train_batch_size = 2048

train_epochs = 15





directory_emb = '../input/embeddings/'

directory_data = '../input/'



def load_glove(word_index):

    EMBEDDING_FILE = directory_emb+'glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.005838499,0.48782197

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

    EMBEDDING_FILE = directory_emb+'wiki-news-300d-1M/wiki-news-300d-1M.vec'

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

    EMBEDDING_FILE = directory_emb + 'paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.0053247833,0.49346462

    embed_size = all_embs.shape[1]

    print(emb_mean,emb_std,"para")



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix

import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, RepeatVector, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D, Permute, Lambda, Layer, BatchNormalization, GRU

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import Callback
import re

import time

import gc

import random

import os



import numpy as np

import pandas as pd



max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 72



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x



mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

train_df = pd.read_csv(directory_data+"train.csv")

test_df = pd.read_csv(directory_data+"test.csv")



#train_X, test_X, train_y, word_index = load_and_prec()



print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 

                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",

                       "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                       "I'm": "I am", "I've": "I have", "i'd": "i would", 

                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 

                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",

                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 

                       "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                       "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", 

                       "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 

                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 

                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 

                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 

                       "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                       "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 

                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 

                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",

                       "y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 

                       "you're": "you are", "you've": "you have" }



punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", 

                 "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", 

                 "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', 

                 "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', 

                 '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }







mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 

                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',

                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 

                'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 

                'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 

                'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',

                'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 

                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 

                'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'trump':'president', 'obama':'president', 'potus':'president'}









def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")





def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text





def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown





def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text



def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x





def transform_df(dfs):

    dfs['question_text'] = dfs['question_text'].apply(lambda x: x.lower())

    dfs['question_text'] = dfs['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

    dfs['question_text'] = dfs['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

    dfs['question_text'] = dfs['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))



    

transform_df(train_df)

transform_df(test_df)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 72 # max number of words in a question to use



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
def squash(x, axis=-1):

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)



# DropConnect

# https://github.com/andry9454/KerasDropconnect



from keras.layers import Wrapper



class DropConnect(Wrapper):

    def __init__(self, layer, prob=1., **kwargs):

        self.prob = prob

        self.layer = layer

        super(DropConnect, self).__init__(layer, **kwargs)

        if 0. < self.prob < 1.:

            self.uses_learning_phase = True



    def build(self, input_shape):

        if not self.layer.built:

            self.layer.build(input_shape)

            self.layer.built = True

        super(DropConnect, self).build()



    def compute_output_shape(self, input_shape):

        return self.layer.compute_output_shape(input_shape)



    def call(self, x):

        if 0. < self.prob < 1.:

            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)

            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)

        return self.layer.call(x)



import numpy as np

import pandas as pd



from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, add, BatchNormalization

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import glorot_normal, orthogonal

from keras.layers import concatenate

from keras.callbacks import *



import tensorflow as tf



import re

import time

import os





from keras.layers.core import *



import tensorflow as tf

import keras.backend as K

from keras.losses import binary_crossentropy



def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def f1_loss(y_true, y_pred):

    

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)





# the 

def my_model_lstm(topic,CONTEXT_LENGTH, embedding_size, max_features, embedding_matrix):

        L = CONTEXT_LENGTH

        NEURONS = 128

        

        inp = Input(shape=(L, ))

        encoding_input = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

        drop_out = Dropout(0.1, name='dropout')(encoding_input)

        lstm_fwd = LSTM(NEURONS, return_sequences=True, name='lstm_fwd')(drop_out)

        lstm_bwd = LSTM(NEURONS, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)

        

        import keras

        bilstm = keras.layers.concatenate([lstm_fwd, lstm_bwd], name='bilstm')

        

        #bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='concat')

        drop_out = Dropout(0.1)(bilstm)

        

        topic_input = Input(shape=(topic,), name='top_in')

        attention_topic = Dense(NEURONS, activation='linear')(topic_input)

        attention_topic =  RepeatVector(L)(attention_topic) #bidirectional attention model

        #attention_topic = Flatten()(attention_topic)

        

        # compute importance for each step

        attention = Dense(NEURONS, activation='linear')(drop_out)

        #attention = Flatten()(attention)

        

        attention = keras.layers.add([attention,attention_topic])

        

        

        attention = Dense(1, activation='tanh')(attention) # sequence + topic

        attention = Flatten()(attention)

        

        attention = Activation('softmax')(attention)

        attention = RepeatVector(NEURONS*2)(attention) #bidirectional attention model

        attention = Permute([2, 1])(attention)

        

        sent_representation = keras.layers.multiply([drop_out, attention])

        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(NEURONS*2,))(sent_representation)

        sent_representation = BatchNormalization(axis=-1)(sent_representation)

        bottleneck_features = Dense(NEURONS, activation='relu', name='bottleneck')(sent_representation) #the projection for smoting  

        out = Dense(1, activation='sigmoid')(bottleneck_features)

        #out2 = Dense(expected_length, activation='softmax')(sent_representation)

        #output = [out,out2]

        model = Model(input=[inp,topic_input], output=out)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1])

        print(model.summary())

        return model  



    

       



def my_small_model(size_intermediate):

    inp = Input(shape=(size_intermediate,))

    bn = BatchNormalization(axis=-1)(inp)

    interm = Dense(64, activation='relu')(bn)

    drop_out = Dropout(0.1)(interm)

    interm = Dense(32, activation='relu')(drop_out)

    drop_out = Dropout(0.1)(interm)

    out = Dense(1, activation='sigmoid')(drop_out)

    model = Model(input=inp, output=out)

    model.compile(loss=[binary_crossentropy], optimizer='adam', metrics=[f1])

    return model      

   







def build_my_model(embedding_matrix):

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = SpatialDropout1D(rate=0.24)(x)

    x = Bidirectional(CuDNNLSTM(80, 

                                return_sequences=True, 

                                kernel_initializer=glorot_normal(seed=1029), 

                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(x)



    x_1 = Attention(maxlen)(x)

    x_1 = DropConnect(Dense(32, activation="relu"), prob=0.1)(x_1)

    

    x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)

    x_2 = Flatten()(x_2)

    x_2 = DropConnect(Dense(32, activation="relu"), prob=0.1)(x_2)



    conc = concatenate([x_1, x_2], name='bottleneck')

    

    # conc = add([x_1, x_2])

    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[f1])

    return model

    

    
np.add([[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]])
# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
#train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()

word_index = tokenizer.word_index

vocab = []

for w,k in word_index.items():

    vocab.append(w)

    if k >= max_features:

        break

embedding_matrix = load_glove(word_index)

embedding_matrix = np.add(embedding_matrix,load_fasttext(word_index))

embedding_matrix = np.add(embedding_matrix, load_para(word_index))

embedding_matrix = embedding_matrix/3



#embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)

np.shape(embedding_matrix)

embedding_matrix_1 = None

embedding_matrix_2 = None

np.shape(embedding_matrix)

'''EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



#add_lower(embeddings_index, vocab)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector'''
embedding_size = 300



#model = my_model_gru_2(topic,CONTEXT_LENGTH, embedding_size, max_features, embedding_matrix)

model = build_my_model(embedding_matrix)

model.summary()
bottleneck_layer = model.get_layer(name='bottleneck')

model_extractor = Model(inputs=model.get_input_at(0), outputs= [bottleneck_layer.get_output_at(0)])

model_extractor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_extractor.summary()
clr = CyclicLR(base_lr=0.001, max_lr=0.002,

               step_size=300., mode='exp_range',

               gamma=0.99994)
## Train the model 

from keras.optimizers import rmsprop

from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('mymodel20.wgt', save_best_only=True)





early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit([train_X], [train_y], batch_size=batch_size_learning,callbacks=[clr,check_point,early_stop], epochs=train_epochs, validation_data=([val_X], [val_y]))



#model.fit([train_X,data_train_lsi], [train_y,train_y], batch_size=2048,callbacks=[clr,check_point,early_stop], epochs=9, validation_data=([val_X,data_val_lsi], [val_y,val_y]))

#model.save_weights('mymodel20.wgt')
model.load_weights('mymodel20.wgt')
'''

def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



def train_pred(model, train_X,split_data_train_lsi, train_y, X_val,split_data_val_lsi, y_val, epochs=2, callback=None):

    model.fit([train_X,split_data_train_lsi], [train_y,train_y], batch_size=batch_size_learning,callbacks=[clr,early_stop], epochs=epochs, validation_data=([X_val,split_data_val_lsi], [y_val,y_val]))

    #pred_val_y = model.predict([X_val,split_data_val_lsi])

    #pred_val_y = importance*pred_val_y[0] + (1-importance)*pred_val_y[1]

        #model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0)

        #pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

        #mythres = threshold_search(y_val,pred_val_y)

        #thr = mythres['threshold']

    

    pred_val_y = model.predict([val_X,data_val_lsi], batch_size=1024, verbose=0)

    pred_val_y = pred_val_y[0]*importance + (1-importance)*pred_val_y[1]

    

    mythres = threshold_search(val_y,pred_val_y)

    thr = mythres['threshold']



    best_score = metrics.f1_score(val_y, (pred_val_y > thr).astype(int))    

    

    print("SPLIT Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([test_X,data_test_lsi], batch_size=1024, verbose=0)

    pred_test_y = pred_test_y[0]*(importance) + (1-importance)*pred_test_y[1]

    print('=' * 60)

    return pred_val_y, pred_test_y, best_score







DATA_SPLIT_SEED = 2018





valid_meta = np.zeros(val_X.shape[0])

test_meta = np.zeros(test_X.shape[0])

splits = list(StratifiedKFold(n_splits=splits_kfold, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))



count = 0

for idx, (train_idx, valid_idx) in enumerate(splits):

        X_train = train_X[train_idx]

        split_data_train_lsi = data_train_lsi[train_idx]

        y_train = train_y[train_idx]

        X_val = train_X[valid_idx]

        split_data_val_lsi = data_train_lsi[valid_idx]

        y_val = train_y[valid_idx]

        #model = model_lstm_atten(embedding_matrix)

        model = my_model_gru_2(topic,CONTEXT_LENGTH, embedding_size, max_features, embedding_matrix)

        pred_val_y, pred_test_y, best_score = train_pred(model, X_train,split_data_train_lsi, y_train, X_val,split_data_val_lsi, y_val, epochs = iterations_learning, callback = [clr])

        if best_score>0.67:

            valid_meta += pred_val_y.reshape(-1) #/len(splits)

            test_meta += pred_test_y.reshape(-1) #/ len(splits)

            count = count +1

            

valid_meta = valid_meta/count            

test_meta = test_meta/count



threshold_search(val_y,valid_meta)



mythres = threshold_search(val_y,valid_meta)

thr = mythres['threshold']

print('-'*60)

print('thr')

print(mythres)

'''
'''sub = pd.read_csv('../input/sample_submission.csv')

sub.prediction = test_meta > thr

sub.to_csv("submission.csv", index=False)'''
thresh_saved = 0.1

max_f1 = 0





pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

#pred_noemb_val_y_1 = model.predict([val_X,data_val_lsi], batch_size=1024, verbose=1)[1]



#pred_noemb_val_y =pred_noemb_val_y #[0]*importance + (1-importance)*pred_noemb_val_y[1] #0.99*pred_noemb_val_y[0] + 0.01*pred_noemb_val_y[1]

pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)



#pred_noemb_test_y =  #importance*pred_noemb_test_y[0] + (1-importance)*pred_noemb_test_y[1]

#pred_noemb_test_y = (pred_noemb_test_y>thresh_saved).astype(int)



#out_df = pd.DataFrame({"qid":test_df["qid"].values})

#out_df['prediction'] = pred_noemb_test_y

#out_df.to_csv("submission.csv", index=False)

#'''
embedding_matrix = None

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.preprocessing import StandardScaler



new_train_X = model_extractor.predict([train_X], batch_size=1024, verbose=1)

new_val_X = model_extractor.predict([val_X], batch_size=1024, verbose=1)

new_test_X= model_extractor.predict([test_X], batch_size=1024, verbose=1)

scl = StandardScaler()
new_train_X_scaled = scl.fit_transform(new_train_X)

new_val_X_scaled = scl.transform(new_val_X)

new_test_X_scaled = scl.transform(new_test_X)
print(np.shape(new_train_X))

print(np.shape(new_train_X_scaled))
print(len(val_y))
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from random import choice, sample



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

neg_sample = 500000

pos_sample = 40000



X_smote = None



X_smote_neg = np.array(sample(new_train_X_scaled[train_y==0,:].tolist(),neg_sample))

X_smote_pos = np.array(sample(new_train_X_scaled[train_y==1,:].tolist(),pos_sample))



y_neg = np.zeros(X_smote_neg.shape[0])

y_pos = np.zeros(X_smote_pos.shape[0])+1



to_smote = np.append(X_smote_neg, X_smote_pos,axis=0)

to_smote_y = np.append(y_neg, y_pos)



print(np.shape(to_smote),np.shape(to_smote_y))
sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(to_smote, to_smote_y)
#pos_sample = 40000



new_x_pos = X_train_res[y_train_res==1,:][pos_sample:]

new_y_pos = y_train_res[y_train_res==1][pos_sample:]



new_train_X_scaled_smoted=np.append(new_train_X_scaled,new_x_pos,axis=0)

train_y_smoted = np.append(train_y,new_y_pos)



print(np.shape(new_train_X_scaled_smoted))



print(np.shape(train_y_smoted))


#clr2 = CyclicLR(base_lr=0.001, max_lr=0.002,

#               step_size=300., mode='exp_range',

#               gamma=0.99994)



#mymod= my_small_model(64)

#mymod.fit(new_train_X_scaled_smoted,train_y_smoted, callbacks=[clr2,early_stop], epochs=20, batch_size=batch_size_learning, validation_data=([new_val_X_scaled], [val_y]))
#predicted_val_smoted_y = mymod.predict(new_val_X_scaled)

#pred_test_smoted_y = mymod.predict(new_test_X_scaled) # ensemble.predict_proba(new_test_X_scaled)[:,1]*0.5 + pred_noemb_test_y*0.5 #svmoc.predict_proba(new_test_X_scaled)[:,1]

#pred_extract_test_y = (pred_extract_test_y>thresh_saved2).astype(int)



check_point2 = ModelCheckpoint('intermediate.wgt', save_best_only=True)









def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



def train_pred(model, train_X, train_y, X_val, y_val, epochs=2, callback=None):

    model.fit([train_X], [train_y], batch_size=batch_size_learning,callbacks=[clr,early_stop,check_point2], epochs=epochs, validation_data=([X_val], [y_val]))

    model.load_weights('intermediate.wgt')

    pred_val_y = model.predict([new_val_X_scaled], batch_size=1024, verbose=0)

    

    mythres = threshold_search(val_y,pred_val_y)

    thr = mythres['threshold']



    best_score = metrics.f1_score(val_y, (pred_val_y > thr).astype(int))    

    

    print("SPLIT Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([new_test_X_scaled], batch_size=1024, verbose=0)

    #pred_test_y = pred_test_y #[0]*(importance) + (1-importance)*pred_test_y[1]

    print('=' * 60)

    return pred_val_y, pred_test_y, best_score







DATA_SPLIT_SEED = 2018





valid_meta = np.zeros(val_X.shape[0])

test_meta = np.zeros(test_X.shape[0])

splits = list(StratifiedKFold(n_splits=splits_kfold, shuffle=True, random_state=DATA_SPLIT_SEED).split(new_train_X_scaled_smoted, train_y_smoted))



count = 0

for idx, (train_idx, valid_idx) in enumerate(splits):

        X_train = new_train_X_scaled_smoted[train_idx]

        y_train = train_y_smoted[train_idx]

        X_val = new_train_X_scaled_smoted[valid_idx]

        y_val = train_y_smoted[valid_idx]

        #model = model_lstm_atten(embedding_matrix)

        model = my_small_model(64)

        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = iterations_learning, callback = [clr])

        #if best_score>0.675:

        valid_meta += pred_val_y.reshape(-1) #/len(splits)

        test_meta += pred_test_y.reshape(-1) #/ len(splits)

        count = count +1

            

valid_meta = valid_meta/count            

test_meta = test_meta/count



threshold_search(val_y,valid_meta)



mythres = threshold_search(val_y,valid_meta)

thr = mythres['threshold']

print('-'*60)

print('thr')

print(mythres)
predict_final = (test_meta>thr).astype(int)



out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = predict_final

out_df.to_csv("submission.csv", index=False)
'''import numpy as np

from sklearn.decomposition import PCA





pca = PCA(n_components=2)

X_proj = pca.fit_transform(new_train_X_scaled_smoted)

X_test_proj = pca.transform(new_val_X_scaled)



import matplotlib.pyplot as plt





plt.plot(X_proj[train_y_smoted==0,0],X_proj[train_y_smoted==0,1],'*r')



plt.plot(X_proj[train_y_smoted==1,0],X_proj[train_y_smoted==1,1],'*b')



plt.plot(X_test_proj[val_y==0,0],X_test_proj[val_y==0,1],'*g')



plt.plot(X_test_proj[val_y==1,0],X_test_proj[val_y==1,1],'*c')





plt.show()

'''
'''estimators = 500

thresh_saved2 = 0.1

max_f1 = 0



#balancing classifier

svmoc = BalancedRandomForestClassifier(n_estimators=estimators, class_weight='balanced') 

#svmoc.fit(new_train_X,train_y)'''
'''

import pandas

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier



estimators = []

model1 = LogisticRegression()

estimators.append(('logistic', model1))

model2 = DecisionTreeClassifier()

estimators.append(('cart', model2))

model3 = SVC()

estimators.append(('svm', model3))



model4 = KNeighborsClassifier(n_neighbors=3)



estimators.append(('knn', model4))



model5 = svmoc



estimators.append(('btree', model5))





# create the ensemble model

ensemble = VotingClassifier(estimators)

ensemble.fit(new_train_X,train_y)

'''







'''

predicted_val_y = ensemble.predict_proba(new_val_X_scaled)[:,1]



#predicted_val_y = 0.5*predicted_val_y + 0.5*pred_noemb_test_y



thresh_saved2 = 0.1

for thresh2 in np.arange(0.1, 1, 0.01):

    thresh2 = np.round(thresh2, 2)

    value = metrics.f1_score(val_y, (predicted_val_y>thresh2).astype(int))

    print("F1 score at threshold {0} is {1}".format(thresh2, value))

    if value> max_f1:

                max_f1 = value

                thresh_saved2=thresh2

print(thresh_saved2)    '''
#print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(test_Y, (pred_noemb_test_y>0.28).astype(int))))