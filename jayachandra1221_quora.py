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
import gc
import re
import operator 
import string
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import svm
import nltk
from imblearn.over_sampling import SMOTE
import itertools
import re
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english')) 
from sklearn.preprocessing import LabelEncoder
def clean_text(x):
    x = str(x)
    for p in string.punctuation + '“…':
        x = x.replace(p, ' ' + p + ' ')
    
    x = x.replace('_', '')
    
    x = re.sub("`","'", x)
    x = re.sub("(?i)n\'t",' not', x)
    x = re.sub("(?i)\'re",' are', x)
    x = re.sub("(?i)\'s",' is', x)
    x = re.sub("(?i)\'d",' would', x)
    x = re.sub("(?i)\'ll",' will', x)
    x = re.sub("(?i)\'t",' not', x)
    x = re.sub("(?i)\'ve",' have', x)
    x = re.sub("(?i)\'m",' am', x)
    
    x = re.sub("(?i)n\’t",' not', x)
    x = re.sub("(?i)\’re",' are', x)
    x = re.sub("(?i)\’s",' is', x)
    x = re.sub("(?i)\’d",' would', x)
    x = re.sub("(?i)\’ll",' will', x)
    x = re.sub("(?i)\’t",' not', x)
    x = re.sub("(?i)\’ve",' have', x)
    x = re.sub("(?i)\’m",' am', x)
    
    x = re.sub('(?i)Quorans','Quora', x)
    x = re.sub('(?i)Qoura','Quora', x)
    x = re.sub('(?i)Quoran','Quora', x)
    x = re.sub('(?i)dropshipping','drop shipping', x)
    x = re.sub('(?i)HackerRank','Hacker Rank', x)
    x = re.sub('(?i)Unacademy','un academy', x)
    x = re.sub('(?i)eLitmus','India hire employees', x)
    x = re.sub('(?i)WooCommerce','Commerce', x)
    x = re.sub('(?i)hairfall','hair fall', x)
    x = re.sub('(?i)marksheet','mark sheet', x)
    x = re.sub('(?i)articleship','article ship', x)
    x = re.sub('(?i)cryptocurrencies','cryptocurrency', x)
    x = re.sub('(?i)coinbase','cryptocurrency', x)
    x = re.sub('(?i)altcoin','bitcoin', x)
    x = re.sub('(?i)altcoins','bitcoins', x)
    x = re.sub('(?i)litecoin','bitcoin', x)
    x = re.sub('(?i)litecoins','bitcoins', x)
    x = re.sub('(?i)demonetisation','demonetization', x)
    x = re.sub('(?i)ethereum','bitcoin', x)
    x = re.sub('(?i)ethereums','bitcoins', x)
    x = re.sub('(?i)quorans','quora', x)
    x = re.sub('(?i)Brexit','britan exit', x)
    x = re.sub('(?i)upwork','freelance', x)
    x = re.sub('(?i)Unacademy','un academy', x)
    x = re.sub('(?i)Blockchain','blockchain', x)
    x = re.sub('(?i)GDPR','General Data Protection Regulation', x)
    x = re.sub('(?i)Qoura','quora', x)
    x = re.sub('(?i)HackerRank','Hacker Rank', x)
    x = re.sub('(?i)Cryptocurrency','cryptocurrency', x)
    x = re.sub('(?i)Binance','cryptocurrency', x)
    x = re.sub('(?i)Redmi','mobile phone', x)
    x = re.sub('(?i)TensorFlow','Tensor Flow', x)
    x = re.sub('(?i)Golang','programming language', x)
    x = re.sub('(?i)eLitmus','India hire employees', x)
    
    return x
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print('Train shape : ', train_df.shape)
print('Test shape : ', test_df.shape)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2,random_state=123)
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas(desc="Example Desc")
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_text(x))
val_df['question_text'] = val_df['question_text'].progress_apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_text(x))
def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_text(x))
val_df['question_text'] = val_df['question_text'].progress_apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_text(x))
import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_text(x))
val_df['question_text'] = val_df['question_text'].progress_apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_text(x))
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: replace_typical_misspell(x))
val_df['question_text'] = val_df['question_text'].progress_apply(lambda x: replace_typical_misspell(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: replace_typical_misspell(x))
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

train_df['question_text'] = train_df['question_text'].str.lower()
val_df['question_text'] = val_df['question_text'].str.lower()
test_df['question_text'] = test_df['question_text'].str.lower()
    
train_df['question_text'] = train_df['question_text'].progress_apply(lambda x: clean_text(x))
val_df['question_text'] = val_df['question_text'].progress_apply(lambda x: clean_text(x))
test_df['question_text'] = test_df['question_text'].progress_apply(lambda x: clean_text(x))
train_df
full_text = list(train_df['question_text'].values) + list(val_df['question_text'].values)+list(test_df['question_text'].values)
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, BatchNormalization
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
tokenizer = Tokenizer(lower = True, filters = '')
tokenizer.fit_on_texts(full_text)
train_tokenized = tokenizer.texts_to_sequences(train_df['question_text'])
val_tokenized = tokenizer.texts_to_sequences(val_df['question_text'])
test_tokenized = tokenizer.texts_to_sequences(test_df['question_text'])
max_len = 70
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_val = pad_sequences(val_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)
X_train.shape
X_val.shape
X_test.shape
print(os.listdir("../input/embeddings"))
embedding_path = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
embed_size = 300
max_features = 100000 
def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')
def get_embed_mat(embedding_path):
    
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    return embedding_matrix
y = train_df['target']

#one_hot_encoder = OneHotEncoder(sparse=False)
#y_one_hot = one_hot_encoder.fit_transform(y.values.reshape(-1, 1))
y_val=val_df['target']
file_path = "model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 10)
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
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(100001, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(max_len)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    history = model.fit(X_train, y, batch_size = 256, epochs = 3, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    #model = load_model(file_path)
    return model
embedding_matrix = get_embed_mat(embedding_path)
model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.5)
#from keras.models import load_model
#model = load_model("../input/quora-new/model.hdf5")
pred = model.predict(X_val, batch_size = 1024, verbose = 1)
predictions = (pred > 0.35).astype(int)
predictions
from sklearn import metrics
print(metrics.accuracy_score(y_val,predictions))
print(metrics.f1_score(y_val,predictions))
sub
pred = model.predict(X_test, batch_size = 1024, verbose = 1)
predictions = (pred > 0.37).astype(int)
sub['prediction'] = predictions
sub.to_csv("submission.csv", index=False)
