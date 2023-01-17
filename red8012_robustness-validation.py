# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import itertools
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, 
    Conv1D ,GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Conv2D, MaxPool2D, concatenate,
    Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, Bidirectional, 
)
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer
from keras import metrics
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K

import matplotlib.pyplot as plt
%matplotlib inline
DEBUG = False
EMBED_SIZE = 300
MAX_FEATURES = 10000 if DEBUG else 90000
SEQUENCE_LENGTH = 50
GLOVE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
WIKI = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
PARAGRAM = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
if DEBUG:
    train_df = train_df[:100000]
train_df['question_text'] = train_df['question_text'].str.lower()
test_df['question_text'] = test_df['question_text'].str.lower()
DEBUG, train_df.shape, test_df.shape
def load_embedding(file, tokenizer):
    def split(line):
        word, *arr = line.split(' ')
        return word, np.asarray(arr, dtype='float32')
    with open(file, encoding='utf8', errors='ignore') as f:
        if DEBUG:
            embeddings = dict(split(line) for line in tqdm(itertools.islice(f, 10000)) if len(line) > 100)
        else:
            embeddings = dict(split(line) for line in tqdm(f) if len(line) > 100)
    values = np.stack(embeddings.values())
    mean = values.mean()
    std = values.std()
    n_words = min(MAX_FEATURES, len(tokenizer.word_index))
    
    embedding_matrix = np.random.normal(mean, std, (n_words, EMBED_SIZE))
    for word, i in tokenizer.word_index.items():
        if i < MAX_FEATURES and word in embeddings:
            embedding_matrix[i] = embeddings[word]
            
    return embedding_matrix
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
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
def model_gru_atten_3(embedding):
    input_ = Input(shape=(SEQUENCE_LENGTH,))
    x = Embedding(MAX_FEATURES, EMBED_SIZE, weights=[embedding])(input_)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(96, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(SEQUENCE_LENGTH)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
class AveragedModel:
    def __init__(self, tokenizer):
        self.embeddings = [
            load_embedding(GLOVE, tokenizer),
            load_embedding(WIKI, tokenizer),
            load_embedding(PARAGRAM, tokenizer),
        ]
        self.models = [model_gru_atten_3(embedding) for embedding in self.embeddings]
        
    def fit(self, *args, **kwargs):
        for model in self.models:
            model.fit(*args, **kwargs)
            
    def predict(self, *args, **kwargs):
        outputs = [model.predict(*args, **kwargs) for model in self.models]
        return np.array(outputs).mean(axis=0)
def plot_base():
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
import random
def drop_randomly(sequences, ratio=.1):
    for sequence in sequences:
        if random.random() < ratio and sequence:
            del sequence[random.randint(0, len(sequence) - 1)]
X = train_df["question_text"]
y = train_df['target'].values
predicted_y = np.zeros_like(y, dtype='float32')

indices = []
splitted_Xy = []
models = []

splitter = StratifiedKFold(2, random_state=0)


for train_index, test_index in splitter.split(X, y):
    indices.append((train_index, test_index))
    
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(X[train_index])
    
    train_X = tokenizer.texts_to_sequences(X[train_index])
    drop_randomly(train_X, ratio=.1) # test for robustness
    train_X = pad_sequences(train_X, maxlen=SEQUENCE_LENGTH)
    test_X = tokenizer.texts_to_sequences(X[test_index])
    test_X = pad_sequences(test_X, maxlen=SEQUENCE_LENGTH)
    train_y = y[train_index]
    test_y = y[test_index]

    splitted_Xy.append((train_X, test_X, train_y, test_y))
    model = AveragedModel(tokenizer)
    models.append(model)
for model, (train_X, test_X, train_y, test_y), (train_index, test_index) in zip(models, splitted_Xy, indices):
    model.fit(train_X, train_y, batch_size=512, epochs=1, verbose=0)
    
    yp = model.predict([test_X], batch_size=512, verbose=0).flatten()
    predicted_y[test_index] = yp
    
    
fpr, tpr, thresholds = roc_curve(y, predicted_y, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.3f)' % roc_auc)
plot_base()
