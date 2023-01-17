#Ignoring the warnings

import warnings

warnings.filterwarnings('ignore')



#Importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re, string, unicodedata

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.pooling import GlobalMaxPooling1D

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

from keras.layers import *

from keras import backend

from sklearn.metrics import f1_score, confusion_matrix
#Importing the dataset



dataset = pd.read_csv('../input/imdb_master.csv', encoding = "ISO-8859-1")

dataset.head()
#Splitting into training and test set

dataset = dataset.drop(['Unnamed: 0', 'file'], axis = 1)

dataset = dataset[dataset.label != 'unsup']

dataset['label'] = dataset['label'].map({'pos': 1, 'neg': 0})

dataset_test = dataset[dataset['type'] == 'test']

dataset_train = dataset[dataset['type'] == 'train']

X_test = dataset_test.iloc[:, 1:2].values

y_test = dataset_test.iloc[:, 2].values

X_train = dataset_train.iloc[:, 1:2].values

y_train = dataset_train.iloc[:, 2].values
#Function for Text Preprocessing

stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()



def clean_text(X):

    processed = []

    for text in X:

        text = text[0]

        text = re.sub(r'[^\w\s]','',text, re.UNICODE)

        text = re.sub('<.*?>', '', text)

        text = text.lower()

        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

        text = [lemmatizer.lemmatize(token, "v") for token in text]

        text = [word for word in text if not word in stop_words]

        text = " ".join(text)

        processed.append(text)

    return processed
X_train_final = clean_text(X_train)

X_test_final = clean_text(X_test)
# Attention Layer

class AttentionWithContext(Layer):

    """

    Attention operation, with a context/query vector, for temporal data.

    Supports Masking.

    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

    "Hierarchical Attention Networks for Document Classification"

    by using a context vector to assist the attention

    # Input shape

        3D tensor with shape: `(samples, steps, features)`.

    # Output shape

        2D tensor with shape: `(samples, features)`.

    How to use:

    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:

        model.add(LSTM(64, return_sequences=True))

        model.add(AttentionWithContext())

        # next add a Dense layer (for classification/regression) or whatever...

    """



    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,

                 W_constraint=None, u_constraint=None, b_constraint=None,

                 bias=True, **kwargs):



        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.u_regularizer = regularizers.get(u_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.u_constraint = constraints.get(u_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        super(AttentionWithContext, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1], input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        if self.bias:

            self.b = self.add_weight((input_shape[-1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)



        self.u = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_u'.format(self.name),

                                 regularizer=self.u_regularizer,

                                 constraint=self.u_constraint)



        super(AttentionWithContext, self).build(input_shape)



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        uit = dot_product(x, self.W)



        if self.bias:

            uit += self.b



        uit = K.tanh(uit)

        ait = dot_product(uit, self.u)



        a = K.exp(ait)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[-1]



def dot_product(x, kernel):

    """

    Wrapper for dot product operation, in order to be compatible with both

    Theano and Tensorflow

    Args:

        x (): input

        kernel (): weights

    Returns:

    """

    if K.backend() == 'tensorflow':

        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

    else:

        return K.dot(x, kernel)
#Tokenization and Padding

vocab_size = 60000

maxlen = 250

encode_dim = 20

batch_size = 32

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train_final)

tokenized_word_list = tokenizer.texts_to_sequences(X_train_final)

X_train_padded = pad_sequences(tokenized_word_list, maxlen = maxlen, padding='post')
#EarlyStopping and ModelCheckpoint



es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)

mc = ModelCheckpoint('model_best.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)
#Building the model

model = Sequential()

embed = Embedding(input_dim = vocab_size, output_dim = 20, input_length = X_train_padded.shape[1], dropout = 0.4) 

model.add(embed)

model.add(Bidirectional(CuDNNLSTM(200, return_sequences = True)))

model.add(Dropout(0.3))

model.add(AttentionWithContext())

model.add(Dropout(0.5))

model.add(Dense(512))

model.add(LeakyReLU(alpha=0.2))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
from sklearn.model_selection import train_test_split

X_train_final2, X_val, y_train_final2, y_val = train_test_split(X_train_padded, y_train, test_size = 0.2)
#Fitting the model

model.fit(X_train_final2, y_train_final2, epochs = 50, batch_size = batch_size, verbose = 1, validation_data = [X_val, y_val], callbacks = [es, mc])
#Padding the test data

tokenized_word_list_test = tokenizer.texts_to_sequences(X_test_final)

X_test_padded = pad_sequences(tokenized_word_list_test, maxlen = maxlen, padding = 'post')
#Evaluating the model

from keras.models import load_model

model = load_model('model_best.h5', custom_objects = {"AttentionWithContext" : AttentionWithContext, "backend" : backend})

score, acc = model.evaluate(X_test_padded, y_test)

print('The accuracy of the model on the test set is ', acc*100)

prediction = model.predict(X_test_padded)

y_pred = (prediction > 0.5)

print('F1-score: ', (f1_score(y_pred, y_test)*100))

print('Confusion matrix:')

print(confusion_matrix(y_pred, y_test))