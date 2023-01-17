import pandas as pd

from nltk import word_tokenize, sent_tokenize

from random import shuffle

from tqdm import tqdm as tqdm

from keras import backend as K

import numpy

from sklearn.metrics import f1_score



from keras.preprocessing.text import Tokenizer

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Bidirectional

from keras.models import Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.preprocessing import sequence

from keras.engine.topology import Layer



import tensorflow as tf
ukara = pd.read_csv("/kaggle/input/ukara-enhanced/dataset.csv")
from nltk import download

download('punkt')
import numpy as np 

import pandas as pd 

import gensim

import seaborn as sns

import nltk

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
model = gensim.models.Word2Vec.load("/kaggle/input/ukara-enhanced/idwiki_word2vec.model")
ukara = pd.read_csv("/kaggle/input/ukara-enhanced/dataset.csv")
ukara.head()
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()

tokenizer.fit_on_texts(ukara['teks'].values)

word_index=tokenizer.word_index
EMBEDDING_SIZE = 100
len_words=len(word_index)+1

embed_matrix=np.zeros((len_words,EMBEDDING_SIZE))

not_available = 0



for word,i in word_index.items():

    if i > len_words:

        continue

    try:

        embed_vector=model[word]

        embed_matrix[i]= (np.random.rand(1, EMBEDDING_SIZE) - 0.5) / 100

    except KeyError:

        not_available += 1
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



    def __init__(self,

                 W_regularizer=None, u_regularizer=None, b_regularizer=None,

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

        print(len(input_shape))

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
groups = {}



kelompok = [1, 3, 4, 7, 8, 9, 10, 'A', 'B']



for I in kelompok:

    groups[(I,0)] = ukara.query("kelompok == '%s' and label == 0 " % (str(I)))['teks'].values 



for I in kelompok:

    groups[(I,1)] = ukara.query("kelompok == '%s' and label == 1 " % (str(I)))['teks'].values



for T in groups:

    length = len(groups[T])

    groups[T] = [list(groups[T][length*(J)//5 : length*(J+1)//5]) for J in range(5)]



def generate_fold_data(test_index, sentence_preprocess):

    train = []

    test  = {}

    

    for T in groups:

        if T[0] not in test:

            test[T[0]] = []

        for index in range(5):

            if test_index == index:

                for M in groups[T][index]:

                    test[T[0]].append((sentence_preprocess(T, M), T[1]))

            else:

                for M in groups[T][index]:

                    train.append((sentence_preprocess(T, M), T[1]))

    

    shuffle(train)

    return train, test
file = open("/kaggle/input/ukara-enhanced/stopword_list.txt")

    

stopwords = [I.strip() for I in file.readlines()]



file.close()



def remove_stopwords(words):

    words = word_tokenize(words.lower())

    words = [I for I in words if I not in stopwords]

    return " ".join(words)
def append_id_short(group_id, sentence):

    dictionary = {1: "aaaa",

                 3: "bbbb",

                 4: "cccc",

                 7: "ffff",

                 8: "gggg",

                 9: "hhhh",

                 10: "iiii",

                 "A": "jjjj",

                 "B": "kkkk"}

    return "%s %s" % (dictionary[group_id], sentence)
def preprocess_stop_short(group_id, sentence):

    sentence = append_id_short(group_id[0], sentence)

    sentence = remove_stopwords(sentence)

    return sentence



def preprocess_short(group_id, sentence):

    sentence = append_id_short(group_id[0], sentence)

    return sentence
MAX_REVIEW_LENGTH = 125

EPOCH = 3

BATCH_SIZE = 64
def generate_lstm(lstm_dropout, dropout, recurrent_dropout, lstm_output):



    model = Sequential()

    model.add(Embedding(len_words,EMBEDDING_SIZE,weights=[embed_matrix],trainable=False, input_length=MAX_REVIEW_LENGTH))

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(LSTM(lstm_output, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout))

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
def generate_bidirectional_lstm(lstm_dropout, dropout, recurrent_dropout, lstm_output):



    model = Sequential()

    model.add(Embedding(len_words,EMBEDDING_SIZE,weights=[embed_matrix],trainable=False, input_length=MAX_REVIEW_LENGTH))

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(lstm_output, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)))

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
def generate_attention_lstm(lstm_dropout, dropout, recurrent_dropout, lstm_output):



    model = Sequential()

    model.add(Embedding(len_words,EMBEDDING_SIZE,weights=[embed_matrix],trainable=False, input_length=MAX_REVIEW_LENGTH))

    model.add(LSTM(lstm_output, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))

    model.add(AttentionWithContext())

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
def evaluate_model(model_function, preprocessing_function):

    scores = {}

    for R in tqdm(range(5)):

        train, test = generate_fold_data(R, preprocessing_function)

        train_X = [I[0] for I in train]

        train_y = [I[1] for I in train]

        

        vector_train = tokenizer.texts_to_sequences(train_X)

        

        train_Z = sequence.pad_sequences(vector_train, maxlen=MAX_REVIEW_LENGTH)

        model = model_function()

        model.fit(train_Z, train_y,  epochs= EPOCH, batch_size= BATCH_SIZE, verbose=0)

        

        for J in test:

            test_X = [T[0] for T in test[J]]

            test_y = [T[1] for T in test[J]]



            

            vector_test = tokenizer.texts_to_sequences(test_X)



            test_Z = sequence.pad_sequences(vector_test, maxlen=MAX_REVIEW_LENGTH)         

            prediction = model.predict(test_Z)

            accuracy = f1_score(test_y, [I[0] > 0.5 for I in prediction])

            if J not in scores:

                scores[J] = 0

            scores[J] += accuracy / 5

    return scores
def lstm():

    return generate_lstm(lstm_dropout=0.2, dropout=0.1, 

                             recurrent_dropout=0.1, lstm_output=100)
def format_accuracy(dictionary_result):

    total = 0

    for I in dictionary_result:

        total += dictionary_result[I]

        print(I, dictionary_result[I])

    print("Macro All", total / len(dictionary_result))   
scores = evaluate_model(lstm, preprocess_stop_short)

format_accuracy(scores)