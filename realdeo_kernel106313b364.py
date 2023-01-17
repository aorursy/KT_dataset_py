import pandas as pd

from nltk import word_tokenize, sent_tokenize

from random import shuffle

from tqdm import tqdm as tqdm

from keras import backend as K

import numpy as np

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



def generate_fold_data(test_index, sentence_preprocess, size):

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

    train = train[:int(len(train) * size)]

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
MAX_REVIEW_LENGTH = 125

EMBEDDING_VECTOR_LENGTH = 32

TOP_WORDS = 4000

EPOCH = 3

BATCH_SIZE = 64
def generate_bidirectional_attention_lstm(lstm_dropout, dropout, recurrent_dropout, lstm_output):



    model = Sequential()

    model.add(Embedding(TOP_WORDS, EMBEDDING_VECTOR_LENGTH, input_length=MAX_REVIEW_LENGTH))

    model.add(Bidirectional(LSTM(lstm_output, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))

    model.add(AttentionWithContext())

    if dropout > 0:

        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
def bidirectional_attention_lstm():

    return generate_bidirectional_attention_lstm(lstm_dropout = 0.2, dropout = 0.1, 

                                       recurrent_dropout = 0.1, lstm_output = 100)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.pipeline import TransformerMixin

from sklearn.base import BaseEstimator



class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):

    """ Sklearn transformer to convert texts to indices list 

    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""

    def __init__(self,  **kwargs):

        super().__init__(**kwargs)

        

    def fit(self, texts, y=None):

        self.fit_on_texts(texts)

        return self

    

    def transform(self, texts, y=None):

        return np.array(self.texts_to_sequences(texts))

        

sequencer = TextsToSequences(num_words=TOP_WORDS)
class Padder(BaseEstimator, TransformerMixin):

    """ Pad and crop uneven lists to the same length. 

    Only the end of lists longernthan the maxlen attribute are

    kept, and lists shorter than maxlen are left-padded with zeros

    

    Attributes

    ----------

    maxlen: int

        sizes of sequences after padding

    max_index: int

        maximum index known by the Padder, if a higher index is met during 

        transform it is transformed to a 0

    """

    def __init__(self, maxlen=500):

        self.maxlen = maxlen

        self.max_index = None

        

    def fit(self, X, y=None):

        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()

        return self

    

    def transform(self, X, y=None):

        X = pad_sequences(X, maxlen=self.maxlen)

        X[X > self.max_index] = 0

        return X



padder = Padder(MAX_REVIEW_LENGTH)
def format_accuracy(dictionary_result):

    total = 0

    for I in dictionary_result:

        total += dictionary_result[I]

        print(I, dictionary_result[I])

    print("Macro All", total / len(dictionary_result))   
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import make_pipeline
def evaluate_model(preprocessing_function, size):

    scores = {}

    for R in tqdm(range(5)):

        train, test = generate_fold_data(R, preprocessing_function, size)

        train_X = [I[0] for I in train]

        train_y = [I[1] for I in train]

        

        attention_model = KerasClassifier(build_fn=bidirectional_attention_lstm, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1)

        pipeline = make_pipeline(sequencer, padder, attention_model)

        pipeline.fit(train_X, train_y)

        

        

        for J in test:

            test_X = [T[0] for T in test[J]]

            test_y = [T[1] for T in test[J]]



                   

            prediction = pipeline.predict(test_X)

            accuracy = f1_score(test_y, [I[0] > 0.5 for I in prediction])

            if J not in scores:

                scores[J] = 0

            scores[J] += accuracy / 5

    return pipeline, scores
pipeline, scores = evaluate_model(preprocess_stop_short, 1)
format_accuracy(scores)
from lime import lime_text

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=[0, 1])
ukara.query("kelompok == 'A' and label == 1").head(25)['teks'].values
string = ("Tantangan pengungsi iklim adalah lahan pekerjaan yang ia tempati dulu akan hilang, karena ia harus meninggalkan "

          + "negaranya karena bencana lingkungan yang ada di negaranya.")

fixed = preprocess_stop_short('A', string)

exp = explainer.explain_instance(fixed, pipeline.predict_proba)
exp.show_in_notebook(text=True)
ukara.query("kelompok == 'A' and label == 0").head(25)['teks'].values
string = ("Pengungsi iklim adalah orang-orang yang terpaksa meninggalkan komunitas atau negaranya karena bencana lingkungan.")

fixed = preprocess_stop_short('A', string)

exp = explainer.explain_instance(fixed, pipeline.predict_proba)
exp.show_in_notebook(text=True)
pipeline, scores = evaluate_model(preprocess_stop_short, 0.8)
format_accuracy(scores)
pipeline, scores = evaluate_model(preprocess_stop_short, 0.6)
format_accuracy(scores)
pipeline, scores = evaluate_model(preprocess_stop_short, 0.4)
format_accuracy(scores)
pipeline, scores = evaluate_model(preprocess_stop_short, 0.2)
format_accuracy(scores)