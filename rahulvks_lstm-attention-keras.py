import os

import re



import numpy as np

import pandas as pd

import tensorflow as tf

import keras.backend as K

from keras.preprocessing import text, sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints

from keras.layers import *

from keras.models import Model





# For reproducibility.

seed = 7

np.random.seed(seed)

tf.set_random_seed(seed)

session_conf = tf.ConfigProto(

    intra_op_parallelism_threads=1,

    inter_op_parallelism_threads=1

)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)





import os

print(os.listdir("../input"))

print(os.listdir("../input/glove-global-vectors-for-word-representation"))

print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))

print("Loading data...")

df_train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

print("Train shape:", df_train.shape)

df_test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv",encoding="latin-1")

print("Test shape:", df_test.shape)







df_train = df_train.rename(columns=({"comment_text":"Reviews"}))

df_train = df_train.rename(columns=({"target":"Label"}))

df_test = df_test.rename(columns=({"comment_text":"Reviews"}))
df_train = df_train[:10000]

df_test = df_test[:10000]
df_train.head(2)
df_train.columns
df_test.head(1)
EMBEDDING_FILE =  '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'



def load_embeddings(filename):

    embeddings = {}

    with open(filename) as f:

        for line in f:

            values = line.rstrip().split(' ')

            word = values[0]

            vector = np.asarray(values[1:], dtype='float32')

            embeddings[word] = vector

    return embeddings



embeddings = load_embeddings(EMBEDDING_FILE)
df_train.Reviews = df_train.Reviews.str.replace("n't", 'not')

df_test.Reviews = df_test.Reviews.str.replace("n't", 'not')
df_train['Label'].value_counts()
target_count=len(df_train['Label'].value_counts())

target_count
df_train.dtypes
df_test.dtypes
df_train['Reviews'] = df_train['Reviews'].astype(str)

df_test['Reviews'] = df_test['Reviews'].astype(str)
df_train.Reviews = df_train.Reviews.apply(lambda x: re.sub(r'[0-9]+', '0', x))

df_test.Reviews = df_test.Reviews.apply(lambda x: re.sub(r'[0-9]+', '0', x))



x_train = df_train['Reviews'].values

x_test  = df_test['Reviews'].values

y_train = df_train['Label'].values

x = np.r_[x_train, x_test]
tokenizer = Tokenizer(lower=True, filters='\n\t')

tokenizer.fit_on_texts(x)

x_train = tokenizer.texts_to_sequences(x_train)

x_test  = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1  # +1 is for zero padding.

print('vocabulary size: {}'.format(vocab_size))
maxlen = len(max((s for s in np.r_[x_train, x_test]), key=len))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')

x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

print('maxlen: {}'.format(maxlen))

print(x_train.shape)

print(x_test.shape)
def filter_embeddings(embeddings, word_index, vocab_size, dim=300):

    embedding_matrix = np.zeros([vocab_size, dim])

    for word, i in word_index.items():

        if i >= vocab_size:

            continue

        vector = embeddings.get(word)

        if vector is not None:

            embedding_matrix[i] = vector

    return embedding_matrix



embedding_size = 200

embedding_matrix = filter_embeddings(embeddings, tokenizer.word_index,

                                     vocab_size, embedding_size)

print('OOV: {}'.format(len(set(tokenizer.word_index) - set(embeddings))))
class Attention(Layer):

    """

    Keras Layer that implements an Attention mechanism for temporal data.

    Supports Masking.

    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

    # Input shape

        3D tensor with shape: `(samples, steps, features)`.

    # Output shape

        2D tensor with shape: `(samples, features)`.

    :param kwargs:

    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

    The dimensions are inferred based on the output shape of the RNN.

    Example:

        model.add(LSTM(64, return_sequences=True))

        model.add(Attention())

    """

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
def build_model(maxlen, vocab_size, embedding_size, embedding_matrix):

    input_words = Input((maxlen, ))

    x_words = Embedding(vocab_size,

                        embedding_size,

                        weights=[embedding_matrix],

                        mask_zero=True,

                        trainable=False)(input_words)

    x_words = SpatialDropout1D(0.3)(x_words)

    x_words = Bidirectional(LSTM(50, return_sequences=True))(x_words)

    x = Attention(maxlen)(x_words)

    x = Dropout(0.2)(x)

    x = Dense(50, activation='relu')(x)

    x = Dropout(0.2)(x)

    pred = Dense(target_count, activation='softmax')(x)



    model = Model(inputs=input_words, outputs=pred)

    return model



model = build_model(maxlen, vocab_size, embedding_size, embedding_matrix)

model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

save_file = 'model.h5'

history = model.fit(x_train, y_train,

                    epochs=15, verbose=1,

                    batch_size=1024, shuffle=True)
y_pred = model.predict(x_test, batch_size=1024)

y_pred = y_pred.argmax(axis=1).astype(int)

y_pred.shape
df_test['prediction'] = y_pred

df_test[['id', 'prediction']].to_csv('submission.csv', index=False)