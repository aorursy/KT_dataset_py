!pip install glove_python
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import WordPunctTokenizer

from collections import Counter

from string import punctuation, ascii_lowercase

import regex as re

from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

from keras import Sequential

from keras.layers import Conv1D, Flatten, MaxPooling1D, Dense, Dropout, Input, LSTM

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, Conv1D, Embedding, Dropout, Flatten, MaxPooling1D

from keras.models import Model

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from sklearn.metrics import mean_squared_log_error

from keras import backend as K

from keras.engine.topology import Layer

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nnfl-assignment-2/final_train.csv")

test_df = pd.read_csv("/kaggle/input/nnfl-assignment-2/final_test.csv")
train_df.head()
def textClean(text):

    text = text.replace("."," ").replace(","," ").replace(";"," ")

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 

             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 

             'them', 'when', 'will', 'that', 'its', 'because', 

             'while', 'those', 'my', 'don', 'again', 'her', 'if',

             'further', 'now', 'does', 'against', 'won', 'same', 

             'a', 'during', 'who', 'here', 'have', 'in', 'being', 

             'it', 'other', 'once', 'itself', 'hers', 'after', 're',

             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 

             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',

             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',

             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',

             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',

             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 

             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',

             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 

             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',

             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 

             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 

             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',

             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 

             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',

             'o', 'before'}

    text = [w for w in text if not w in stops]    

    text = " ".join(text)

    

    return(text)
train_df["clean_text"] = [textClean(t) for t in train_df.desc]

test_df["clean_text"] = [textClean(t) for t in test_df.desc]
num_words = 50000

tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

                                   lower=True,split=' ')

tokenizer.fit_on_texts(train_df['clean_text'].values)

X = tokenizer.texts_to_sequences(train_df['clean_text'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



max_length_of_text = 200

X = pad_sequences(X, maxlen=max_length_of_text)



print(word_index)
y = train_df['rating']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
embed_dim = 100 #Change to observe effects



inputs = Input((max_length_of_text, ))

x = Embedding(num_words, embed_dim)(inputs)    

x = Conv1D(32, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = Conv1D(32, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = MaxPooling1D(2)(x)

x = Dropout(0.5)(x)

x = Conv1D(64, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = Conv1D(64, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = MaxPooling1D(2)(x)

x = Dropout(0.5)(x)

x = Conv1D(64, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = Conv1D(64, 3, activation = "relu", kernel_initializer="he_uniform")(x)

x = MaxPooling1D(2)(x)

x = Dropout(0.5)(x)

x = Flatten()(x)

x = Dense(128, activation = "relu", kernel_initializer="he_uniform")(x)

x = Dense(1)(x)

model = Model(inputs, x)

print(model.summary())
model.compile(loss = 'mean_squared_logarithmic_error', optimizer='adam',metrics = ['accuracy'])

hist = model.fit(X, y, validation_split=0.1 ,batch_size = 1032, epochs = 10)
history = pd.DataFrame(hist.history)

plt.figure(figsize=(12,12))

plt.plot(history["loss"])

plt.plot(history["val_loss"])

plt.title("Loss with pretrained word vectors")

plt.show()
predict = model.predict(X_test)

predict = [float(np.round(i)) if i<10.0 else 10.0 for i in predict]
np.sqrt(mean_squared_log_error(y_test,predict))
x_test = tokenizer.texts_to_sequences(test_df.clean_text.values)

x_test = pad_sequences(x_test, maxlen=max_length_of_text)
pred = model.predict(x_test)
pred = [float(np.round(i)) if i<10.0 else 10.0 for i in pred]

print(pred)
final_df = pd.DataFrame(list(zip(test_df["Id"],pred)), columns = ["Id","Rating"])

final_df.head()
final_df.to_csv("submission_cnn.csv",index=False)
model.save_weights("model_cnn.h5")
from IPython.display import FileLink
FileLink("submission_cnn.csv")
FileLink("model_cnn.h5")
train = pd.read_csv("/kaggle/input/nnfl-assignment-2/final_train.csv")

test = pd.read_csv("/kaggle/input/nnfl-assignment-2/final_test.csv")

# train.head()

train.clean_text = [textClean(i) for i in train.desc]

test.clean_text = [textClean(i) for i in test.desc]

X_train = train.clean_text

y_train = train.rating.values 

X_test = test.clean_text



# sample_pred = np.zeros_like(test["rating"], dtype=np.float32)
lines = list(X_train) + list(X_test)

new_lines = []

for i in lines:

    new_lines.append(i.split(' '))
X_train[0]
tokenizer = Tokenizer(num_words=50000)

tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

x_train = pad_sequences(X_train, maxlen=200)

x_test = pad_sequences(X_test, maxlen=200)
from glove import Corpus, Glove
corpus = Corpus()

# print(x_train))

corpus.fit(new_lines, window=10)



glove = Glove(no_components=200, learning_rate=0.05) 

glove.fit(corpus.matrix, epochs=10, no_threads=10, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove.model')
embeddings_index = {}

for i in glove.dictionary.keys():

    embeddings_index[i] = glove.word_vectors[glove.dictionary[i]]
word_index = tokenizer.word_index

nb_words = min(50000, len(word_index))

embedding_matrix = np.zeros((nb_words, 200))

for word, i in word_index.items():

    if i >= 50000: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
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
inp = Input(shape=(200, ))

x = Embedding(50000, 200, weights=[embedding_matrix],trainable = False)(inp)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x = Bidirectional(CuDNNLSTM(128, return_sequences=False))(x)

# x = Attention(100)(x)

x = Dense(64, activation="relu")(x)

x = Dense(1, activation="relu")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss="mean_squared_logarithmic_error",optimizer="adam",metrics=['accuracy'])

print(model.summary())
hist = model.fit(x_train, y_train, validation_split=0.1 ,batch_size = 2032, epochs = 10)
history = pd.DataFrame(hist.history)

plt.figure(figsize=(12,12))

plt.plot(history["loss"])

plt.plot(history["val_loss"])

plt.title("Loss with pretrained word vectors")

plt.show()
pred = model.predict(x_test)
# new_pred = pred*10

# print(new_pred[:10])

new_pred = [max(1.0,min(float(np.round(i)),10.0)) for i in pred]

print(new_pred[:10])
final_df = pd.DataFrame(list(zip(test_df["Id"],new_pred)), columns = ["Id","Rating"])

final_df.head()
final_df.to_csv("submission_bilstm1.csv",index=False)
from IPython.display import FileLink
FileLink("submission_bilstm1.csv")