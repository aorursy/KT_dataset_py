# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import plotly_express as px



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sodata = pd.read_csv("/kaggle/input/stack-overflow-questions/stack-overflow-data.csv")
sodata.head()
tags_counts = sodata.groupby('tags').aggregate({'post':np.count_nonzero}).reset_index().rename(columns = {'post':'tag_count'})

px.bar(tags_counts,x='tags',y='tag_count',color='tags')
# Creating train-test Split

from sklearn.model_selection import train_test_split

X = sodata[['post']]

y = sodata[['tags']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline



vec = CountVectorizer()

clf = LogisticRegression()

pipe = make_pipeline(vec, clf)

pipe.fit(X_train.post, y_train.tags)
from sklearn import metrics



def print_report(pipe):

    y_actuals = y_test['tags']

    y_preds = pipe.predict(X_test['post'])

    report = metrics.classification_report(y_actuals, y_preds)

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, y_preds)))



print_report(pipe)
clf.classes_
for i, tag in enumerate(clf.classes_):

    coefficients = clf.coef_[i]

    weights = list(zip(vec.get_feature_names(),coefficients))

    print('Tag:',tag)

    print('Most Positive Coefficients:')

    print(sorted(weights,key=lambda x: -x[1])[:10])

    print('Most Negative Coefficients:')

    print(sorted(weights,key=lambda x: x[1])[:10])

    print("--------------------------------------")
import eli5

eli5.show_weights(clf, vec=vec, top=50)
y_preds = pipe.predict(sodata['post'])

sodata['predicted_label'] = y_preds

misclassified_examples = sodata[(sodata['tags']!=sodata['predicted_label'])&(sodata['tags']=='python')&(sodata['predicted_label']=='java')]
misclassified_examples.head(50)
eli5.show_prediction(clf, misclassified_examples['post'].values[1], vec=vec)
# Some imports, we are not gong to use all the imports in this workbook but in subsequent workbooks we surely will.

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

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder





from keras.layers import *

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *

import tensorflow as tf

import os

import time

import gc

import re

import glob
# Define some Global Variables



max_features = 100000 # Maximum Number of words we want to include in our dictionary

maxlen = 72 # No of words in question we want to create a sequence with

embed_size = 300# Size of word to vec embedding we are using
# Some preprocesssing that will be common to all the text classification methods you will see. 





# Loading the data

def load_and_prec():



    ## split to train and val

    train_df, val_df = train_test_split(sodata, test_size=0.2, random_state=2019)



    ## fill up the missing values

    train_X = train_df["post"].values

    val_X = val_df["post"].values





    ## Tokenize the sentences

    '''

    keras.preprocessing.text.Tokenizer tokenizes(splits) the texts into tokens(words).

    Signature:

    Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 

    lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs)



    The num_words parameter keeps a prespecified number of words in the text only. 

    It also filters some non wanted tokens by default and converts the text into lowercase.



    It keeps an index of words(dictionary of words which we can use to assign a unique number to a word) 

    which can be accessed by tokenizer.word_index.

    For example - For a text corpus the tokenizer word index might look like. 

    The words in the indexed dictionary are sort of ranked in order of frequencies,

    {'the': 1,'what': 2,'is': 3, 'a': 4, 'to': 5, 'in': 6, 'of': 7, 'i': 8, 'how': 9}

    

    The texts_to_sequence function converts every word(token) to its respective index in the word_index

    

    So Lets say we started with 

    train_X as something like ['This is a sentence','This is another bigger sentence']

    and after fitting our tokenizer we get the word_index as {'this':1,'is':2,'sentence':3,'a':4,'another':5,'bigger':6}

    The texts_to_sequence function will tokenize the sentences and replace words with individual tokens to give us 

    train_X = [[1,2,4,3],[1,2,5,6,3]]

    '''

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    val_X = tokenizer.texts_to_sequences(val_X)





    ## Pad the sentences. We need to pad the sequence with 0's to achieve consistent length across examples.

    '''

    We had train_X = [[1,2,4,3],[1,2,5,6,3]]

    lets say maxlen=6

        We will then get 

        train_X = [[1,2,4,3,0,0],[1,2,5,6,3,0]]

    '''

    train_X = pad_sequences(train_X, maxlen=maxlen)

    val_X = pad_sequences(val_X, maxlen=maxlen)





    ## Get the target values

    train_y = train_df['tags'].values

    val_y = val_df['tags'].values

    

    encoder = LabelEncoder()

    encoder.fit(train_y)

    train_y = encoder.transform(train_y)

    val_y = encoder.transform(val_y)

    train_y = to_categorical(train_y)

    val_y = to_categorical(val_y)



    

    

    #shuffling the data

    np.random.seed(2018)

    trn_idx = np.random.permutation(len(train_X))



    train_X = train_X[trn_idx]

    train_y = train_y[trn_idx]

    

    return train_X, val_X,  train_y, val_y, val_df, tokenizer, encoder
train_X, val_X,  train_y, val_y, val_df,tokenizer , encoder = load_and_prec()
# Word 2 vec Embedding



def load_glove(word_index):

    '''We want to create an embedding matrix in which we keep only the word2vec for words which are in our word_index

    '''

    EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

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
embedding_matrix = load_glove(tokenizer.word_index)
# https://www.kaggle.com/yekenot/2dcnn-textclassifier

def model_cnn(embedding_matrix):

    filter_sizes = [1,2,3,5]

    num_filters = 36



    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = Reshape((maxlen, embed_size, 1))(x)



    maxpool_pool = []

    for i in range(len(filter_sizes)):

        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),

                                     kernel_initializer='he_normal', activation='elu')(x)

        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))



    z = Concatenate(axis=1)(maxpool_pool)   

    z = Flatten()(z)

    z = Dropout(0.1)(z)



    outp = Dense(20, activation="softmax")(z)



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
model = model_cnn(embedding_matrix)

model.summary()
def train(model, epochs=5):

    filepath="weights_best.h5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')

    callbacks = [checkpoint, reduce_lr]

    for e in range(epochs):

        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y),callbacks=callbacks)

    model.load_weights(filepath)



train(model, epochs = 10)
from sklearn import metrics



def print_report_nn(model):

    a=np.argmax(val_y, axis = 1)

    y_actuals = encoder.inverse_transform(a)

    y_preds = model.predict([val_X], batch_size=1024, verbose=0)

    prediction_ = np.argmax(y_preds, axis = 1)

    prediction_ = encoder.inverse_transform(prediction_)

    report = metrics.classification_report(y_actuals, prediction_)

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_actuals, prediction_)))



print_report_nn(model)
def predict_complex(docs):

    # preprocess the docs

    val_X = tokenizer.texts_to_sequences(docs)

    val_X = pad_sequences(val_X, maxlen=maxlen)

    y_preds = model.predict([val_X], batch_size=1024, verbose=0)

    return y_preds

                            
import eli5

from eli5.lime import TextExplainer



te = TextExplainer(random_state=2019)

te.fit(misclassified_examples['post'].values[1], predict_complex)

te.show_prediction(target_names=list(encoder.classes_))