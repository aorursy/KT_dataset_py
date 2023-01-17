import os

import csv

import json

import string

import keras

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from math import floor

import spacy



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import MultinomialNB 

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score, log_loss



import lightgbm as lgb



import time

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, Concatenate, Add, Flatten, CuDNNLSTM

from keras.models import Model

from keras import backend as K

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.engine.topology import Layer

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict





# keras libraries

from keras.models import Model, load_model,Sequential

from keras.layers import Dense, Input, Dropout,Bidirectional, GRU, Activation, concatenate, Embedding, SpatialDropout1D

from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D ,GlobalMaxPool1D, GlobalAvgPool1D, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence, text

from keras import layers



train = pd.read_csv('../input/gapdevelopment/repository/google-research-datasets-gap-coreference-83135f2/gap-development.tsv',delimiter='\t',encoding='utf-8')

test = pd.read_csv('../input/gapdevelopment/repository/google-research-datasets-gap-coreference-83135f2/gap-test.tsv',delimiter='\t',encoding='utf-8');

validation = pd.read_csv('../input/gapdevelopment/repository/google-research-datasets-gap-coreference-83135f2/gap-validation.tsv',delimiter='\t',encoding='utf-8');



print(train.shape)

print(test.shape)

print(validation.shape)
train.head()
test.head()
validation.head()
true_B = train.loc[train['B-coref']== True ]

true_B.drop('A',axis=1,inplace=True)

true_B.drop('A-offset',axis=1,inplace=True)

true_B.drop('A-coref',axis=1,inplace=True)

true_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

true_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



true_A = train.loc[train['A-coref']== True ]

true_A.drop('B',axis=1,inplace=True)

true_A.drop('B-offset',axis=1,inplace=True)

true_A.drop('B-coref',axis=1,inplace=True)

true_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_A = train.loc[train['A-coref']== False ]

false_A.drop('B',axis=1,inplace=True)

false_A.drop('B-offset',axis=1,inplace=True)

false_A.drop('B-coref',axis=1,inplace=True)

false_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_B = train.loc[train['B-coref']== False ]

false_B.drop('A',axis=1,inplace=True)

false_B.drop('A-offset',axis=1,inplace=True)

false_B.drop('A-coref',axis=1,inplace=True)

false_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

false_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)
frames = [true_A,false_A,true_B,false_B]

new_train=pd.concat(frames)

new_train.loc[new_train['A-offset']== float('nan')]

new_train.dropna(how='all')
true_test_B = test.loc[test['B-coref']== True ]

true_test_B.drop('A',axis=1,inplace=True)

true_test_B.drop('A-offset',axis=1,inplace=True)

true_test_B.drop('A-coref',axis=1,inplace=True)

true_test_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

true_test_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



true_test_A = test.loc[test['A-coref']== True ]

true_test_A.drop('B',axis=1,inplace=True)

true_test_A.drop('B-offset',axis=1,inplace=True)

true_test_A.drop('B-coref',axis=1,inplace=True)

true_test_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_test_A = test.loc[test['A-coref']== False ]

false_test_A.drop('B',axis=1,inplace=True)

false_test_A.drop('B-offset',axis=1,inplace=True)

false_test_A.drop('B-coref',axis=1,inplace=True)

false_test_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_test_B = test.loc[test['B-coref']== False ]

false_test_B.drop('A',axis=1,inplace=True)

false_test_B.drop('A-offset',axis=1,inplace=True)

false_test_B.drop('A-coref',axis=1,inplace=True)

false_test_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

false_test_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)
frames = [true_test_A,false_test_A,true_test_B,false_test_B]

new_test=pd.concat(frames)

new_test.loc[new_test['A-offset']== float('nan')]

new_test.dropna(how='all')
true_validation_B = validation.loc[validation['B-coref']== True ]

true_validation_B.drop('A',axis=1,inplace=True)

true_validation_B.drop('A-offset',axis=1,inplace=True)

true_validation_B.drop('A-coref',axis=1,inplace=True)

true_validation_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

true_validation_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



true_validation_A = validation.loc[validation['A-coref']== True ]

true_validation_A.drop('B',axis=1,inplace=True)

true_validation_A.drop('B-offset',axis=1,inplace=True)

true_validation_A.drop('B-coref',axis=1,inplace=True)

true_validation_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_validation_A = validation.loc[validation['A-coref']== False ]

false_validation_A.drop('B',axis=1,inplace=True)

false_validation_A.drop('B-offset',axis=1,inplace=True)

false_validation_A.drop('B-coref',axis=1,inplace=True)

false_validation_A.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)



false_validation_B = validation.loc[validation['B-coref']== False ]

false_validation_B.drop('A',axis=1,inplace=True)

false_validation_B.drop('A-offset',axis=1,inplace=True)

false_validation_B.drop('A-coref',axis=1,inplace=True)

false_validation_B.rename(columns={'B-offset':'A-offset','B':'A','B-coref':'A-coref'},inplace=True)

false_validation_B.sort_values(by=['ID','Text','Pronoun','Pronoun-offset','A','URL'], axis=0,inplace=True)
frames = [true_validation_A,false_validation_A,true_validation_B,false_validation_B]

new_validation=pd.concat(frames)

new_validation.loc[new_validation['A-offset']== float('nan')]

new_validation.dropna(how='all')
def prepare_data(data, label = None, test=False):    



    

    text = []

    A = []

    Pronoun = []

    distances = []

    for row in data[['Text','A','Pronoun','A-offset','Pronoun-offset']].values:

        text.append(row[0])

        A.append(row[1])

        Pronoun.append(row[2])

        distances.append(row[4]-row[3])

        

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(text)

    texts = tokenizer.texts_to_sequences(text)  

    texts = pad_sequences(texts, maxlen=maxlen)

    word_index = tokenizer.word_index

    prev_text = []

    

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    A_terms = tokenizer.texts_to_sequences(A)  

    A_terms = pad_sequences(A_terms, maxlen=100)

     

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    

    pronouns = tokenizer.texts_to_sequences(Pronoun)  

    pronouns = pad_sequences(A_terms, maxlen=100)

    

    X = []

    

    i = 0 

    data_examples = len(data)

    for i in range(0,data_examples):

        aux = np.append(texts[i],A_terms[i])

        aux = np.append(aux,pronouns[i])

        X.append(np.append(aux,distances[i]))



    X = np.asarray(X)

    print(X.shape)

    #Y = pd.get_dummies(data[label]).values

    

    Y = pd.get_dummies(data["A-coref"].values)

    if test == True:

        return X, word_index, tokenizer

    else:

        return X, Y, word_index, tokenizer
def get_previous_text(data,test=False):

    

    text_before_A = []

    text_before_Pronoun = []

    for row in data[['Text','A-offset','Pronoun-offset']].values:

        text_before_A.append(row[0][:row[1]-1])

        text_before_Pronoun.append(row[0][:row[2]-1])

        

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(text_before_A)

    text_A = tokenizer.texts_to_sequences(text_before_A)  

    texts_A = pad_sequences(text_A, maxlen=maxlen)

    word_index = tokenizer.word_index

    

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    tokenizer.fit_on_texts(text_before_Pronoun)

    text_pronoun = tokenizer.texts_to_sequences(text_before_Pronoun)  

    text_pronoun = pad_sequences(text_pronoun, maxlen=maxlen)

    word_index = tokenizer.word_index

    

    X = []

    

    i = 0 

    data_examples = len(data)

    for i in range(0,data_examples):

        aux = np.append(text_pronoun[i],texts_A[i])

        X.append(aux)



    X = np.asarray(X)

    #Y = pd.get_dummies(data[label]).values

    

    Y = pd.get_dummies(data["A-coref"].values)

    if test == True:

        return X, word_index, tokenizer

    else:

        return X, Y, word_index, tokenizer
# Parâmetros de Vetorização

maxlen = 220

embed_size = 500

max_features = 7000



# Aplicação nos dados de treinamento



tokenizer = Tokenizer(num_words=max_features)

tokenizer_list = list(train.Text.values)

tokenizer.fit_on_texts(tokenizer_list)



train_X = tokenizer.texts_to_sequences(train.Text.values)

train_auX = tokenizer.texts_to_sequences(train.Text.values)



train_X = pad_sequences(train_X, maxlen=maxlen)

train_auX = pad_sequences(train_auX, maxlen=maxlen)



y_train = pd.get_dummies(train["A-coref"].values)

word_index = tokenizer.word_index

max_features = len(word_index)



# Aplicação nos dados de validação



tokenizer = Tokenizer(num_words=max_features)

tokenizer_list = list(validation.Text.values)

tokenizer.fit_on_texts(tokenizer_list)



validation_X = tokenizer.texts_to_sequences(validation.Text.values)

validation_auX = tokenizer.texts_to_sequences(validation.Text.values)



validation_X = pad_sequences(validation_X, maxlen=maxlen)

validation_auX = pad_sequences(validation_auX, maxlen=maxlen)



y_validation = pd.get_dummies(validation["A-coref"].values)

word_index = tokenizer.word_index





# Aplicação nos dados de teste



tokenizer = Tokenizer(num_words=max_features)

tokenizer_list = list(test.Text.values)

tokenizer.fit_on_texts(tokenizer_list)



test_X = tokenizer.texts_to_sequences(test.Text.values)

test_auX = tokenizer.texts_to_sequences(test.Text.values)



test_X = pad_sequences(test_X, maxlen=maxlen)

test_auX = pad_sequences(test_auX, maxlen=maxlen)



y_test = pd.get_dummies(test["A-coref"].values)

word_index = tokenizer.word_index





print(train_X.shape)

print(y_train.shape)

print(test_X.shape)

print(y_test.shape)



#y[1] --> True = 1 e False = 0 
def get_model_SVM(X_train, X_teste, Y_train):

    model = SVC()

    model.fit(X_train,Y_train)

    return model.predict(X_teste)



def get_model_Bayes(X_train, X_teste, Y_train):

    model = MultinomialNB()

    model.fit(X_train,Y_train)

    return model.predict(X_teste)



def get_model_forest(X_train,X_teste, Y_train):

    model = RandomForestClassifier()

    model.fit(X_train,Y_train)

    return model.predict(X_teste)



def get_model_Regression(X_train,X_teste, Y_train):

    model = LogisticRegression()

    model.fit(X_train, Y_train)

    return model.predict(X_teste)
print("Loss SVM:", log_loss(y_test[1], get_model_SVM(train_X,test_X,y_train[1])))

print("Loss Bayes:", log_loss(y_test[1],get_model_Bayes(train_X,test_X,y_train[1])))

print("Loss Random Forest:", log_loss(y_test[1],get_model_forest(train_X,test_X,y_train[1])))

print("Loss Regressão Linear:", log_loss(y_test[1],get_model_Regression(train_X,test_X,y_train[1])))
def dot_product(x, kernel):



    if K.backend() == 'tensorflow':

        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

    else:

        return K.dot(x, kernel)



      

class AttentionWithContext(Layer):



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

def get_model_LSTM():

  inp1 = Input(shape=(maxlen,))

  inp2 = Input(shape=(maxlen,))



  model1_out = Embedding(max_features, embed_size)(inp1)

  model1_out = Bidirectional(LSTM(256, return_sequences=True))(model1_out)

  model1_out = AttentionWithContext()(model1_out)

  model1_out = Dropout(0.1)(model1_out)



  model2_out = Embedding(max_features, embed_size)(inp2)

  model2_out = Bidirectional(LSTM(256, return_sequences=True))(model2_out)

  model2_out = AttentionWithContext()(model2_out)

  model2_out = Dropout(0.1)(model2_out)



  merged_out = keras.layers.Concatenate(axis=1)([model1_out, model2_out])



  merged_out = Dense(32, activation="relu")(merged_out)

  merged_out = Dropout(0.1)(merged_out)

  

  merged_out = Dense(2, activation="sigmoid")(merged_out)

  model = Model(inputs=[inp1,inp2], outputs=merged_out)

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  

  return model
model = get_model_LSTM()

print(model.summary())
model.fit([train_X, train_auX], y_train, batch_size=512, epochs=30, validation_data=([validation_X,validation_auX],y_validation))

#Perdas do modelo

model.evaluate([test_X, test_auX],y_test,verbose=1,batch_size=100)