import gc
import numpy as np
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from keras import initializers, regularizers, constraints
from keras.layers import Dense, Input, Flatten, RepeatVector, Permute
from keras.layers import Input, Dense, LSTM,merge, Merge, Bidirectional, concatenate, SpatialDropout1D, GRU, BatchNormalization, Dropout, Activation,TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.layers import Convolution1D, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,Lambda
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import *
from keras import backend as K

import tensorflow as tf
train = pd.read_csv('../input/donorschoose-application-screening/train.csv')
test = pd.read_csv('../input/donorschoose-application-screening/test.csv')
resources = pd.read_csv('../input/donorschoose-application-screening/resources.csv')
train = train.sort_values(by="project_submitted_datetime")
char_cols = ['project_title', 'project_essay_1', 'project_essay_2',
             'project_essay_3', 'project_essay_4', 'project_resource_summary']

train = train.fillna('NA')
test = test.fillna('NA')

teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)

train['text'] = ''
for col in char_cols:
    train['text'] += train[col]
    train['text'] += ' '

test['text'] = ''
for col in char_cols:
    test['text'] += test[col]
    test['text'] += ' '

def preprocess(string):
    '''
    :param string:
    :return:
    '''
    string = re.sub(r'(\\r)', ' ', string)
    string = re.sub(r'(\\n)', ' ', string)
    string = re.sub(r'(\\r\\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)
    string = re.sub(r'\d+', ' ', string) 
    string = re.sub(r'(\")', ' ', string)
    string = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n]', ' ', string)

    return string

train["text"]=train["text"].apply(preprocess)
test["text"]=test["text"].apply(preprocess)

y = train.project_is_approved.values

train_text = train.text.values
test_text = test.text.values

tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(train_text) + list(test_text))

len(tokenizer.word_index)
word_index = tokenizer.word_index

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 600

sequences = tokenizer.texts_to_sequences(train_text)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

sequence_test = tokenizer.texts_to_sequences(test_text)
test_data = pad_sequences(sequence_test, maxlen=MAX_SEQUENCE_LENGTH)
X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.1, random_state=666)
embedding_matrix = np.load('../input/crawlembeddingmatric/embedding_matrix.npy')

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P
def get_model():
    class AttLayer(Layer):
        def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                     bias_regularizer=None, kernel_constraint=None, 
                     bias_constraint=None,  **kwargs):
            self.supports_masking = True
            self.init = initializers.get(init)
            self.kernel_initializer = initializers.get(init)

            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(kernel_regularizer)

            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)

            super(AttLayer, self).__init__(** kwargs)

        def build(self, input_shape):
            assert len(input_shape)==3
            self.W = self.add_weight((input_shape[-1], 1),
                                     initializer=self.kernel_initializer,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
            self.u = self.add_weight((input_shape[1],),
                                     initializer=self.kernel_initializer,
                                     name='{}_u'.format(self.name),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

            self.built = True

        def compute_mask(self, input, input_mask=None):
            return None

        def call(self, x, mask=None):
            uit = K.dot(x, self.W) # (x, 40, 1)
            uit = K.squeeze(uit, -1) # (x, 40)
            uit = uit + self.b # (x, 40) + (40,)
            uit = K.tanh(uit) # (x, 40)

            ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
            ait = K.exp(ait) # (X, 1)

            if mask is not None:
                mask = K.cast(mask, K.floatx()) #(x, 40)
                ait = mask*ait #(x, 40) * (x, 40, )

            ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            ait = K.expand_dims(ait)
            weighted_input = x * ait
            output = K.sum(weighted_input, axis=1)
            return output

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    embed = embedding_layer(inputs)
    gru = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))(embed)
    attention = AttLayer()(gru)
    output = Dense(1, activation='sigmoid')(attention)
    model = Model(inputs, output)
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[auc])
    
    return model
model = get_model()
model.load_weights("../input/donorweight/atten0.7822.h5")
checkpoint = ModelCheckpoint('atten.h5', monitor="val_auc", verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.000001)
# model.fit(X_train, y_train, validation_data=(X_val, y_val),
#           epochs=1, batch_size=128, callbacks=[checkpoint, reduce_lr])
p_sub= model.predict([test_data]).T[0]

sub = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')

sub.project_is_approved = p_sub

sub.to_csv('attention.csv', index=False)