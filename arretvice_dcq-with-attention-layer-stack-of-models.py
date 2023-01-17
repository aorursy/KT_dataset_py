import re
import os
import sys
import math
import string
import random
import datetime

import numpy as np
import pandas as pd

from time import time
from tqdm import tqdm, trange

from sklearn import metrics

import keras.layers
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, CuDNNGRU, Dense, Bidirectional, Dropout, BatchNormalization, Activation, Input
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
print(f'Starting at {datetime.datetime.now()}')
# training data
df=pd.read_csv('../input/train.csv')
df=df[['question_text','target']]
df=df.dropna()
max_word_len=60
# embeddings
embedding_path='glove.840B.300d/glove.840B.300d.txt'
vocab={}
with open('../input/embeddings/' + embedding_path,'r') as f:
    for line_number,line in enumerate(tqdm(f)):
        key,values=line.split(' ')[0],line.split(' ')[1:]
        if not any(char in string.punctuation for char in key): 
            vocab[key]=np.asarray(values,dtype='float32')
print(f'Total of {len(vocab.keys())} words in vocabulary')
trans_table={key:' ' for key in string.punctuation}
unknown_word=np.zeros(300)
def str_prep(s):
    s=s.translate(str.maketrans(trans_table)).lower().strip() # remove punctuation
    s=re.sub(' +',' ',s) # get rid of multiple spaces inside
    s=s.split(' ')
    ar=np.asarray([vocab.get(x,unknown_word) for x in s],dtype='float32')
    return ar
def batch_gen(df,min_batch_size=1024,transformation_func=str_prep, training_mode=True):
    n_batches=math.ceil(len(df)/min_batch_size)
    while True:
        df=df.sample(frac=1).reset_index(drop=True)
        for batch in range(n_batches):
            start=batch*min_batch_size
            end=start+min_batch_size
            if batch==n_batches-1:
                X=df['question_text'][start:].apply(transformation_func)
                X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
                y=np.array(df['target'][start:])
            else:
                X=df['question_text'][start:end].apply(transformation_func)
                X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
                y=np.array(df['target'][start:end])
            yield X, y
def train_val_split(df,frac=0.2):
    df=df.sample(frac=1).reset_index(drop=True) # random shuffling
    df_sincere=df[df['target']==0]
    df_taxic=df[df['target']==1]
    sincere_border=int(len(df_sincere)*(1-frac))
    taxic_border=int(len(df_taxic)*(1-frac))
    df_train=pd.concat([df_sincere[:sincere_border],df_taxic[:taxic_border]])
    df_val=pd.concat([df_sincere[sincere_border:],df_taxic[taxic_border:]])
    print('Training data peppered (prepared)!')
    return df_train.sample(frac=1).reset_index(drop=True), df_val.sample(frac=1).reset_index(drop=True)
# attention layer
# idea stolen from https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, bias=True, **kwargs):
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
        self.W = self.add_weight((input_shape[-1],), initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],), initializer='zero',
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
# # experimental model with attention layer
# def train_model(train_df,val_df,n_epochs=5,batch_size=1024):
    
#     #########################################
    
#     inp = Input(shape=(max_word_len, 300))
#     lstm1=Bidirectional(CuDNNLSTM(32, return_sequences=True))(inp)
#     lstm2=Bidirectional(CuDNNLSTM(32, return_sequences=True))(lstm1)
#     attention1=Attention(max_word_len)(lstm2)
#     dense1=Dense(64, activation='elu')(attention1)
#     dense2=Dense(64, activation='elu')(dense1)
#     outp = Dense(1, activation="sigmoid")(dense2)
    
#     #########################################
#     opt=Adam(lr=0.00321)
#     model = Model(inputs=inp, outputs=outp)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     model.fit_generator(batch_gen(train_df, min_batch_size=batch_size), epochs=n_epochs,
#                         steps_per_epoch=math.ceil(len(train_df)/batch_size),
#                         validation_data=batch_gen(val_df, min_batch_size=batch_size),
#                         validation_steps=math.ceil(len(val_df)/batch_size),
#                         verbose=True)
#     return model
# # experimental model - GRU vs LSTM vs BOTH
# def train_model(train_df,val_df,n_epochs=5,batch_size=1024):
    
#     '''
#     Quick reminder for me
#     Your output shape depends on how you configure the net. 
#     If your LSTM/RNN has return_sequences=False, you'll have one
#     label per sequence; if you set return_sequences=True, 
#     you'll have one label per timestep.
#     '''
    
#     input_layer = Input(shape=(max_word_len, 300))
#     #########################################
#     # left branch
#     lstm_left=Bidirectional(CuDNNLSTM(32, return_sequences=True))(input_layer)
#     gru_left=Bidirectional(CuDNNGRU(32, return_sequences=True))(lstm_left)
#     attention_left=Attention(max_word_len)(gru_left)
#     dense_left=Dense(256, activation='elu')(attention_left)
    
#     # right branch
#     gru_right=Bidirectional(CuDNNGRU(32, return_sequences=True))(input_layer)
#     lstm_right=Bidirectional(CuDNNLSTM(32, return_sequences=True))(gru_right)
#     attention_right=Attention(max_word_len)(lstm_right)
#     dense_right=Dense(256, activation='elu')(attention_right)   
    
#     # only lstm
#     only_lstm1=Bidirectional(CuDNNLSTM(32, return_sequences=True))(input_layer)
#     only_lstm2=Bidirectional(CuDNNLSTM(32, return_sequences=True))(only_lstm1)
#     attention_only_lstm=Attention(max_word_len)(only_lstm2)
#     dense_only_lstm=Dense(256, activation='elu')(attention_only_lstm)
    
#     # only gru
#     only_gru1=Bidirectional(CuDNNGRU(32, return_sequences=True))(input_layer)
#     only_gru2=Bidirectional(CuDNNGRU(32, return_sequences=True))(only_gru1)
#     attention_only_gru=Attention(max_word_len)(only_gru2)
#     dense_only_gru=Dense(256, activation='elu')(attention_only_gru) 
    
#     # concatenation
#     concatenate_layer=keras.layers.concatenate([dense_left,dense_right,
#                                                dense_only_lstm,dense_only_gru],
#                                                axis=1)
#     out=Dense(128, activation='elu')(concatenate_layer)
#     output_layer = Dense(1, activation='sigmoid', name='Output')(out)
    
#     #########################################
#     opt=Adam(lr=0.003)
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     model.fit_generator(batch_gen(train_df, min_batch_size=batch_size), epochs=n_epochs,
#                         steps_per_epoch=math.ceil(len(train_df)/batch_size),
#                         validation_data=batch_gen(val_df, min_batch_size=batch_size),
#                         validation_steps=math.ceil(len(val_df)/batch_size),
#                         verbose=True)
#     return model
# experimental model - GRU and LSTM
def train_model(train_df,val_df,n_epochs=5,batch_size=1024):
    
    '''
    Quick reminder for me
    Your output shape depends on how you configure the net. 
    If your LSTM/RNN has return_sequences=False, you'll have one
    label per sequence; if you set return_sequences=True, 
    you'll have one label per timestep.
    '''
    
    input_layer = Input(shape=(max_word_len, 300))
    #########################################
    # only lstm
    only_lstm1=Bidirectional(CuDNNLSTM(32, return_sequences=True))(input_layer)
    only_lstm2=Bidirectional(CuDNNLSTM(32, return_sequences=True))(only_lstm1)
    attention_only_lstm=Attention(max_word_len)(only_lstm2)
    dense_only_lstm=Dense(64, activation='elu')(attention_only_lstm)
    
    # only gru
    only_gru1=Bidirectional(CuDNNGRU(32, return_sequences=True))(input_layer)
    only_gru2=Bidirectional(CuDNNGRU(32, return_sequences=True))(only_gru1)
    attention_only_gru=Attention(max_word_len)(only_gru2)
    dense_only_gru=Dense(64, activation='elu')(attention_only_gru) 
    
    # concatenation
    concatenate_layer=keras.layers.concatenate([dense_only_lstm,dense_only_gru],
                                               axis=1)
    out=Dense(64, activation='elu')(concatenate_layer)
    output_layer = Dense(1, activation='sigmoid', name='Output')(out)
    
    #########################################
    opt=Adam(lr=0.003)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.fit_generator(batch_gen(train_df, min_batch_size=batch_size), epochs=n_epochs,
                        steps_per_epoch=math.ceil(len(train_df)/batch_size),
                        validation_data=batch_gen(val_df, min_batch_size=batch_size),
                        validation_steps=math.ceil(len(val_df)/batch_size),
                        verbose=True)
    return model
def get_predictions(df, model, min_batch_size=1024,transformation_func=str_prep):
    predictions=np.ndarray(shape=(0,1))
    print(f'Total values to predict: {len(df)}')
    n_batches=math.ceil(len(df)/min_batch_size)
    for batch in trange(n_batches):
        start=batch*min_batch_size
        end=start+min_batch_size
        if batch==n_batches-1:
            X=df['question_text'][start:].apply(transformation_func)
            X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
            y_predicted=model.predict(X)
        else:
            X=df['question_text'][start:end].apply(transformation_func)
            X=pad_sequences(X,maxlen=60,dtype='float32',padding='post')
            y_predicted=model.predict(X)
        predictions=np.append(predictions,y_predicted,axis=0)
    return predictions
def select_best_threshold(y_true, y_predicted):
    thresholds = []
    for thresh in np.arange(0.0, 1, 0.001):
        thresh = np.round(thresh, 3)
        res=metrics.f1_score(y_true, (y_predicted>thresh).astype(int))
        thresholds.append([thresh, res])
    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    print("Best threshold: ", best_thresh)
    return best_thresh
stack_size=3
model_stack={}
t_start=time()
for ind in range(stack_size):
    print(f'\n=== Model number {ind+1} ===\n')
    train_df, val_df=train_val_split(df,frac=0.2)
    model=train_model(train_df,val_df, n_epochs=5)
    y_predicted=get_predictions(val_df,model)
    y_true=val_df['target'].values
    best_thresh=select_best_threshold(y_true, y_predicted)
    model_stack[ind]=(model, best_thresh)
    print(f'Model number {ind+1} finished!')
t_finish=time()
print(f'Total time for training and calculating thresholds is {t_finish-t_start:.2f} seconds')
# predict on test set
test_set=pd.read_csv('../input/test.csv')
predictions={}
for key, (model, threshold) in model_stack.items():
    predictions[key]=(get_predictions(test_set,model)>threshold).astype(int)
preds=np.stack([value for value in predictions.values()],axis=1).reshape(-1,stack_size)
preds=np.average(preds,axis=1)
preds=np.around(preds,decimals=0).astype(int)
# test data
test_set['prediction']=preds
to_submit=test_set[['qid','prediction']]
to_submit.to_csv("submission.csv", index=False)
print('Submissions saved to file!')
print(f'Finishing at {datetime.datetime.now()}')
