#Let's load in some basics and make sure our files are all here

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import math

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))



from keras import Sequential

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import KFold

from sklearn.metrics import *

from keras.models import Sequential,Model

from keras.layers import *



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')

sample_submission = pd.read_csv('../input/sample_submission.csv')
print(test.columns)

print(train.columns)

print(sample_submission.head(4))
print(train.shape)

print(test.shape)
print(train[train['is_turkey']==1].index)
print(train['audio_embedding'].head())



#see the possible list lengths of the first dimension

print("train's audio_embedding have this many frames(seconds): "+ str(train['audio_embedding'].apply(lambda x: len(x)).unique())) 

print("test's audio_embedding have this many frames(seconds): "+ str(test['audio_embedding'].apply(lambda x: len(x)).unique())) 



#see the possible list lengths of the first element

print("each frame has this many features: "+str(train['audio_embedding'].apply(lambda x: len(x[0])).unique()))
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
def get_model1():

    model = Sequential()

    model.add(BatchNormalization(input_shape=(10, 128)))

    model.add(Bidirectional(GRU(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))

    model.add(Attention(10))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
xtrain = [k for k in train['audio_embedding']]

test_data = test['audio_embedding'].tolist()

ytrain = train['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long

x_train = pad_sequences(xtrain,maxlen=10)

x_test = pad_sequences(test_data,maxlen=10)



kf = KFold(n_splits=10, shuffle=True, random_state=42069)

preds = []

fold = 0

aucs = 0

for train_idx, val_idx in kf.split(x_train):

    x_train_f = x_train[train_idx]

    y_train_f = ytrain[train_idx]

    x_val_f = x_train[val_idx]

    y_val_f = ytrain[val_idx]

    model = get_model1()

    model.fit(x_train_f, y_train_f,

              batch_size=50,

              epochs=16,

              verbose = 0,

              validation_data=(x_val_f, y_val_f))

    # Get accuracy of model on validation data. It's not AUC but it's something at least!

    preds_val = model.predict([x_val_f], batch_size=512)

    preds.append(model.predict(x_test))

    fold+=1

    fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)

    aucs += auc(fpr,tpr)

    print('Fold {}, AUC = {}'.format(fold,auc(fpr, tpr)))

print("Cross Validation AUC = {}".format(aucs/10))
test_data = [k for k in test['audio_embedding']]

submission = model.predict_classes(pad_sequences(test_data))

submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
print(submission.head()) #check to see that it looks like the sample submission
submission.to_csv('lstm_starter.csv', index=False) #drop the index so it matches the submission format.