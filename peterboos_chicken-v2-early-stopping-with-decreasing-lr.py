import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))

from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras import optimizers
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization,CuDNNLSTM, GRU, CuDNNGRU
from keras.layers import Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, GaussianNoise, GaussianDropout
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from sklearn.model_selection import KFold, cross_val_score, train_test_split
train = pd.read_json('../input/train.json')
train, train_val = train_test_split(train)
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')

train_train, train_val = train_test_split(train)
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)
print("loaded")
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
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
print("Custom layer created.")

try:
    print("try to delete previous hfd5") 
    filepath="Chicken-weights.best.hdf5"
    import os.path
    import os
    os.remove(filepath)
    print("File Removed!")
except:
    print("no file there")
runs = 100
estop = 50 #early stop
ep = 150   #epochs  we do early stopping but we do also 100 repeated runs !!
dp = 0.4   #dropout rate
bs =16     # batch size

import time
import random
from datetime import datetime
random.seed(datetime.now())
np.random.seed(int(time.time()))

print ("learn another ",ep," epochs")
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
import os.path
import os
filepath="Chicken-weights.best.hdf5"
act = 'tanh'
model = Sequential()
# model.add(BatchNormalization(input_shape=(10, 128)))
# model.add(Bidirectional(LSTM(128, dropout=dp, recurrent_dropout=dp, activation=act, return_sequences=True)))
# #model.add(Bidirectional(LSTM(128, dropout=dp, recurrent_dropout=dp, activation=act, return_sequences=False)))
# model.add(Bidirectional(LSTM(128, dropout=dp, recurrent_dropout=dp, activation=act, return_sequences=True)))
model.add(GaussianNoise( stddev=0.03 )) # adding noice to wiggle input improving the robustness search (i hope)
model.add(GaussianDropout(0.2))
#GaussianDropout(rate)
model.add(BatchNormalization(momentum=0.98,input_shape=(10, 128)))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
#model.add(Bidirectional(LSTM(128, dropout=dp, recurrent_dropout=dp, activation=act, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))
model.add(Attention(10))
# model.add(GaussianNoise( stddev=0.05 )) # adding noice as to make it harder to find and improving the search (i hope)
model.add(BatchNormalization(momentum=0.98,input_shape=(10, 128)))
model.add(Dense(32,activation=act))

model.add(Dense(1,activation='sigmoid'))
#model.summary()

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, verbose=1, min_lr=1e-8)
early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=estop,  restore_best_weights=True)
# reduce_lr1 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,reduce_lr,early_stop]
callbacks_list = [checkpoint,early_stop]
# callbacks_list = [checkpoint]
# each time we run we want to improve so make sure we randomize our libraries, its not about reproduceable results,
#eventually its about perfect wieghts in our networks, after each training.


if (os.path.exists(filepath)):
    print("Extending training of previous run")
    
#     with open('model_architecture.json', 'r') as f:
#          model = model_from_json(f.read())
    model.load_weights(filepath, by_name=False)
    model.compile(loss='binary_crossentropy',optimizer = optimizers.Adam(lr=w) , metrics=['accuracy'])    
    score, acc = model.evaluate(x_val, y_val, batch_size=32)
    print('Previous test accuracy:', acc)
    for x in range(1, runs):
        w = 0.1/(x*x*3)
        print("main run ",x, " lr=",w)
        model.compile(loss='binary_crossentropy', optimizer = optimizers.Adam(lr=w), metrics=['accuracy'])#lr 0.001
   

else:
    print("First run")      
    for x in range(1, runs):
        w = 0.1/(x*x*3)
        print("main run ",x, " lr=",w)
        model.compile(loss='binary_crossentropy', optimizer = optimizers.Adam(lr=w), metrics=['accuracy'])#lr 0.001
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=ep, batch_size=bs, callbacks=callbacks_list, verbose=0)

print("you can run this cell again to keep on training, or go on")
# Get accuracy of model on validation data. It's not AUC but it's something at least!
model.load_weights(filepath, by_name=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(filepath, by_name=False)
score, acc = model.evaluate(x_val, y_val, batch_size=32)
print('Test accuracy:', acc)
test_data = test['audio_embedding'].tolist()
submission = model.predict(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
submission['is_turkey'] = submission.is_turkey
# submission.is_turkey =submission.is_turkey.round(0)
print(submission.head(40))
submission.to_csv('submission.csv', index=False)