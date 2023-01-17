#Let's load in some basics and make sure our files are all here
import numpy as np
import pandas as pd
import os

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print(train.columns)
sample_submission.head(3)
cols = ["vid_id", "start_time_seconds_youtube_clip", "end_time_seconds_youtube_clip",
        "audio_embedding", "is_turkey"]
print(train.shape)
print(test.shape)
train[train['is_turkey']==1][cols].head(3)
"is_turkey rate is " + str(train[train['is_turkey']==1].shape[0] / train.shape[0])
print(train['audio_embedding'].head())

#see the possible list lengths of the first dimension
print("train's audio_embedding can have this many frames: "+ str(train['audio_embedding'].apply(lambda x: len(x)).unique())) 
print("test's audio_embedding can have this many frames: "+ str(test['audio_embedding'].apply(lambda x: len(x)).unique())) 

#see the possible list lengths of the first element
print("each frame can have this many features: "+str(train['audio_embedding'].apply(lambda x: len(x[0])).unique()))
sns.countplot(train['audio_embedding'].apply(lambda x: len(x)))
plt.show()
# train["time_seconds"] = train["end_time_seconds_youtube_clip"] - train["start_time_seconds_youtube_clip"]
# test["time_seconds"] = test["end_time_seconds_youtube_clip"] - test["start_time_seconds_youtube_clip"]
# sns.countplot(train["time_seconds"])
# plt.ylim(0,100)
# plt.show()
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout, Input
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
#split the training data to have a validation set
train_train, train_val = train_test_split(train, test_size=0.2, random_state=42, stratify=train["is_turkey"])
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)
## https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
    
## https://github.com/keras-team/keras/issues/3230#issuecomment-292535661
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
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
## https://www.kaggle.com/suicaokhoailang/lstm-with-attention-baseline-0-989-lb/notebook
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

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
#Define a basic LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, input_shape=(10, 128))))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Attention(10))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#maybe there is something better to use, but let's use binary_crossentropy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=["accuracy", auc])

# Callback
es = EarlyStopping(monitor='val_auc', min_delta=0, patience=5, verbose=0, mode='max')
roc_cb = roc_callback(training_data=(x_train, y_train),validation_data=(x_val, y_val))

#fit on a portion of the training data, and validate on the rest
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=256,
                    epochs=20,
                    verbose=2,
                    callbacks=[es, roc_cb])

# Evaluate
score, acc, auc = model.evaluate(x_val, y_val, batch_size=256, verbose=0)
print('Validation AUC:', auc)
plt.figure(figsize=(8,5))
sns.lineplot(range(1, len(history.history['auc'])+1), history.history['auc'], label='Train AUC')
sns.lineplot(range(1, len(history.history['auc'])+1), history.history['val_auc'], label='Test AUC')
plt.show()
df_val = pd.DataFrame({'vid_id':train_val['vid_id'].values,
                       'is_turkey':[x for y in model.predict(x_val) for x in y],
                       'start_time': train_val['start_time_seconds_youtube_clip'].values})
df_val.head(3)
plt.figure(figsize=(6,4))
sns.distplot(df_val["is_turkey"], bins=20, kde=False)
plt.show()
df_val[(df_val["is_turkey"]>0.05) & (df_val["is_turkey"]<0.95)]
test_data = [k for k in test['audio_embedding']]
submission = model.predict(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
print(submission.head()) #check to see that it looks like the sample submission
submission.to_csv('lstm_starter.csv', index=False) #drop the index so it matches the submission format.
from IPython.display import YouTubeVideo
YouTubeVideo("HvSsSQddil4",start=0)
