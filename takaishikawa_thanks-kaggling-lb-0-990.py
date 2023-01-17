import numpy as np
import pandas as pd
import os

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import gc
import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print(train.columns)
print(sample_submission.head(3))
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
# plt.ylim(0,100)
plt.show()
np.array(train['audio_embedding'].iloc[0]).shape
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout, Input, CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
#split the training data to have a validation set
xtrain = train['audio_embedding'].tolist()
ytrain = train['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
test_data = pad_sequences(test['audio_embedding'].tolist())

y_train = np.asarray(ytrain)
## https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
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

        self.attention_weights = self.add_weight((input_shape[-1],),  # https://github.com/keras-team/keras/issues/7736
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
                        K.reshape(self.attention_weights, (features_dim, 1))), (-1, step_dim))

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
def get_model():
    #Define a basic LSTM model
    model = Sequential()
    model.add(BatchNormalization(input_shape=(10, 128)))
    model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Attention(10))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    #maybe there is something better to use, but let's use binary_crossentropy
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[auc])

    # Callback
    best_weights_filepath = 'best_weights.hdf5'
    es = EarlyStopping(monitor='val_auc', min_delta=0, patience=5, verbose=0, mode='max')
    mcp = ModelCheckpoint(best_weights_filepath, monitor='val_auc', verbose=0, save_best_only=True, mode='max')
#     roc_cb = roc_callback(training_data=(x_tr, y_tr),validation_data=(x_val, y_val))
    
    return model, es, mcp, best_weights_filepath
epochs = 15
# folds = KFold(n_splits=5)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = []
for k, (tr_idx, val_idx) in enumerate(folds.split(x_train, ytrain)):
    x_tr = x_train[tr_idx]
    y_tr = y_train[tr_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    model, es, mcp, bst_w_fp = get_model()
    model.fit(x_tr, 
              y_tr,
              validation_data=(x_val, y_val),
              batch_size=256,
              epochs=epochs,
              callbacks=[es, mcp],
              verbose=0)
    # Evaluate and Prediction
    model.load_weights(bst_w_fp)
    loss, auc_score = model.evaluate(x_val, y_val, batch_size=256, verbose=0)
    preds.append(model.predict(test_data))
    del model, x_tr, y_tr, x_val, y_val
    gc.collect()
    print('Validation AUC:', auc_score)
preds = np.asarray(preds)[...,0]
preds_1 = np.mean(preds, axis=0)
submission1 = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds_1})
submission1['is_turkey'] = submission1.is_turkey
submission1.to_csv('submission1.csv', index=False)
submission1.head(10)
def remove_minmax(arr, mode):
    if mode=="min":
        m = np.min(arr)
    elif mode=="max":
        m = np.max(arr)
    m_loc = np.where(arr == m)[0][0]
    arr = np.delete(arr, m_loc)
    return arr
preds_2 = []
for row in preds.T:
    row = remove_minmax(row, "min")
    row = remove_minmax(row, "max")
    rmax = np.max(row)
    rmin = np.max(row)
    if rmax > 1 - rmin:
        preds_2.append(rmax)
    else:
        preds_2.append(rmin)
submission2 = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds_2})
submission2['is_turkey'] = submission2.is_turkey
submission2.to_csv('submission_2.csv', index=False)
submission2.head(10)
preds_3 = np.median(preds, axis=0)
submission3 = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds_3})
submission3['is_turkey'] = submission3.is_turkey
submission3.to_csv('submission_3.csv', index=False)
submission3.head(10)
preds_4 = []
for row in preds.T:
    row = remove_minmax(row, "min")
    row = remove_minmax(row, "max")
    preds_4.append(np.mean(row))
submission4 = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds_4})
submission4['is_turkey'] = submission4.is_turkey
submission4.to_csv('submission_4.csv', index=False)
submission4.head(10)
