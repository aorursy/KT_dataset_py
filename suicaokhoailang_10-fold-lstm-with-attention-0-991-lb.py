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
def get_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(10, 128)))
#     model.add(Bidirectional(RNN(64, activation='relu', return_sequences=True)))
    model.add(Bidirectional(GRU(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
    model.add(Attention(10))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_model_v2():
    inp = Input(shape=(10,128))
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(inp)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    
    atten_1 = Attention(10)(x) # skip connect
    atten_2 = Attention(10)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
xtrain = [k for k in train['audio_embedding']]
for idx,x in enumerate(xtrain):
    if len(x) > 10:
        print(idx)
test_data = test['audio_embedding'].tolist()
ytrain = train['is_turkey'].values
# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain,maxlen=10)
# for x in xtrain:
#     x = np.array(x)
#     if x.shape[0] < 10:
#         x_ = np.zeros((10,128))
#         x_[:x.shape[0],:] = x
#         n_pads = math.ceil((10 - x.shape[0])/x.shape[0])
#         len_to_pad = min(x.shape[0],10-x.shape[0])
#         for i in range(n_pads):
#             x_[x.shape[0]*(i+1):x.shape[0]*(i+2),:] = x[:len_to_pad if i < n_pads - 1 \
#                                                         else 10 - x.shape[0] - i*len_to_pad,:]
#         x = x_
#     x_train.append(x)
x_train = np.array(x_train)
y_train = np.asarray(ytrain)

x_test = pad_sequences(test_data,maxlen=10)
# for x in test_data:
#     x = np.array(x)
#     if x.shape[0] < 10:
#         x_ = np.zeros((10,128))
#         x_[:x.shape[0],:] = x
#         n_pads = math.ceil((10 - x.shape[0])/x.shape[0])
#         len_to_pad = min(x.shape[0],10-x.shape[0])
#         for i in range(n_pads):
#             x_[x.shape[0]*(i+1):x.shape[0]*(i+2),:] = x[:len_to_pad if i < n_pads - 1 \
#                                                         else 10 - x.shape[0] - i*len_to_pad,:]
#         x = x_
#     x_test.append(x)
x_test = np.array(x_test)
kf = KFold(n_splits=10, shuffle=True, random_state=42069)
preds = []
fold = 0
aucs = 0
for train_idx, val_idx in kf.split(x_train):
    x_train_f = x_train[train_idx]
    y_train_f = y_train[train_idx]
    x_val_f = x_train[val_idx]
    y_val_f = y_train[val_idx]
    model = get_model()
    model.fit(x_train_f, y_train_f,
              batch_size=256,
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
preds = np.asarray(preds)[...,0]
preds = np.mean(preds, axis=0)
sub_df = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds})
# sub_df.to_csv('submission.csv', index=False)

probs = sub_df.is_turkey.values
n,bins,_ = plt.hist(probs,bins=100)
print(n, bins)
pos_threshold = 0.99
neg_threshold = 0.01
pseudo_index = np.argwhere(np.logical_or(probs > pos_threshold, probs < neg_threshold ))[:,0]
pseudo_x_train = x_test[pseudo_index]
pseudo_y_train = probs[pseudo_index]
pseudo_y_train[pseudo_y_train > 0.5] = 1
pseudo_y_train[pseudo_y_train <= 0.5] = 0
# x_train = np.concatenate([x_train, pseudo_x_train],axis=0)
# y_train = np.concatenate([y_train,pseudo_y_train])
# print(x_train.shape, y_train.shape)
kf = KFold(n_splits=10, shuffle=True, random_state=42069)
preds = []
fold = 0
aucs = 0
for train_idx, val_idx in kf.split(x_train):
    x_train_f = x_train[train_idx]
    y_train_f = y_train[train_idx]
    x_val_f = x_train[val_idx]
    y_val_f = y_train[val_idx]
    model = get_model()
    model.fit(pseudo_x_train, pseudo_y_train,
              batch_size=256,
              epochs=3,
              verbose = 0,
              validation_data=(x_val_f, y_val_f))
    model.fit(x_train_f, y_train_f,
              batch_size=256,
              epochs=16,
              verbose = 0,
              validation_data=(x_val_f, y_val_f))
    preds_val = model.predict([x_val_f], batch_size=512)
    preds.append(model.predict(x_test))
    fold+=1
    fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)
    aucs += auc(fpr,tpr)
    print('Fold {}, AUC = {}'.format(fold,auc(fpr, tpr)))
print("Cross Validation AUC = {}".format(aucs/10))
preds = np.asarray(preds)[...,0]
preds = np.mean(preds, axis=0)
sub_df = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds})
sub_df.to_csv('submission.csv', index=False)