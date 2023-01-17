import numpy as np

import pandas as pd

import os



import sklearn.model_selection

import sklearn.metrics

from catboost import CatBoostClassifier



import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

import gc

import time



from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes

from keras.models import *

from tqdm import tqdm # Processing time measurement

from sklearn.model_selection import train_test_split 

from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class

from keras import optimizers # Allow us to access the Adam class to modify some parameters

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model

from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting

from keras import activations

from keras import regularizers

from keras import initializers

from keras import constraints

from sklearn.preprocessing import MinMaxScaler

from numba import jit

from math import log, floor

from sklearn.neighbors import KDTree

from scipy.signal import periodogram, welch

from keras.engine import Layer

from keras.engine import InputSpec

from keras.objectives import categorical_crossentropy

from keras.objectives import sparse_categorical_crossentropy

import tensorflow as tf



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



plt.style.use('seaborn')

sns.set(font_scale=1)
df_train = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_train.csv')

df_train = df_train.set_index(['id_measurement', 'phase'])



X = np.load("../input/folk-base-neural-network-using-lstm/X.npy")

y = np.load("../input/folk-base-neural-network-using-lstm/y.npy")

features = np.load("../input/folk-base-neural-network-using-lstm/features.npy")
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
X.shape, y.shape
%%time

X, y = augment(X,y)

print(X.shape, y.shape)
features = np.load('../input/vsb-aug-features/aug_features.npy')

features.shape
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
def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator / (denominator + K.epsilon())
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.01 for i in range(100)]):

        score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}

    return search_result
# This is NN LSTM Model creation

def model_lstm(input_shape, feat_shape):

    inp = Input(shape=(input_shape[1], input_shape[2],))

    feat = Input(shape=(feat_shape[1],))



    bi_lstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(inp)

    bi_lstm_2 = Bidirectional(CuDNNGRU(64, return_sequences=True), merge_mode='concat')(bi_lstm_1)

    

    attention = Attention(input_shape[1])(bi_lstm_2)

    

    x = concatenate([attention, feat], axis=1)

    x = Dense(64, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    

    model = Model(inputs=[inp, feat], outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

    

    return model

# Here is where the training happens

# First, create a set of indexes of the 5 folds

N_SPLITS = 5

splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019).split(X, y))

preds_val = []

y_val = []

# Then, iteract with each fold

# If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]

for idx, (train_idx, val_idx) in enumerate(splits):

    K.clear_session() # I dont know what it do, but I imagine that it "clear session" :)

    print("Beginning fold {}".format(idx+1))

    # use the indexes to extract the folds in the train and validation data

    train_X, train_feat, train_y, val_X, val_feat, val_y = X[train_idx], features[train_idx], y[train_idx], X[val_idx], features[val_idx], y[val_idx]

    # instantiate the model for this fold

    model = model_lstm(train_X.shape, features.shape)

    # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an

    # validation matthews_correlation greater than the last one.

    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')

    # Train, train, train

    model.fit([train_X, train_feat], train_y, batch_size=128, epochs=50, validation_data=([val_X, val_feat], val_y), callbacks=[ckpt])

    # loads the best weights saved by the checkpoint

    model.load_weights('weights_{}.h5'.format(idx))

    # Add the predictions of the validation to the list preds_val

    preds_val.append(model.predict([val_X, val_feat], batch_size=512))

    # and the val true y

    y_val.append(val_y)



# concatenates all and prints the shape    

preds_val = np.concatenate(preds_val)[...,0]

y_val = np.concatenate(y_val)

preds_val.shape, y_val.shape

def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    

    y_pred = tf.convert_to_tensor(y_pred, np.float64)

    y_true = tf.convert_to_tensor(y_true, np.float64)

    

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator / (denominator + K.epsilon())
optimal_values = threshold_search(y_val, preds_val)

best_threshold = optimal_values['threshold']

best_score = optimal_values['matthews_correlation']
print("""------------------------------

Finished training a LSTM model.

CV scores: %.3f

------------------------------""" % (best_score))
meta_test = pd.read_csv('../input/vsb-power-line-fault-detection/metadata_test.csv')

X_test_input = np.load("../input/folk-base-neural-network-using-lstm/X_test.npy")

features_test = np.load("../input/folk-base-neural-network-using-lstm/features_test.npy")

submission = pd.read_csv('../input/vsb-power-line-fault-detection/sample_submission.csv')
preds_test = []

for i in range(N_SPLITS):

    model.load_weights('weights_{}.h5'.format(i))

    pred = model.predict([X_test_input, features_test], batch_size=300, verbose=1)

    pred_3 = []

    for pred_scalar in pred:

        for i in range(3):

            pred_3.append(pred_scalar)

    preds_test.append(pred_3)

optimal_values = threshold_search(y_val, preds_val)

best_threshold = optimal_values['threshold']

best_score = optimal_values['matthews_correlation']

preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)

preds_test.shape
submission['target'] = preds_test

submission.to_csv('submission.csv', index=False)

submission.head()
submission[submission.target == 1].info()