# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.
import random
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import roc_curve, auc, accuracy_score, cohen_kappa_score
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix

import tensorflow as tf
import tensorflow.keras as keras


from tensorflow.keras.models import Sequential, Model

#from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM, Bidirectional, add, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv2DTranspose, AveragePooling1D, UpSampling1D
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, TimeDistributed
from tensorflow.keras.layers import Multiply, Add, Concatenate, Flatten, Average, Lambda

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.constraints import unit_norm, max_norm
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import np_utils

from tensorflow.keras import backend as K
#to reduce data size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col != 'time':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_stats(df):
    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])
    for col in df.columns:
        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]
    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]
    return stats

def print_header():
    print('col         conversion        dtype    na    uniq  size')
    print()
    
def print_values(name, conversion, col):
    template = '{:10}  {:16}  {:>7}  {:2}  {:6}  {:1.2f}MB'
    print(template.format(name, conversion, str(col.dtypes), col.isna().sum(), col.nunique(dropna=False), col.memory_usage(deep=True, index=False) / 1024 ** 2))
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)    
import tensorflow as tf
import keras.backend as K

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
def display_set(df, column, n_sample, figsize ):
    f, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = figsize )
    sns.lineplot(x= df.index[::n_sample], y = df[column][::n_sample], ax=ax1)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.random.set_seed(seed)  
# Showing Confusion Matrix
# credit to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(14,14)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
def get_class_weight(classes, exp=1):
    '''
    Weight of the class is inversely proportional to the population of the class.
    There is an exponent for adding more weight.
    '''
    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)
    class_weight = hist.sum()/np.power(hist, exp)
    
    return class_weight

# credit to https://www.kaggle.com/siavrez/simple-eda-model
def MacroF1Metric(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    return ('MacroF1Metric', score, True)

def multiclass_F1_Metric(preds, dtrain):
    labels = dtrain.get_label()
    num_labels = 11
    preds = preds.reshape(num_labels, len(preds)//num_labels)
    preds = np.argmax(preds, axis=0)
    score = f1_score(labels, preds, average="macro")
    return ('MacroF1Metric', score, True)
from functools import partial
import scipy as sp
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize F1 (Macro) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -f1_score(y, X_p, average = 'macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
    
    
def optimize_predictions(prediction, coefficients):
    prediction[prediction <= coefficients[0]] = 0
    prediction[np.where(np.logical_and(prediction > coefficients[0], prediction <= coefficients[1]))] = 1
    prediction[np.where(np.logical_and(prediction > coefficients[1], prediction <= coefficients[2]))] = 2
    prediction[np.where(np.logical_and(prediction > coefficients[2], prediction <= coefficients[3]))] = 3
    prediction[np.where(np.logical_and(prediction > coefficients[3], prediction <= coefficients[4]))] = 4
    prediction[np.where(np.logical_and(prediction > coefficients[4], prediction <= coefficients[5]))] = 5
    prediction[np.where(np.logical_and(prediction > coefficients[5], prediction <= coefficients[6]))] = 6
    prediction[np.where(np.logical_and(prediction > coefficients[6], prediction <= coefficients[7]))] = 7
    prediction[np.where(np.logical_and(prediction > coefficients[7], prediction <= coefficients[8]))] = 8
    prediction[np.where(np.logical_and(prediction > coefficients[8], prediction <= coefficients[9]))] = 9
    prediction[prediction > coefficients[9]] = 10
    
    return prediction    

PATH = '/kaggle/input/liverpool-ion-switching/'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')

train.head()
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
f, ax = plt.subplots(figsize=(15, 6))
sns.countplot(x="open_channels", data=train, ax=ax)
RANDOM_SEED = 42
GROUP_BATCH_SIZE = 2000

seed_everything(RANDOM_SEED)
def baseline_model(input_shape, units = 64, max_channels = 11, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units))
    model.add(Dense(units))
    model.add(Dense(max_channels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])
    return model
## it seems LSTM model will not converge with all generated features (at least for me :) 
## so, select some that work

## reshape for LSTM
#target=train['open_channels']
#train=train.drop(['open_channels'],axis=1)

X = train.values.reshape(-1,2,1)
## using categorical_crossentropy
yy = to_categorical(target, num_classes=11)

train_idx, val_idx = train_test_split(np.arange(X.shape[0]), random_state = RANDOM_SEED, test_size = 0.2)

X_t = X[train_idx] 
y_t = yy[train_idx] 
X_v = X[val_idx]
y_v = yy[val_idx]

BATCH_SIZE = 64
EPOCHS = 6

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000001, verbose=1)

adam = Adam(0.003)

model = baseline_model(X_t.shape, optimizer=adam)
history = model.fit( X_t, y_t, validation_data=(X_v, y_v), callbacks=[es,lr],
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, 
                    shuffle=False, workers=8, use_multiprocessing=True )
_, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (16,8))
sns.lineplot(data=np.asarray(history.history['loss']), label='loss', ax=ax1)
sns.lineplot(data=np.asarray(history.history['val_loss']), label='val_loss', ax=ax1)
sns.lineplot(data=np.asarray(history.history['acc']), label='acc', ax=ax1)
sns.lineplot(data=np.asarray(history.history['val_acc']), label='val_acc', ax=ax1)
sns.lineplot(data=np.asarray(history.history['f1']), label='f1', ax=ax1)
sns.lineplot(data=np.asarray(history.history['val_f1']), label='val_f1', ax=ax1)
y_pred = np.argmax(model.predict(X_v), axis=1).reshape(-1)
yhat = y_train[val_idx]

print("F1 MACRO: ", f1_score(yhat, y_pred, average="macro"))
y_pred = np.argmax(model.predict(X_all_test.values.reshape(-1,len(SELECTED_FEATURES), 1)), axis=1).reshape(-1)
sub = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
sub.open_channels = np.array(np.round(y_pred,0), np.int)
sub.to_csv("submission.csv",index=False)
sub.head(25)
