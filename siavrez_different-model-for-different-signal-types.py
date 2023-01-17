!pip install --no-warn-conflicts -q tensorflow-addons
from tensorflow.keras.layers import (TimeDistributed, Dropout, BatchNormalization, Flatten, Convolution1D, Activation, Input, Dense, LSTM, Lambda, Bidirectional,

                                     Add, AveragePooling1D, Multiply, GRU, GRUCell, LSTMCell, SimpleRNNCell, SimpleRNN, TimeDistributed, RNN,

                                     RepeatVector, Conv1D, MaxPooling1D, Concatenate, GlobalAveragePooling1D, UpSampling1D)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau, LearningRateScheduler

from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer,  StandardScaler, MaxAbsScaler

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras import losses, models, optimizers

from tensorflow.keras import backend as K

import tensorflow as tf

from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error

from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO

from sklearn.model_selection import KFold, GroupKFold

from tqdm import tqdm_notebook as tqdm

from contextlib import contextmanager

from joblib import Parallel, delayed

from IPython.display import display

from sklearn import preprocessing

from scipy.special import erfinv

import tensorflow_addons as tfa

import scipy.stats as stats

import random as rn

import pandas as pd

import numpy as np

import scipy as sp

import itertools

import warnings

import time

import pywt

import os

import gc





warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 500)

%matplotlib inline
EPOCHS=125

NNBATCHSIZE=16

BATCHSIZE = 4000

SERIESIZE = 500000

SEED = 321

SELECT = True

SPLITS = 5

LR = 0.001

fe_config = [

    (True, 4000),

]


def init_logger():

    handler = StreamHandler()

    handler.setLevel(INFO)

    handler.setFormatter(Formatter(LOGFORMAT))

    fh_handler = FileHandler('{}.log'.format(MODELNAME))

    fh_handler.setFormatter(Formatter(LOGFORMAT))

    logger.setLevel(INFO)

    logger.addHandler(handler)

    logger.addHandler(fh_handler)

    


@contextmanager

def timer(name : Text):

    t0 = time.time()

    yield

    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')



COMPETITION = 'ION-Switching'

logger = getLogger(COMPETITION)

LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'

MODELNAME = 'WaveNet'



def seed_everything(seed : int) -> NoReturn :

    

    rn.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(SEED)



def read_data(base : os.path.abspath) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    

    train = pd.read_csv('/kaggle/input/detrendedwithkalman/train_kalman.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('/kaggle/input/detrendedwithkalman/test_kalman.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})

    

    return train, test, sub



def batching(df : pd.DataFrame,

             batch_size : int,

             colname: Text) -> pd.DataFrame :

    

    df[colname] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df[colname] = df[colname].astype(np.uint16)

        

    return df



def reduce_mem_usage(df: pd.DataFrame,

                     verbose: bool = True) -> pd.DataFrame:

    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2



    for col in df.columns:

        col_type = df[col].dtypes



        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()



            if str(col_type)[:3] == 'int':



                if (c_min > np.iinfo(np.int32).min

                      and c_max < np.iinfo(np.int32).max):

                    df[col] = df[col].astype(np.int32)

                elif (c_min > np.iinfo(np.int64).min

                      and c_max < np.iinfo(np.int64).max):

                    df[col] = df[col].astype(np.int64)

            else:

                if (c_min > np.finfo(np.float16).min

                        and c_max < np.finfo(np.float16).max):

                    df[col] = df[col].astype(np.float16)

                elif (c_min > np.finfo(np.float32).min

                      and c_max < np.finfo(np.float32).max):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    reduction = (start_mem - end_mem) / start_mem



    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'

    if verbose:

        print(msg)



    return df



def lag_with_pct_change(df : pd.DataFrame,

                        shift_sizes : Optional[List]=[1, 2],

                        add_pct_change : Optional[bool]=False,

                        add_pct_change_lag : Optional[bool]=False) -> pd.DataFrame:

    

    for shift_size in shift_sizes:    

        df['signal_shift_pos_'+str(shift_size)] = df.groupby('group')['signal'].shift(shift_size).fillna(0)

        df['signal_shift_neg_'+str(shift_size)] = df.groupby('group')['signal'].shift(-1*shift_size).fillna(0)



    if add_pct_change:

        df['pct_change'] = df['signal'].pct_change()

        if add_pct_change_lag:

            for shift_size in shift_sizes:    

                df['pct_change_shift_pos_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(shift_size).fillna(0)

                df['pct_change_shift_neg_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(-1*shift_size).fillna(0)

    return df



def run_feat_enginnering(df : pd.DataFrame,

                         create_all_data_feats : bool,

                         batch_size : int) -> pd.DataFrame:

    

    df = batching(df, batch_size=batch_size, colname='group')



    if create_all_data_feats:

        df = lag_with_pct_change(df, [1, 2, 3],  add_pct_change=False, add_pct_change_lag=False)

    df['signal_2'] = df['signal'] ** 2

    return df

def feature_selection(df : pd.DataFrame,

                      df_test : pd.DataFrame) -> Tuple[pd.DataFrame , pd.DataFrame, List]:

    use_cols = [col for col in df.columns if col not in ['index','group', 'open_channels', 'time', 'batch']]

    df = df.replace([np.inf, -np.inf], np.nan)

    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    for col in use_cols:

        col_mean = pd.concat([df[col], df_test[col]], axis=0).mean()

        df[col] = df[col].fillna(col_mean)

        df_test[col] = df_test[col].fillna(col_mean)

   

    gc.collect()

    return df, df_test, use_cols



def augment(X: np.array, y:np.array) -> Tuple[np.array, np.array]:

    

    X = np.vstack((X, np.flip(X, axis=1)))

    y = np.vstack((y, np.flip(y, axis=1)))

    

    return X, y


def run_cv_model_by_batch(train : pd.DataFrame,

                          test : pd.DataFrame,

                          splits : int,

                          batch_col : Text,

                          feats : List,

                          sample_submission: pd.DataFrame,

                          nn_epochs : int,

                          nn_batch_size : int) -> NoReturn:

    seed_everything(SEED)

    K.clear_session()

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

    tf.compat.v1.keras.backend.set_session(sess)



    oof_ = np.zeros((len(train), 11))

    preds_ = np.zeros((len(test), 11))

    target = ['open_channels']

    group = train['group']

    kf = GroupKFold(n_splits=5)

    splits = [x for x in kf.split(train, train[target], group)]



    new_splits = []

    for sp in splits:

        new_split = []

        new_split.append(np.unique(group[sp[0]]))

        new_split.append(np.unique(group[sp[1]]))

        new_split.append(sp[1])    

        new_splits.append(new_split)

        

    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)



    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

    target_cols = ['target_'+str(i) for i in range(11)]

    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))

    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))



    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):

        train_x, train_y = train[tr_idx], train_tr[tr_idx]

        valid_x, valid_y = train[val_idx], train_tr[val_idx]

        train_x, train_y = augment(train_x, train_y)

        

        gc.collect()

        shape_ = (None, train_x.shape[2])

        model = Classifier(shape_)

        cb_lr_schedule = LearningRateScheduler(lr_schedule)

        cb_prg = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False,leave_overall_progress=False, show_epoch_progress=False,show_overall_progress=True)

        model.fit(train_x,train_y,

                  epochs=nn_epochs,

                  callbacks=[cb_prg, cb_lr_schedule, MacroF1(model, valid_x,valid_y)],

                  batch_size=nn_batch_size,verbose=0,

                  validation_data=(valid_x,valid_y))

        preds_f = model.predict(valid_x)

        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro')

        logger.info(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')

        preds_f = preds_f.reshape(-1, preds_f.shape[-1])

        oof_[val_orig_idx,:] += preds_f

        te_preds = model.predict(test)

        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           

        preds_ += te_preds / SPLITS

    f1_score_ =f1_score(np.argmax(train_tr, axis=2).reshape(-1),  np.argmax(oof_, axis=1), average = 'macro')

    logger.info(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')

    sample_submission['open_channels'] = np.argmax(preds_, axis=1).astype(int)

    sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')

    display(sample_submission.head())

    np.save('oof.npy', oof_)

    np.save('preds.npy', preds_)



    return 

def lr_schedule(epoch):

    if epoch < 40:

        lr = LR

    elif epoch < 50:

        lr = LR / 3

    elif epoch < 60:

        lr = LR / 6

    elif epoch < 75:

        lr = LR / 9

    elif epoch < 85:

        lr = LR / 12

    elif epoch < 100:

        lr = LR / 15

    elif epoch < 110:    

        lr = LR / 25       

    else:

        lr = LR / 50

    return lr
class Mish(tf.keras.layers.Layer):



    def __init__(self, **kwargs):

        super(Mish, self).__init__(**kwargs)

        self.supports_masking = True



    def call(self, inputs):

        return inputs * K.tanh(K.softplus(inputs))



    def get_config(self):

        base_config = super(Mish, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



    def compute_output_shape(self, input_shape):

        return input_shape

def mish(x):

	return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

 

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.layers import Activation

get_custom_objects().update({'mish': Activation(mish)})

def categorical_focal_loss(gamma=2.0, alpha=0.25):

    """

    Implementation of Focal Loss from the paper in multiclass classification

    Formula:

        loss = -alpha*((1-p)^gamma)*log(p)

    Parameters:

        alpha -- the same as wighting factor in balanced cross entropy

        gamma -- focusing parameter for modulating factor (1-p)

    Default value:

        gamma -- 2.0 as mentioned in the paper

        alpha -- 0.25 as mentioned in the paper

    """

    def focal_loss(y_true, y_pred):

        # Define epsilon so that the backpropagation will not result in NaN

        # for 0 divisor case

        epsilon = K.epsilon()

        # Add the epsilon to prediction value

        #y_pred = y_pred + epsilon

        # Clip the prediction value

        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)

        # Calculate cross entropy

        cross_entropy = -y_true*K.log(y_pred)

        # Calculate weight that consists of  modulating factor and weighting factor

        weight = alpha * y_true * K.pow((1-y_pred), gamma)

        # Calculate focal loss

        loss = weight * cross_entropy

        # Sum the losses in mini_batch

        loss = K.sum(loss, axis=1)

        return loss

    

    return focal_loss
def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):



    def build_residual_block(l_input):

        resid_input = l_input

        for dilation_rate in [2**i for i in range(stacked_layer)]:

            l_sigmoid_conv1d = Conv1D(

              num_filters, kernel_size, dilation_rate=dilation_rate,

              padding='same', activation='sigmoid')(l_input)

            l_tanh_conv1d = Conv1D(

             num_filters, kernel_size, dilation_rate=dilation_rate,

             padding='same', activation='mish')(l_input)

            l_input = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])

            l_input = Conv1D(num_filters, 1, padding='same')(l_input)

            resid_input = Add()([resid_input ,l_input])

        return resid_input

    return build_residual_block





def Classifier(shape_):

    num_filters_ = 16

    kernel_size_ = 4

    stacked_layers_ = [12, 8, 4, 1]

    l_input = Input(shape=(shape_))

    x = Conv1D(num_filters_, 1, padding='same')(l_input)

    x = WaveNetResidualConv1D(num_filters_, kernel_size_, stacked_layers_[0])(x)

    x = Conv1D(num_filters_*2, 1, padding='same')(x)

    x = WaveNetResidualConv1D(num_filters_*2, kernel_size_, stacked_layers_[1])(x)

    x = Conv1D(num_filters_*4, 1, padding='same')(x)

    x = WaveNetResidualConv1D(num_filters_*4, kernel_size_, stacked_layers_[2])(x)

    x = Conv1D(num_filters_*8, 1, padding='same')(x)

    x = WaveNetResidualConv1D(num_filters_*8, kernel_size_, stacked_layers_[3])(x)

    l_output = Dense(11, activation='softmax')(x)

    model = models.Model(inputs=[l_input], outputs=[l_output])

    opt = Adam(lr=LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(loss=categorical_focal_loss(gamma=3.0, alpha=0.25), optimizer=opt, metrics=['categorical_accuracy'])

    return model
class MacroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis=2).reshape(-1)



    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)

        score = f1_score(self.targets, pred, average="macro")

        print(f' F1Macro: {score:.5f}')    

    
class GaussRankScaler():



    def __init__( self ):

        self.epsilon = 0.001

        self.lower = -1 + self.epsilon

        self.upper =  1 - self.epsilon

        self.range = self.upper - self.lower



    def fit_transform( self, X ):

	

        i = np.argsort( X, axis = 0 )

        j = np.argsort( i, axis = 0 )



        assert ( j.min() == 0 ).all()

        assert ( j.max() == len( j ) - 1 ).all()



        j_range = len( j ) - 1

        self.divider = j_range / self.range



        transformed = j / self.divider

        transformed = transformed - self.upper

        transformed = erfinv( transformed )

        return transformed

def rankgaus_transform(train, test):

    

  shape_ = train.shape[0]

  data = pd.concat([train[['signal']], test[['signal']]]).values

  rgsc = GaussRankScaler()

  data = rgsc.fit_transform(data)

  train['signal'] = data[:shape_]

  test['signal'] = data[shape_:]

    

  return train, test
def scale(train, test):

    

    sc = StandardScaler()

    train['signal'] = sc.fit_transform(train.signal.values.reshape(-1, 1))

    test['signal'] = sc.transform(test.signal.values.reshape(-1, 1))



    return train, test


def run_everything(fe_config : List) -> NoReturn:

    not_feats_cols = ['time']

    target_col = ['open_channels']

    init_logger()

    with timer(f'Reading Data'):

        logger.info('Reading Data Started ...')

        base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')

        train, test, sample_submission = read_data(base)

        train = batching(train, batch_size=SERIESIZE, colname='batch')

        selection = [0,2,3,4,5]

        train = train.loc[train['batch'].isin(selection)]

        train = train.reset_index(drop=True)

        train = train.drop('batch', axis=1)

        train, test = scale(train, test)    

        logger.info('Reading and Normalizing Data Completed ...')

    with timer(f'Creating Features'):

        logger.info('Feature Enginnering Started ...')

        for config in fe_config:

            train = run_feat_enginnering(train, create_all_data_feats=config[0], batch_size=config[1])

            test  = run_feat_enginnering(test,  create_all_data_feats=config[0], batch_size=config[1])

        train, test, feats = feature_selection(train, test)

        logger.info('Feature Enginnering Completed ...')



    with timer(f'Running Wavenet model'):

        logger.info(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started ...')

        run_cv_model_by_batch(train, test, splits=SPLITS, batch_col='group', feats=feats, sample_submission=sample_submission, nn_epochs=EPOCHS, nn_batch_size=NNBATCHSIZE)

        logger.info(f'Training completed ...')

run_everything(fe_config)