!pip install --no-warn-conflicts -q tensorflow-addons
from tensorflow.keras.layers import (Dropout, BatchNormalization, Flatten, Convolution1D, Activation, Input, Dense, LSTM, Lambda, Bidirectional,

                                     GRU, GRUCell, LSTMCell, SimpleRNNCell, SimpleRNN, TimeDistributed, RNN, RepeatVector)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau, LearningRateScheduler

from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras import losses, models, optimizers

from tensorflow.keras import backend as K

from tensorflow.keras import losses

import tensorflow as tf

from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error

from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO

from scipy.signal import butter, lfilter, filtfilt, savgol_filter, detrend

from sklearn.model_selection import KFold, GroupKFold

from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.linear_model import LinearRegression

from tqdm import tqdm_notebook as tqdm

from contextlib import contextmanager

from joblib import Parallel, delayed

from IPython.display import display

from sklearn import preprocessing

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
EPOCHS=50

NNBATCHSIZE=16

BATCHSIZE = 50000

SEED = 321

SELECT = True

SPLITS = 5

LR = 0.01

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

MODELNAME = 'Baseline'



def seed_everything(seed : int) -> NoReturn :

    

    rn.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(SEED)



def read_data(base : os.path.abspath) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    

    train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('../input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv(os.path.join(base + '/sample_submission.csv'), dtype={'time': np.float32})

    

    return train, test, sub







def batching(df : pd.DataFrame,

             batch_size : int) -> pd.DataFrame :

    

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

        

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

            df['pct_change_shift_pos_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(shift_size).fillna(0)

            df['pct_change_shift_neg_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(-1*shift_size).fillna(0)

    return df



def run_feat_enginnering(df : pd.DataFrame,

                         create_all_data_feats : bool,

                         batch_size : int) -> pd.DataFrame:

    

    df = batching(df, batch_size=batch_size)

    if create_all_data_feats:

        df = lag_with_pct_change(df, [1, 2, 3, 4, 5, 6],  add_pct_change=False, add_pct_change_lag=False)

    

    return df

def feature_selection(df : pd.DataFrame,

                      df_test : pd.DataFrame) -> Tuple[pd.DataFrame , pd.DataFrame, List]:

    use_cols = [col for col in df.columns if col not in ['index','group', 'open_channels', 'time']]

    df = df.replace([np.inf, -np.inf], np.nan)

    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    for col in use_cols:

        col_mean = pd.concat([df[col], df_test[col]], axis=0).mean()

        df[col] = df[col].fillna(col_mean)

        df_test[col] = df_test[col].fillna(col_mean)

   

    gc.collect()

    return df, df_test, use_cols



def run_cv_model_by_batch(train : pd.DataFrame,

                          test : pd.DataFrame,

                          splits : int,

                          batch_col : Text,

                          feats : List,

                          sample_submission: pd.DataFrame,

                          nn_epochs : int,

                          nn_batch_size : int) -> NoReturn:

    

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



        gc.collect()

        K.clear_session()

        shape_ = (None, train_x.shape[2])

        model = Classifier(shape_)

        cb_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        cb_Early_Stop=EarlyStopping(monitor='val_loss',patience=10)

        cb_clr = CyclicLR(base_lr=1e-7, max_lr = 1e-4, step_size= int(1.0*(test.shape[0])/(nn_batch_size*4)) , mode='exp_range', gamma=1.0, scale_fn=None, scale_mode='cycle')

        cb_prg = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False,leave_overall_progress=False, show_epoch_progress=False,show_overall_progress=True)

        model.fit(train_x,train_y,

                  epochs=nn_epochs,

                  callbacks=[cb_prg, cb_schedule, MacroF1(model, valid_x,valid_y)],

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

class CyclicLR(tf.keras.callbacks.Callback):



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1 / (2. ** (x - 1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma ** (x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.



    def clr(self):

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(

                self.clr_iterations)



    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())



    def on_batch_end(self, epoch, logs=None):



        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        K.set_value(self.model.optimizer.lr, self.clr())

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

def Classifier(shape_):



    inp = Input(shape=(shape_))



    x = Bidirectional(GRU(128,return_sequences=True))(inp)

    x = Bidirectional(GRU(128,return_sequences=True))(x)

    x = Bidirectional(GRU(128,return_sequences=True))(x)

    

    out = Dense(11, activation='softmax', name='out')(x)



    model = models.Model(inputs=inp, outputs=out)    

    opt = Adam(lr=LR)

    #opt = tfa.optimizers.Lookahead(opt)

    #loss = tfa.losses.SigmoidFocalCrossEntropy()

    model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), optimizer=opt, metrics=['accuracy'])

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

    
def normalize(train, test):

    

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal-train_input_mean)/train_input_sigma

    test['signal'] = (test.signal-train_input_mean)/train_input_sigma



    return train, test


def run_everything(fe_config : List) -> NoReturn:

    not_feats_cols = ['time']

    target_col = ['open_channels']

    init_logger()

    with timer(f'Reading Data'):

        logger.info('Reading Data Started ...')

        base = os.path.abspath('/kaggle/input/liverpool-ion-switching/')

        train, test, sample_submission = read_data(base)

        logger.info('Reading Data Completed ...')

    train, test = normalize(train, test)    

    with timer(f'Creating Features'):

        logger.info('Feature Enginnering Started ...')

        for config in fe_config:

            train = run_feat_enginnering(train, create_all_data_feats=config[0], batch_size=config[1])

            test  = run_feat_enginnering(test,  create_all_data_feats=config[0], batch_size=config[1])

        train, test, feats = feature_selection(train, test)

        logger.info('Feature Enginnering Completed ...')



    with timer(f'Running NN model'):

        logger.info(f'Training NN model with {SPLITS} folds Started ...')

        run_cv_model_by_batch(train, test, splits=SPLITS, batch_col='group', feats=feats, sample_submission=sample_submission, nn_epochs=EPOCHS, nn_batch_size=NNBATCHSIZE)

        logger.info(f'Training completed ...')

run_everything(fe_config)