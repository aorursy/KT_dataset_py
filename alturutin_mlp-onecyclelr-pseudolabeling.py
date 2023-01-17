import sys
sys.path.append("../input/keras-one-cycle-lr/")
sys.path.append("../input/keras-one-cycle-lr/one_cycle_lr")

!pip install ../input/keras-one-cycle-lr
sys.path.append('../input/moa-scripts')
from moa import load_datasets, preprocess, split, submit, submit_preds
from metrics import logloss
from oof import OOFTrainer
from ml_stratifiers import MultilabelStratifiedKFold

# np, pd
import numpy as np 
import pandas as pd 

# misc
import warnings
warnings.simplefilter('ignore')
from glob import glob
from tqdm.auto import tqdm
import os
import random
import copy
import joblib
import gc 
from functools import partial

# viz
import matplotlib.pyplot as plt
import seaborn as sns

# tf
import tensorflow as tf

# ml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# seed before keras import
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

seed_everything()

# import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from one_cycle_lr import one_cycle_scheduler

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# # if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
train, target, y_nonscored, test, submission = load_datasets("../input/lish-moa")
X, y, test, test_control = preprocess(train, target, test, standard=False, onehot=True)
X, y, X_holdout, y_holdout, split_index, index, classnames, features = split(X, y, drop_rare_labels=False)
genes = [i for i,col in enumerate(features) if col.startswith('g-')]
cells = [i for i,col in enumerate(features) if col.startswith('c-')]
# PCA
n = len(X)
c1, c2 = 64, 16
pca1,pca2 = PCA(c1), PCA(c2)
pca1_res  = pca1.fit_transform(np.concatenate([X, test]))
pca2_res  = pca2.fit_transform(np.concatenate([X, test]))
X = np.concatenate([X, pca1_res[:n], pca2_res[:n]], axis=1)
test = np.concatenate([test, pca1_res[n:], pca2_res[n:]], axis=1)
features = list(features)
features += [f'pca_genes_{i}' for i in range(c1)] + [f'pca_cells_{i}' for i in range(c2)]
assert len(features) == X.shape[1] == test.shape[1]
# variance feature selection
from sklearn.feature_selection import VarianceThreshold
var = VarianceThreshold(threshold=0.5)
var.fit(np.concatenate([X, test])[:, 4:])
mask = np.concatenate([[True]*4, var.get_support()])
X = X[:, mask]
test = test[:, mask]

def weighted_logloss_keras(w=0.5):
    alpha = K.variable(w)
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        return - 2 * K.mean( ( alpha * y_true * K.log(y_pred) + (1 - alpha) * (1 - y_true) * (K.log(1 - y_pred))) )
    return loss

def logloss_keras(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return - K.mean( y_true * K.log(y_pred) + (1 - y_true) * (K.log(1 - y_pred)) )

max_norm = tf.keras.constraints.max_norm
he_uniform = lambda scale, seed: tf.keras.initializers.VarianceScaling(mode='fan_in', distribution='uniform', scale=scale, seed=seed)
constant = lambda x: tf.keras.initializers.Constant(value=x)
l2_reg = lambda x: tf.keras.regularizers.l2(x)

class DenseBlock(L.Layer):
    def __init__(self, 
                 dropout_rate=0.5, 
                 dense=(1024, 6.0, 1e-5, None), 
                 prelu=(0.25, 0.0, None), 
                 clipnorm=None,
                 seed=42):
        super(DenseBlock, self).__init__()
        
        self.batch_norm = L.BatchNormalization()
        self.dropout = L.Dropout(dropout_rate)
        self.dense = L.Dense(
            dense[0], activation=None, 
            kernel_initializer=he_uniform(dense[1], seed), 
            kernel_regularizer=l2_reg(dense[2]), 
            kernel_constraint =max_norm(dense[3]) if dense[3] else None
            )
        # self.activation = L.PReLU(
        #     alpha_initializer=constant(prelu[0]),
        #     alpha_regularizer=l2_reg(prelu[1]), 
        #     alpha_constraint =max_norm(prelu[2]) if prelu[2] else None
        #     )
        self.activation = L.ReLU()
        self.clipnorm = tf.keras.constraints.MaxNorm(clipnorm) if clipnorm else None
        
    def call(self, inputs):
        x = self.batch_norm(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        if self.clipnorm:
            x = self.clipnorm(x)
        return x

def create_model(num_columns, 
                 
                 dropout=(0.25, 0.5), 
                 hidden=(1024, 1024), 
                 he_scale=6.0, 
                 kernel=(1e-5, None),
                 activation=(0.25, 0.0, None),

                 top_layer_bias=1.0,
                 clipnorm=None,

                 learning_rate=1e-3, 
                 weight_decay=1e-5, 
                 lookahead_sync=7,
                 label_smoothing=1e-6,

                 l2top=1,

                 seed=42, 
                 metrics=[]):
    
    # input 
    model = tf.keras.Sequential()
    model.add(L.Input(num_columns))
    
    # dense blocks
    for i in range(len(hidden)):
        d = dropout[0] if i == 0 else dropout[1]
        model.add(DenseBlock
                  (dropout_rate=d, 
                   dense=(hidden[i], he_scale, kernel[0], kernel[1]), 
                   prelu=activation,
                   clipnorm=clipnorm,
                   seed=seed
                   )
                  )
        
    # top layer
    model.add(L.BatchNormalization())
    model.add(L.Dropout(dropout[1]))
    model.add(tfa.layers.WeightNormalization(L.Dense(
        206, activation="sigmoid", 
        kernel_initializer=he_uniform(he_scale, seed),
        bias_initializer=constant(top_layer_bias),
        kernel_regularizer=l2_reg(kernel[0]) if l2top else None
        )))
    
    # compile
    model.compile(
        optimizer=tfa.optimizers.Lookahead(tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay), sync_period=lookahead_sync), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing), 
        metrics=metrics
        )
    return model


model_params = dict(
    dropout         =(0.05, 0.65), 
    hidden          =(1024, 1024),
    kernel          =(3e-6, None),
    activation      =(0.05, 6e-6, 4),
    he_scale        =6,
    clipnorm        =None,
    label_smoothing =0,
    weight_decay    =1e-5,
    l2top           =1,
    top_layer_bias  =1.0,
    learning_rate   =5e-4, 
    lookahead_sync  =7,
    seed=42
    # metrics=[logloss_keras]
)
scheduler_params = dict(
    max_lr=1e-2, 
    pct_start=0.2, 
    start_div=1e4, 
    end_div=1e5
)
fit_params = dict(
    epochs=25, 
    batch_size=96, 
    verbose=0
)
%%time 

def crossval(X, y, X_test, 
             model_params, scheduler_params, fit_params, 
             splits=[(MultilabelStratifiedKFold, 5)], 
             seeds=[42], 
             fold_mask=None,
             verbose=1, 
             pseudo=-1
            ):
    
    oof = np.zeros((len(X_test), y.shape[1]), dtype=np.float64)
    preds = np.zeros_like(y, dtype=np.float64)
    history = []
    for seed in seeds:
        seed_everything(seed)
        model_params['seed'] = seed
        
        # split cv
        for kf, n_splits in splits:
            cv = kf(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
                if fold_mask and fold not in fold_mask:
                    continue
                iter_id = f'{str(kf).split(".")[-1][:-2]}-seed{seed}-fold{fold+1}'
                if verbose: print(iter_id, end=': ')
                X_train, y_train, X_valid, y_valid = X[train_index].copy(), y[train_index].copy(), X[valid_index].copy(), y[valid_index].copy()

                # fit model
                model = create_model(X.shape[1], **model_params)
                checkpoint_path = f'{iter_id}.h5'
                cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
                ocp = one_cycle_scheduler.OneCycleScheduler(verbose=0, **scheduler_params)
                model.fit(X_train, y_train, 
                          validation_data=(X_valid, y_valid),
                          callbacks=[cb_checkpt, ocp], 
                          **fit_params)
                model.load_weights(checkpoint_path)

                # pseudo labels
                if pseudo > 0:
                    if verbose: print(f'{logloss_keras(y_valid, model.predict(X_valid)):.5f}', end=' ')
                    pseudo_preds = model.predict(X_test)[~test_control]
                    fit_params_p = fit_params.copy(); fit_params_p['epochs'] = pseudo
                    model.fit(X_test[~test_control], pseudo_preds, 
                              validation_data=(X_valid, y_valid),
                              callbacks=[cb_checkpt], **fit_params_p)
                    model.load_weights(checkpoint_path)
                
                # predict
                fold_preds = model.predict(X_valid)
                preds[valid_index] += fold_preds
                oof += model.predict(X_test)
                val_loss = logloss_keras(y_valid, fold_preds)
                history.append(val_loss)
                if verbose: print(f'{val_loss:.5f}')
                
    tries = len(seeds) * len(splits)
    folds = len(seeds) * sum([n for _, n in splits])
    preds /= tries
    oof /= folds
    return preds, oof, np.array(history)

# ------------------------------------------------------------------------------------------
pseudo_epochs = 5
splits = [(MultilabelStratifiedKFold, 6)]
seeds = range(5)
# seeds=[0]
fold_mask = None
preds, oof, history = crossval(X, y, test, model_params, scheduler_params, fit_params=fit_params, splits=splits, seeds=seeds, fold_mask=fold_mask, pseudo=pseudo_epochs)
print(history.mean(), history.std())
joblib.dump(preds, 'train.pkl')
joblib.dump(oof, 'oof.pkl')
submit_preds(oof, submission, test_control, classnames)
pd.read_csv('submission.csv').iloc[:5, :5]
