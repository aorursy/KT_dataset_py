import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

import random

import os

from tensorflow.keras.callbacks import ReduceLROnPlateau
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
COLS = ['cp_type','cp_dose']

FE = []

for col in COLS:

    for mod in train_features[col].unique():

        FE.append(mod)

        train_features[mod] = (train_features[col] == mod).astype(int)

del train_features['sig_id']

del train_features['cp_type']

del train_features['cp_dose']

FE+=list(train_features.columns) 

del train_targets['sig_id']
def model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(877),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(4096, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(4096, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(206, activation="sigmoid")

        ])



    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2.75e-5), loss='binary_crossentropy', metrics=["accuracy", "AUC"])

    return model

# model.fit(train_dataset, epochs=10, batch_size=128)
from sklearn.model_selection import KFold

NFOLD = 5

kf = KFold(n_splits=NFOLD)



BATCH_SIZE=128

EPOCHS=30



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

for col in COLS:

    for mod in test_features[col].unique():

        test_features[mod] = (test_features[col] == mod).astype(int)

sig_id = pd.DataFrame()

sig_id = test_features.pop('sig_id')

del test_features['cp_type']

del test_features['cp_dose']



pe = np.zeros((test_features.shape[0], 206))



train_features = train_features.values

train_targets = train_targets.values

pred = np.zeros((train_features.shape[0], 206))



cnt=0

for tr_idx, val_idx in kf.split(train_features):

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    cnt += 1

    print(f"FOLD {cnt}")

    net = model()

    net.fit(train_features[tr_idx], train_targets[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 

            validation_data=(train_features[val_idx], train_targets[val_idx]), verbose=0, callbacks=[reduce_lr_loss])

    print("train", net.evaluate(train_features[tr_idx], train_targets[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", net.evaluate(train_features[val_idx], train_targets[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")

    pred[val_idx] = net.predict(train_features[val_idx], batch_size=BATCH_SIZE, verbose=0)

    print("predict test...")

    pe += net.predict(test_features, batch_size=BATCH_SIZE, verbose=0) / NFOLD
pe.shape



columns = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

del columns['sig_id']

sub = pd.DataFrame(data=pe, columns=columns.columns)
sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

sub.insert(0, column = 'sig_id', value=sample['sig_id'])
sub.to_csv('submission.csv', index=False)
# Make sure all the columns between our submission and the sample submission are the same

def Diff(list1, list2): 

    return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1)))) 



Diff (sub.columns, pd.read_csv('../input/lish-moa/sample_submission.csv').columns)