import os

import gc

import joblib

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

import xgboost as xgb

import tensorflow as tf

from keras.layers import Dense, Input

from collections import Counter

from keras import layers

from keras.layers import BatchNormalization

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

from keras.layers import Dropout



import warnings

warnings.filterwarnings("ignore")
BATCH_SIZE = 1024

EPOCHS = 50
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def create_model(data, catcols, numcols):    

    inputs = []

    outputs = []

    for c in catcols:

        num_unique_values = int(data[c].nunique())

        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)

        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        inputs.append(inp)

        outputs.append(out)

    

    num_input = layers.Input(shape=(data[numcols].shape[1], ))

    inputs.append(num_input)

    outputs.append(num_input)

    

    x = layers.Concatenate()(outputs)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.3)(x)

    x = BatchNormalization()(x)

    x = Dense(32, activation="relu")(x)

    x = Dropout(0.3)(x)

    x = BatchNormalization()(x)

    y = Dense(1, activation="sigmoid")(x)

    

    model = Model(inputs=inputs, outputs=y)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test["target"] = -1

sample = pd.read_csv('../input/sample_submission.csv')



test_ids = test.id.values

test = test[train.columns]



data = pd.concat([train, test])
catcols = ['wheezy-copper-turtle-magic']

numcols = [c for c in data.columns if c not in ["id", "target"] + catcols]
scl = preprocessing.StandardScaler()

useful_data = data[numcols]

scaled_data = scl.fit_transform(useful_data)

useful_data = pd.DataFrame(scaled_data, columns=useful_data.columns)



useful_data["id"] = data.id.values

useful_data["target"] = data.target.values

for c in catcols:

    if c in ["id", "target"]:

        continue

    useful_data[c] = data[c].values



train = useful_data[useful_data.target != -1].reset_index(drop=True)

test = useful_data[useful_data.target == -1].reset_index(drop=True)



y = train.target.values
clf = create_model(data, catcols, numcols)
clf.fit([train.loc[:, catcols].values[:, k] for k in range(train.loc[:, catcols].values.shape[1])] + [train.loc[:, numcols].values], 

        train.target.values,

        batch_size=BATCH_SIZE,

        epochs=EPOCHS)

test_preds = clf.predict([test.loc[:, catcols].values[:, k] for k in range(test.loc[:, catcols].values.shape[1])] + [test.loc[:, numcols].values])
sample.target = test_preds

sample.to_csv("submission.csv", index=False)