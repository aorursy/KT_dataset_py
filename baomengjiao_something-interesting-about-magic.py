import os

import gc

import pandas as pd

import numpy as np



from sklearn.model_selection import StratifiedKFold,KFold

from sklearn import metrics, preprocessing

from sklearn.preprocessing import OneHotEncoder



import tensorflow as tf

from keras.layers import Dense, Input

from collections import Counter

from keras.layers import BatchNormalization

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

from keras.layers import Dropout



import warnings

warnings.filterwarnings("ignore")
def submit(predictions):

    submit = pd.read_csv('../input/sample_submission.csv')

    submit["target"] = predictions

    submit.to_csv("submission.csv", index=False)



def fallback_auc(y_true, y_pred):

    try:

        return metrics.roc_auc_score(y_true, y_pred)

    except:

        return 0.5



def auc(y_true, y_pred):

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
df_tr = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
NFOLDS = 5

RANDOM_STATE = 42

numeric = [c for c in df_tr.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
df_tr.drop(['target'], inplace = True, axis =1)
len_train = df_tr.shape[0]

data = pd.concat([df_tr, df_test])

data = pd.concat([data, pd.get_dummies(data['wheezy-copper-turtle-magic'], prefix = 'magic')], axis=1, sort=False)

df_tr = data[:len_train]

df_test = data[len_train:]
target_column = [i for i in df_tr.columns if i.startswith('magic_')]
folds = KFold(n_splits=NFOLDS)##, shuffle=True, random_state=RANDOM_STATE)
gc.collect()
#y = df_tr[target_column].values

ids = df_tr.id.values

train = df_tr[numeric]#.drop(['id', 'target'], axis=1)

test_ids = df_test.id.values

test = df_test[numeric]

oof_preds = np.zeros((len(train)))

test_preds = np.zeros((len(test)))

scl = preprocessing.StandardScaler()

scl.fit(pd.concat([train, test]))

train = scl.transform(train)

test = scl.transform(test)
for fold_, (trn_, val_) in enumerate(folds.split(ids, ids)):

    print("Current Fold: {}".format(fold_))

    trn_x, trn_y = train[trn_, :], df_tr[target_column].iloc[trn_]

    val_x, val_y = train[val_, :], df_tr[target_column].iloc[val_]

    inp = Input(shape=(trn_x.shape[1],))

    x = Dense(2000, activation="relu")(inp)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    x = Dense(1000, activation="relu")(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    x = Dense(500, activation="relu")(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = Dense(100, activation="relu")(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    out = Dense(512, activation="softmax")(x)

    clf = Model(inputs=inp, outputs=out)

    clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10,

                                 verbose=1, mode='min', baseline=None, restore_best_weights=True)



    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                      patience=3, min_lr=1e-6, mode='min', verbose=1)



    clf.fit(trn_x, trn_y, validation_data=(val_x, val_y), callbacks=[es, rlr], epochs=100, batch_size=1024)

    

    test_fold_preds = clf.predict(test)

    trans_y = [test_fold_preds[i,df_test['wheezy-copper-turtle-magic'][i]] for i in range(len(test_fold_preds))]

    test_preds += np.array(trans_y)/NFOLDS 

    

    K.clear_session()

    gc.collect()
submit(test_preds)
sub = pd.read_csv('submission.csv')
sub.head(10)