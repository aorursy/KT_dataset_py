# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
test["target"] = -1

data = pd.concat([train, test]).reset_index(drop=True)
features = [x for x in data.columns if x not in ["id", "target"]]

feature_map = {}

cnt = 0

for feature in features:

    feature_unique = data[feature].unique()

    d = dict(zip(feature_unique, range(cnt, len(feature_unique) + cnt)))

    cnt += len(feature_unique)

    feature_map[feature] = d
print(cnt)
for feature in features:

    train[feature] = train[feature].map(feature_map[feature])

    test[feature] = test[feature].map(feature_map[feature])
train_data = train[features]

train_label = train[['target']]

test_data = test[features]

print("train_data.shape = ", train_data.shape)

print("train_label.shape = ", train_label.shape)

print("test_data.shape = ", test_data.shape)
import os

import gc

import joblib

import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

from keras import layers

from keras import optimizers

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

from keras import utils

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.layers import Flatten, Input, Embedding, BatchNormalization

import warnings

warnings.filterwarnings("ignore")
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def create_model(): 

    inp = layers.Input(shape=(23,))

    x = layers.Embedding(5725, 10)(inp)

    x = layers.Flatten()(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(500, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(300, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.BatchNormalization()(x)

    

    y = layers.Dense(2, activation="softmax")(x)



    model = Model(inputs=inp, outputs=y)

    return model
model = create_model()
model.summary()
oof_preds = np.zeros((len(train_data)))

test_preds = np.zeros((len(test_data)))
%%time

skf = StratifiedKFold(n_splits=10)

k = 0

for train_index, test_index in skf.split(train_data, train_label):

    x_train, x_test = train_data.iloc[train_index, :], train_data.iloc[test_index, :]

    y_train, y_test = train_label.iloc[train_index, :], train_label.iloc[test_index, :]

    model = create_model()

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[auc])

    lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.1,

                                     patience=3, min_lr=1e-6, mode='max', verbose=1)

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,

                                 verbose=1, mode='max', baseline=None, restore_best_weights=True)

    model.fit(x_train,

              utils.to_categorical(y_train),

              validation_data=(x_test, utils.to_categorical(y_test)),

              verbose=1,

              batch_size=1024,

              callbacks=[lr, es],

              epochs=20

             )

    valid_fold_preds = model.predict(x_test)[:, 1]

    test_fold_preds = model.predict(test_data)[:, 1]

    oof_preds[test_index] = valid_fold_preds.ravel()

    test_preds += test_fold_preds.ravel()

    print("KFold %d, the best auc is %.4f" % (k, metrics.roc_auc_score(y_test, valid_fold_preds)))

    k += 1

    K.clear_session()
test_preds /= 10

test_ids = test.id.values

print("Saving submission file")

submission = pd.DataFrame.from_dict({

    'id': test_ids,

    'target': test_preds

})

submission.to_csv("submission.csv", index=False)