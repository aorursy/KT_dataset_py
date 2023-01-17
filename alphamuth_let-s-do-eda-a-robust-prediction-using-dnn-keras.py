# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import os

import gc

import random

import math

import time



import numpy as np

import pandas as pd



from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.metrics import log_loss



import category_encoders as ce



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

import torch.optim as optim

import torch.nn.functional as F



import warnings

#warnings.filterwarnings("ignore")
sample_submission=pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_targets_scored=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored=pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_features=pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features=pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_features.head()
train_features.shape
train_features.sig_id.nunique()
train_targets_scored.head()
train_targets_scored.sum()[1:].sort_values()
g = train_features[:1][[col for col in train_features if 'g-' in col]]

g
g= g.values.reshape(-1,1)

plt.plot(g)
print("Number of features (g-) = ",train_features.columns.str.startswith('g-').sum())
c = train_features[:1][[col for col in train_features if 'c-' in col]]

c
c=c.values.reshape(-1,1)

plt.plot(c)
print("Number of features (c-) = ",train_features.columns.str.startswith('c-').sum())
plt.figure(figsize=(5,5))

plt.subplot(1, 2, 1)

x_value = train_features.cp_type.value_counts()

plt.title("Training data")

plt.bar(x_value.index,x_value.values)
sns.heatmap(train_features.loc[:, ['g-0', 'g-1', 'g-2','g-3','g-4', 'c-95', 'c-96', 'c-97','c-98', 'c-99']].corr(), annot=True)

plt.show()
train_targets_scored.head(10)
train_targets_nonscored.head(10)
# for g- feature columns

g = [col for col in train_features if 'g-' in col]

g = sns.pairplot(train_features[g[:10]])

plt.show()
# for c- feature columns

g_vis = sns.pairplot(train_features[[col for col in train_features if 'c-' in col][:10]])

plt.show()
import numpy as np 

import pandas as pd 

import os



from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

 

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M
SEED = 123

EPOCHS = 49

BATCH_SIZE = 256

FOLDS =5

REPEATS = 5

target_cols = train_targets_scored.columns[1:]

input_shape = len(target_cols)
def seed(seed):

    np.random.seed(seed)

    os.environ['seed'] = str(seed)

    tf.random.set_seed(seed)

def multi_log_loss(y_true, y_pred):

    losses = []

    for col in y_true.columns:

        losses.append(log_loss(y_true.loc[:, col], y_pred.loc[:, col]))

    return np.mean(losses)
def preprocess_input(data):

    data['cp_type'] = (data['cp_type'] == 'trt_cp').astype(int)

    data['cp_dose'] = (data['cp_dose'] == 'D2').astype(int)

    return data

x_train = preprocess_input(train_features.drop(columns="sig_id"))

x_test =preprocess_input(test_features.drop(columns="sig_id"))

y_train = train_targets_scored.drop(columns="sig_id")

N_FEATURES = x_train.shape[1]
def dnn_model():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(N_FEATURES),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),  

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(input_shape, activation="sigmoid"))

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=["accuracy"])

    model.summary()

    return model



print(x_train.shape)

print(y_train.shape)
def build_train(resume_models = None, repeat_number = 0, folds = 5, skip_folds = 0):

    

    models = []

    preds_df = y_train.copy()

    



    kfold = KFold(folds, shuffle = True)

    for fold, (train_ind, val_ind) in enumerate(kfold.split(x_train)):

        print('\n')

        print('-'*50)

        print(f'Training fold {fold + 1}')

        

        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'binary_crossentropy', factor = 0.4, patience = 2, verbose = 1, min_delta = 0.0001, mode = 'auto')

        checkpoint_path = f'repeat:{repeat_number}_Fold:{fold}.hdf5'

        cb_checkpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = True, mode = 'min')



        model = dnn_model()

        model.fit(x_train.values[train_ind],

              y_train.values[train_ind],

              validation_data=(x_train.values[val_ind], y_train.values[val_ind]),

              callbacks = [cb_lr_schedule, cb_checkpt],

              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1

             )

        model.load_weights(checkpoint_path)

        preds_df.loc[val_ind, :] = model.predict(x_train.values[val_ind])

        models.append(model)



    return models, preds_df
models = []

preds_df = []

# seed everything

seed(SEED)

for i in range(REPEATS):

    m, f = build_train(repeat_number = i, folds=FOLDS)

    models = models + m

    preds_df.append(f)
mean_oof_preds = y_train.copy()

mean_oof_preds.loc[:, target_cols] = 0

for i, p in enumerate(preds_df):

    print(f"Iterate{i + 1} Log Loss: {multi_log_loss(y_train, p)}")

    mean_oof_preds.loc[:, target_cols] += p[target_cols]



mean_oof_preds.loc[:, target_cols] /= len(preds_df)

print(f"Mean Log Loss: {multi_log_loss(y_train, mean_oof_preds)}")

mean_oof_preds.loc[x_train['cp_type'] == 0, target_cols] = 0

print(f"Mean Log Loss: {multi_log_loss(y_train, mean_oof_preds)}")
test_preds = sample_submission.copy()

test_preds[target_cols] = 0

for model in models:

    test_preds.loc[:,target_cols] += model.predict(x_test)

test_preds.loc[:,target_cols] /= len(models)

test_preds.loc[x_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)