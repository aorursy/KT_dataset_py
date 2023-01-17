import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']
from sklearn.model_selection import KFold

from sklearn.metrics import log_loss
def create_model():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(len(list(train_features.columns))),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(2048, activation="relu"),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2048, activation="relu"),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(206, activation="sigmoid")

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy","binary_crossentropy"])

    return model
ss.loc[:, train_targets.columns] = 0

res = train_targets.copy()

for n, (tr, te) in enumerate(KFold(n_splits=5, random_state=0, shuffle=True).split(train_targets)):

    print(f'Fold {n}')

    

    model = create_model()

    

    model.fit(train.values[tr],

              train_targets.values[tr],

              epochs=13, batch_size=128,

             )

    

    ss.loc[:, train_targets.columns] += model.predict(test_features)

    res.loc[te, train_targets.columns] = model.predict(train.values[te])

    print('')

    

ss.loc[:, train_targets.columns] /= (n+1)



metrics = []

for _target in train_targets.columns:

    metrics.append(log_loss(train_targets.loc[:, _target], res.loc[:, _target]))

print(f'OOF Metric: {np.mean(metrics)}')
ss.to_csv('submission.csv', index=False)