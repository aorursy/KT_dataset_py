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
import numpy as np

import pandas as pd

import seaborn as sns

import os

import random

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import tensorflow as tf

from sklearn.metrics import log_loss

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau
def seeds(seed):

    np.random.seed(seed)

    #tf.random.set_random_seed(seed) # tensorflow v1.14

    tf.random.set_seed(seed) #tensorflow v2.0

    

seeds(42)
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train_features['cp_type'].value_counts()

sns.barplot(train_features['cp_type'].value_counts().index, train_features['cp_type'].value_counts())
train_features['cp_time'].value_counts()

sns.barplot(train_features['cp_time'].value_counts().index, train_features['cp_time'].value_counts())
train_features['cp_dose'].value_counts()

sns.barplot(train_features['cp_dose'].value_counts().index, train_features['cp_dose'].value_counts())
def normalize(df, cols):

    scaler = StandardScaler()

    df[cols] = scaler.fit_transform(df[cols])

    return df

#    test_features[cols] = scaler.fit_transform(test_features[cols])



cols = [x for x in train_features.columns if x not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

train_features = normalize(train_features, cols)

test_features = normalize(test_features, cols)
def ohe(df, cols):

    for i in cols:

        df = df.merge(pd.get_dummies(df[i]),left_index=True,right_index=True)

        df.drop(i, axis=1,inplace=True)

    df.drop('sig_id', axis=1, inplace=True)

    return df



train_features = ohe(train_features, ['cp_type', 'cp_dose', 'cp_time'])

test_features = ohe(test_features, ['cp_type', 'cp_dose', 'cp_time'])



train_targets_scored.drop('sig_id',axis=1,inplace=True)
train_features.head()
def model_struct(num_of_cols):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(1024, input_dim=num_of_cols, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(256, activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(206, activation='sigmoid')

        ])

    

    #RMSprop(lr=0.001)

    

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
submission.loc[:,train_targets_scored.columns] = 0

res = train_targets_scored.copy()

n_loop = 3

for loop in range(n_loop):

    for n, (trn_ind, val_ind) in enumerate(KFold(n_splits=5,shuffle=True,random_state=loop).split(train_features)):

        print("\n")

        print('-'*50)

        print("Loop ", loop, " Fold ", n)

        model = model_struct(train_features.shape[1])

        lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_delta=0.00001)

        model.fit(train_features.values[trn_ind], 

          train_targets_scored.values[trn_ind], 

          epochs=35, 

          batch_size=128, 

          validation_data=(train_features.values[val_ind], train_targets_scored.values[val_ind]), 

          verbose=1,

          callbacks=[lr_loss])

        

        test_pred = model.predict(test_features.values)

        val_pred = model.predict(train_features.values[val_ind])

        

        submission.loc[:,train_targets_scored.columns] += test_pred

        res.loc[val_ind,train_targets_scored.columns] +=val_pred
res.loc[:,train_targets_scored.columns] /= (n_loop*(n+1))

def metric(y_true, y_pred):

    metrics = []

    for _target in train_targets_scored.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

    return np.mean(metrics)



print(f'OOF Metric: {metric(train_targets_scored, res)}')



# 0.008535362743676658 * 0.008531169160687455

# 0.008528504555434469

# 0.008642192757732887
submission.loc[:,train_targets_scored.columns] /= (n_loop*(n+1))

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test_ctl_vehicle_idx = (test['cp_type'] == 'ctl_vehicle')

submission.loc[test_ctl_vehicle_idx, 1:] = 0

submission.loc[test_ctl_vehicle_idx].iloc[:, 1:].sum().sum()
submission.to_csv('submission.csv', index=False)