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

import tensorflow

np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input



from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa

from sklearn.metrics import log_loss

import tensorflow as tf
data_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

data_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

data_train_target_ns = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

data_train_target_s = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 0.5, 72:1})

    del df['sig_id']

    return df



train = preprocess(data_train)

test = preprocess(data_test)



del data_train_target_s['sig_id']

def create_model(num_columns):

    model = Sequential()

    model.add(Input(num_columns))

    model.add(BatchNormalization())

    model.add(Dense(8912, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(4096, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(2048, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    

    model.add(Dense(1024, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(206, activation='sigmoid'))

    

    optimizer = tfa.optimizers.Lookahead('adam',sync_period=10)

    

    model.compile(optimizer=optimizer,

                  loss='binary_crossentropy', 

                  metrics=['accuracy'])

    

    model.summary()

    return model




def metric(y_true, y_pred):

    metrics = []

    for _target in data_train_target_s.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

    return np.mean(metrics)



N_STARTS = 4

tf.random.set_seed(42)



res = data_train_target_s.copy()

sub.loc[:, data_train_target_s.columns] = 0

sub.loc[:, data_train_target_s.columns] = 0



for seed in range(N_STARTS):

    for n, (train_idx, test_idx) in enumerate(KFold(n_splits=5, random_state=seed, shuffle=True).split(data_train_target_s, data_train_target_s)):

        print(f'Fold {n}')

    

        model = create_model(875)

#         checkpoint_path = f'repeat:{seed}_Fold:{n}.h5'

#         reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

#         cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

#                                      save_weights_only = True, mode = 'min')

        model.fit(train.values[train_idx],

                  data_train_target_s.values[train_idx],

                  validation_data=(train.values[test_idx], data_train_target_s.values[test_idx]),

                  epochs=25, batch_size=128,verbose=1

#                   callbacks=[reduce_lr_loss, cb_checkpt],

                 )

        

#         model.load_weights(checkpoint_path)

        test_predict = model.predict(test.values)

        val_predict = model.predict(train.values[test_idx])

        

        sub.loc[:, data_train_target_s.columns] += test_predict

        res.loc[test_idx, data_train_target_s.columns] += val_predict

        print('')

    

sub.loc[:, data_train_target_s.columns] /= ((n+1) * N_STARTS)

res.loc[:, data_train_target_s.columns] /= N_STARTS
sub.loc[test['cp_type']==1, data_train_target_s.columns] = 0
sub.to_csv('submission.csv', index=False)