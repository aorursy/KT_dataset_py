import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss



import tensorflow_addons as tfa



np.random.seed(666)
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72:2})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)

del train_targets['sig_id']
train_targets['cp_type'] = train['cp_type']
train = train[train['cp_type'] != 'ctl_vehicle']

train_targets = train_targets[train_targets['cp_type'] != 'ctl_vehicle']

train = train.drop(['cp_type'], axis=1)

train_targets = train_targets.drop(['cp_type'], axis=1)
train = train.reset_index().drop(['index'], axis=1)

train_targets = train_targets.reset_index().drop(['index'], axis=1)
train_categories = train[['cp_dose', 'cp_time']]

test_categories = test[['cp_dose', 'cp_time']]
test_cp_type = test['cp_type']

test = test.drop(['cp_type'], axis=1)
train
def create_autoencoder():

    input_vector = Input(shape=(874,))

    encoded = Dense(3000, activation='elu')(input_vector)

    encoded = Dense(2000, activation='elu')(encoded)

    decoded = Dense(3000, activation='elu')(encoded)

    decoded = Dense(874, activation='elu')(decoded)

    

    autoencoder = tf.keras.Model(input_vector, decoded)

    autoencoder.compile(optimizer='adadelta', loss='mse')

    

    return autoencoder
autoencoder = create_autoencoder()
mu, sigma = 0, 0.1



noise = np.random.normal(mu, sigma, [21948,874]) 

noised_train = train + noise
autoencoder.fit(noised_train, train,

                epochs=4000,

                batch_size=128,

                shuffle=True,

                validation_split=0.2)
encoder = tf.keras.Model(autoencoder.input, autoencoder.layers[2].output)
train_features = pd.DataFrame(encoder.predict(train))

test_features = pd.DataFrame(encoder.predict(test))
train_features
def create_model():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(2000),

    tf.keras.layers.BatchNormalization(),



    tfa.layers.WeightNormalization(tf.keras.layers.Dense(500)),

    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

        

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(500)),

    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

        

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(256)),

    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

        

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation="sigmoid"))

    ])

    model.compile(optimizer=tfa.optimizers.AdamW(lr = 1e-3, weight_decay = 1e-5, clipvalue = 700), loss='binary_crossentropy')

    return model
submission.loc[:, train_targets.columns] = 0

res = train_targets.copy()

for n, (tr, te) in enumerate(KFold(n_splits=7, random_state=666, shuffle=True).split(train_targets)):

    print(f'Fold {n}')

    

    model = create_model()

    

    model.fit(

        train_features.values[tr],

        train_targets.values[tr],

        epochs=50, 

        batch_size=128

    )

    

    submission.loc[:, train_targets.columns] += model.predict(test_features)

    res.loc[te, train_targets.columns] = model.predict(train_features.values[te])

    

submission.loc[:, train_targets.columns] /= (n+1)



metrics = []

for _target in train_targets.columns:

    metrics.append(log_loss(train_targets.loc[:, _target], res.loc[:, _target]))
print(f'OOF Metric: {np.mean(metrics)}')
submission['cp_type'] = test_cp_type

for col in submission.columns:

    if col in ['sig_id', 'cp_type', 'cp_dose', 'cp_time']:

        continue

    submission.loc[submission['cp_type'] == 'ctl_vehicle', col] = 0



submission = submission.drop(['cp_type'], axis=1)
submission.to_csv('submission.csv', index=False)