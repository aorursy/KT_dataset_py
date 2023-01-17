import itertools



import numpy as np

import pandas as pd



from tqdm import tqdm



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.metrics import MeanSquaredError

from keras.callbacks import EarlyStopping



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense



from matplotlib import pyplot as plt
# Read the data

submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

train_data = pd.read_csv('../input/covid19v7/train_data.csv')

test_data = pd.read_csv('../input/covid19v7/test_data.csv')
train_data.shape, test_data.shape
unused_columns = ['id', 'id_seqpos', 'deg_50C', 'deg_pH10']

train_data = train_data.drop(['id', 'id_seqpos', 'deg_50C', 'deg_pH10'], axis=1)

test_data = test_data.drop(['id', 'id_seqpos'], axis=1)
X_train = train_data.drop(['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'], axis=1)

Y_train = train_data[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']]
# one hot encoding

enc_targets = ['sequence', 'structure', 'predicted_loop_type']

cat_cols = []



for t in enc_targets:

    for c in [c for c in X_train.columns if t in c]:

        cat_cols.append(c)
ohe = OneHotEncoder(handle_unknown='ignore')

ohe.fit(X_train[cat_cols])



X_train = ohe.transform(X_train[cat_cols]).toarray()

test = ohe.transform(test_data[cat_cols]).toarray()
FOLD_N = 5

EPOCHS = 25

kf = KFold(n_splits=FOLD_N)
X_train.shape, Y_train.shape, type(X_train), type(Y_train)
Y_train = Y_train.values.astype(np.float32)
def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true))) / 3



def get_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(383),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1024, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(512, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(92, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(3, activation="relu")

    ])

    model.compile(optimizer='adam', loss='mse', metrics=[root_mean_squared_error])

    return model
early_stopping = EarlyStopping(

    monitor='val_loss',

    min_delta=0,

    patience=3,

    verbose=1000,

    mode='auto'

)
preds = np.zeros((len(test_data), 3))



for n, (train_idx, val_idx) in enumerate(kf.split(X_train)):

    x_train, y_train = X_train[train_idx], Y_train[train_idx]

    x_val, y_val = X_train[val_idx], Y_train[val_idx]

    

    print(f'Training fold #{n}')

    model = get_model()

    results = model.fit(

        x_train,

        y_train,

        epochs=100,

        batch_size=8192,

        validation_data=(x_val, y_val),

        callbacks=[early_stopping]

    )

    

    pred = model.predict(test)

    preds += pred / FOLD_N
preds.shape
submission[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']] = preds

submission
submission.to_csv('submission.csv', index=False)