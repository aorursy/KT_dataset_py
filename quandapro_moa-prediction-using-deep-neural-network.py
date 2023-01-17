import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import random
def set_random_seed():

    random.seed(2021)

    tf.random.set_seed(2020)

    np.random.seed(2019)

set_random_seed()
# CONFIG

model_version = 'v0'

BATCH_SIZE = 128

EPOCHS = 15

SPLITS = 10
TRAIN_FEATURES = pd.read_csv('../input/lish-moa/train_features.csv')

TRAIN_FEATURES.describe()
TEST_FEATURES = pd.read_csv('../input/lish-moa/test_features.csv')

TEST_FEATURES.describe()
# cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle)

train_cp_type = np.unique(TRAIN_FEATURES['cp_type'])

print("Train cp types:", train_cp_type)



# cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low)

train_cp_dose = np.unique(TRAIN_FEATURES['cp_dose'])

print("Train cp_dose:", train_cp_dose)
# Check for column correlation

# TRAIN_FEATURES.corr(method='pearson')
TRAIN_FEATURES = pd.get_dummies(TRAIN_FEATURES, columns=['cp_type', 'cp_dose'])

TRAIN_FEATURES.head()
TEST_FEATURES = pd.get_dummies(TEST_FEATURES, columns=['cp_type', 'cp_dose'])

TEST_FEATURES.head()
MEAN_STD = {}

training_features = TRAIN_FEATURES.columns.tolist()

training_features.remove('sig_id')

for column in training_features:

    # Skip categorical column

    if len(np.unique(TRAIN_FEATURES[column])) == 2 or column == 'sig_id':

        print("Skip categorical column: ", column)

        continue

    # Standardize continous column

    (mu, sigma) = TRAIN_FEATURES[column].mean(), TRAIN_FEATURES[column].std()

    TRAIN_FEATURES[column] = (TRAIN_FEATURES[column] - mu) / sigma

    TEST_FEATURES[column] = (TEST_FEATURES[column] - mu) / sigma

    MEAN_STD[column] = (mu, sigma)

print(TRAIN_FEATURES.describe())

print(TEST_FEATURES.describe())
TRAIN_TARGETS = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

TRAIN_TARGETS.describe()
targets = TRAIN_TARGETS.columns.tolist()

targets.remove('sig_id')

print("Num of classes: ", len(targets))
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



def dense_layer(x, num_of_nodes=1024, activation='tanh'):

    d = Dense(num_of_nodes)(x)

    b = BatchNormalization()(d)

    a = Activation(activation)(b)

    return a



def build_model():

    inp = Input(shape=(len(training_features),))

    d = dense_layer(inp, 1024, 'tanh')

    d = dense_layer(d, 1024, 'tanh')

    out = Dense(len(targets), activation='sigmoid')(d)

    

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer=Adam(), loss=BinaryCrossentropy())

    

    return model 



model = build_model()

model.summary()
TRAIN_DF = pd.merge(TRAIN_FEATURES, TRAIN_TARGETS, on=['sig_id'])

# TRAIN_DF.d

TRAIN_DF.head()
X, y = TRAIN_DF[training_features].values, TRAIN_DF[targets].values

X = np.asarray(X, dtype='float32')

y = np.asarray(y, dtype='float32')

print("Shape of X training: ", X.shape)

print("Shape of y training: ", y.shape)
from tensorflow.keras.callbacks import *

def get_callback(fold):

    return [

#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5),

        ModelCheckpoint(f'model_{model_version}_{fold}.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

    ]
from sklearn.model_selection import KFold

import tensorflow.keras.backend as K

import gc

kfold = KFold(n_splits=SPLITS)

fold = 0

history = []

for train_idx, test_idx in kfold.split(X, y):

    K.clear_session()

    gc.collect()

    print("FOLD ", fold)

    X_train, y_train = X[train_idx], y[train_idx]

    X_test, y_test = X[test_idx], y[test_idx]

    

    callbacks = get_callback(fold)

    

    model = build_model()

    hist = model.fit(X_train, y_train, 

                     batch_size = BATCH_SIZE,

                     epochs = EPOCHS,

                     validation_data=(X_test, y_test), 

                     callbacks = callbacks)

    

    

    history.append(hist)

    fold += 1

    K.clear_session()

    gc.collect()
best_val_loss = [np.min(hist.history['val_loss']) for hist in history]

print("Best val loss for each fold: ", best_val_loss)

print("OOF val loss: ", np.mean(best_val_loss))
# Plot learning curve for each fold

for fold in range(SPLITS):

    fig, ax = plt.subplots()

    ax.plot(history[fold].history['loss'])

    ax.plot(history[fold].history['val_loss'])

    ax.legend(['train', 'test'], loc='upper left')

plt.show()
models = []

predictions = []

for i in range(SPLITS):

    model = build_model()

    model.load_weights(f'model_{model_version}_{i}.h5')

    models.append(model)

    

X_test = np.asarray(TEST_FEATURES[training_features].values, dtype='float32')

for model in models:

    predictions.append(model.predict(X_test, verbose=1))

    

final_prediction = predictions[0]

for i in range(1, SPLITS):

    final_prediction += predictions[i]

final_prediction /= len(models)
print(final_prediction.shape)
submission_data = {}

submission_data['sig_id'] = TEST_FEATURES.sig_id.values

for i, target in enumerate(targets):

    submission_data[target] = final_prediction[:, i]

submission_csv = pd.DataFrame(data=submission_data)

submission_csv.to_csv('submission.csv', index=False)

submission_csv.head()
print("Done!")