# LOAD PACKAGES

import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import tensorflow as tf

from tensorflow.keras import activations, regularizers, Sequential, utils, callbacks, optimizers

from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
# LOAD DATA

BASEPATH = '../input/digit-recognizer/'

def reload(df='train.csv', msg=True, path=BASEPATH):

    o = pd.read_csv(BASEPATH + df)

    print(f'{df} loaded!')

    return o

train = reload()

y_train = train.label.copy()

y_train = utils.to_categorical(y_train, 10)

X_train = train.drop(columns='label')

def preprocess(df):

    df = df / 256.0

    df = df.values.reshape(-1, 28, 28)

    return df

X_train = preprocess(X_train)

test = reload('test.csv')

X_test = preprocess(test)
# PREPROCESS DATA

VAL_SIZE = 0.2

BATCH = 30

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=713)

# Transform data to tf.data format in case we decide to ues TPU later

train_set = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(BATCH)

val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH)

test_set = tf.data.Dataset.from_tensor_slices((X_test,))
# MODEL SETUP

# I've tried my best to labeled parts where I implemented the tricks mentioned

# above, but feel free to drop any questions you may have

L1 = 0

L2 = 0.01

INI_LR = 0.001

EPOCHS = 100

TRAIN_SIZE = 42000

sched_lr_val_acc = False # Because SELU activation doesn't accept Exponential Scheduling



# Regularizers

def get_regularizer(l1=L1, l2=L2):

    return regularizers.l1_l2(l1=l1, l2=l2)



# Learning rate scheduling and Adam optimization technique

def get_optimizer(lr=INI_LR, beta_1=0.9, beta_2=0.999):

    # Learning rate scheduling

    if sched_lr_val_acc == False:

        lr = optimizers.schedules.ExponentialDecay(

            lr

            , decay_steps=500

            , decay_rate=0.96

        )

    # Adam optimizer

    return optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2)



def get_callbacks(monitor='val_accuracy', min_delta=0.001

                  , patience_es=10, patience_lr=5):

    cb = []

    # Early stopping

    cb.append(callbacks.EarlyStopping(

        monitor=monitor, min_delta=min_delta, patience=patience_es

        , restore_best_weights=True

    ))

    # Another learning rate schedule that lowers learning rate

    # using validation set's accuracy instead of automatically

    # reducing learning rate after each step. Try searching

    # ``ReduceLROnPlateau`` for more information

    if sched_lr_val_acc:

        cb.append(callbacks.ReduceLROnPlateau(

            monitor=monitor, factor=0.5, patience=patience_lr

            , min_delta=min_delta, min_lr=INI_LR*0.1**5

        ))

    # This is to shut down model in case of gradient explosion

    cb.append(callbacks.TerminateOnNaN())

    return cb    



def get_model(compiling=True, activation='selu', layers=[300, 100, 50], p=0.2):

    model = Sequential([Flatten(input_shape=(28, 28))

                       , BatchNormalization()

                       , Dropout(rate=p)])

    regu = get_regularizer()

    # Non-saturating activation function: SELU

    if activation == 'selu':

        initializer = 'lecun_normal' # SELU works best with Lecun's normal initialization

    elif activation in ['relu', 'elu']:

        initializer = 'he_normal' # while RELU and ELU works best with He's normal initialization

    else:

        initializer = 'glorot_normal' # Other functions (Tanh, Sigmoid, etc.) works best

                                      # with the default Glorot's normal initialization

    for u in layers:

        model.add(Dense(units = u

                        , use_bias = False

                        , kernel_initializer = initializer

                        , kernel_regularizer = regu

                       ))

        # Batch Normalization

        model.add(BatchNormalization())

        model.add(Activation(activation))

        # Dropout

        model.add(Dropout(rate=p))

    model.add(Dense(10, activation='softmax'))

    if compiling:

        opt = get_optimizer()

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# tf.config.experimental_connect_to_cluster(tpu)

# tf.tpu.experimental.initialize_tpu_system(tpu)

# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# VERBOSE = 1

# with tpu_strategy.scope():

#     NN = get_model(False)



# opt = get_optimizer()

# NN.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
VERBOSE = 1

model = get_model()

cb = get_callbacks()

history = model.fit(train_set, epochs=EPOCHS

                    , validation_data=(X_val, y_val), verbose=VERBOSE

                    , callbacks=cb)
perf = pd.DataFrame(history.history)

plt.style.use('seaborn')

perf.plot(y=['accuracy', 'val_accuracy'], figsize=(10,7))

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')

plt.show()
y_val_pred = model.predict(X_val)

y_val_pred = np.argmax(y_val_pred, axis=1)

y_val_act = np.argmax(y_val, axis=1)

prediction = np.equal(y_val_act, y_val_pred)



def plot_predictions(correct_pred=True, ncol=6, nrow=3):

    plt.figure(figsize=(2*ncol,2.5*nrow))

    index = np.where(prediction == correct_pred)

    for i in np.arange(ncol*nrow):

        plt.subplot(nrow, ncol, i+1)

        plt.imshow(X_val[index][i], interpolation='nearest')

        plt.axis('off')

        title = ''

        if correct_pred == False:

            title =  f'- Actual {y_val_act[index][i]}'

        plt.title(f'Pred {y_val_pred[index][i]}'+title)

    suptitle = 'Accurate ' if correct_pred else 'Inaccurate '

    plt.suptitle(suptitle + 'predictions', size=17)



plot_predictions()
plot_predictions(False)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

submission = reload('sample_submission.csv')

submission['Label'] = y_pred

submission.to_csv('submission.csv', index=False)