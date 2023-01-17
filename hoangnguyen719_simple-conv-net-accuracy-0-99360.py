# LOAD PACKAGES

from functools import partial

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.data import Dataset

from tensorflow.keras import (

    activations, regularizers, Sequential

    , utils, callbacks, optimizers

)

from tensorflow.keras.layers import (

    Flatten, Dense, BatchNormalization

    , Activation, Dropout, Conv2D, MaxPool2D

)
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

    df = df.values.reshape(-1, 28, 28, 1)

    return df

X_train = preprocess(X_train)

test = reload('test.csv')

X_test = preprocess(test)
def plot_numbers(x, label, predict=None, correct_pred=True, ncol=6, nrow=3):

    plt.style.use('seaborn')

    plt.figure(figsize=(2*ncol,2.5*nrow))

    if predict is None:

        index = np.random.randint(0, x.shape[0], ncol * nrow)

        plt.suptitle('Random numbers', size=17)

    elif correct_pred:

        index = np.where(label == predict)[0]

        plt.suptitle('Accurate predictions', size=17)

    else:

        index = np.where(label != predict)[0]

        plt.suptitle('Inaccurate predictions', size=17)

    for i in np.arange(min(ncol * nrow, len(index))):

        plt.subplot(nrow, ncol, i+1)

        plt.imshow(x[index[i]], interpolation='nearest')

        plt.axis('off')

        title = f'Label {label[index[i]]}'

        if predict is not None:

            title += f' - Predict {predict[index[i]]}'

        plt.title(title)

    plt.show()



plot_numbers(np.squeeze(X_train), np.argmax(y_train, axis=1))
# Some hyper-parameters

l1 = 0 # l1 regularization parameter

l2 = 0.01 # l1 regularization parameter

ini_lr = 0.001

val_size = 0.3

batch_size = 30

activation = 'relu'

if activation == 'selu':

    initializer = 'lecun_normal' # SELU works best with Lecun's normal initialization

elif activation in ['relu', 'elu']:

    initializer = 'he_normal' # RELU and ELU are best with He's normal initialization

else:

    initializer = 'glorot_normal' # Other functions (Tanh, Sigmoid, etc.) works best

                                  # with the default Glorot's normal initialization

sched_lr_val_acc = True # Whether to schedule learning rate using 

                         # validation accuracy (similar to early stopping) or not

decay_rate = 0.97 # If ``sched_lr_val_acc``=False then use `ExponentialDecay``



# PREPROCESS DATA

X_tr, X_val, y_tr, y_val = train_test_split(

    X_train, y_train, test_size=val_size, random_state=713)

n_train_val_samples = X_tr.shape[0]

n_train_samples = X_tr.shape[0] + X_val.shape[0]

# Transform data to tf.data format in case we decide to ues TPU later

def generate_tfdata(X, y=None, batch_size=batch_size

                    , shuffle_repeat=True):

    '''

        Generate a tfdata dataset where each tensor is a tuple of (input, label).

        If ``shuffle_repeat``=True then the dataset is shuffled

        and repeated indefinitely (should be False for validation/test set).

    '''

    if y is None:

        data = Dataset.from_tensor_slices((X, ))

    else:

        data = Dataset.from_tensor_slices((X, y))

    if shuffle_repeat:

        data = data.shuffle(10000, 713, reshuffle_each_iteration=True).repeat()

    if batch_size:

        data = data.batch(batch_size)

    return data.prefetch(1)

tr_set = generate_tfdata(X_tr, y_tr)

val_set = generate_tfdata(X_val, y_val, shuffle_repeat=False)

train_set = generate_tfdata(X_train, y_train)

test_set = generate_tfdata(X_test, shuffle_repeat=False)
# Set up model

# Regularizers

def get_regularizer(l1=l1, l2=l2):

    return regularizers.l1_l2(l1=l1, l2=l2)



# Learning rate and Optimizer

def get_optimizer(lr=ini_lr, beta_1=0.9, beta_2=0.999

                  , decay_rate=decay_rate

                  , n_samples=n_train_samples, batch_size=batch_size):

    # Learning rate scheduling

    # Only relevant if ``sched_lr_val_acc``=False

    if sched_lr_val_acc == False:

        lr = optimizers.schedules.ExponentialDecay(

            lr

            , decay_steps = n_samples // batch_size // 2

            , decay_rate = decay_rate

        )

    # Adam optimizer

    return optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2)



# Callbacks

def get_callbacks(es_params=['val_loss', 1e-3, 15] # [monitor, min_delta, patience]

                  , lr_params=['val_accuracy', 0.001, 5]): # [monitor, min_delta, patience]

    cb = []

    # Early stopping

    cb.append(callbacks.EarlyStopping(

        monitor=es_params[0], min_delta=es_params[1], patience=es_params[2]

        , verbose=1, restore_best_weights=True

    ))

    # Learning rate scheduling using validation accuracy

    # Only relevant if ``sched_lr_val_acc``=True

    if sched_lr_val_acc:

        cb.append(callbacks.ReduceLROnPlateau(

            monitor=lr_params[0], factor=0.5, patience=lr_params[2]

            , min_delta=lr_params[1], min_lr=ini_lr*0.1**5

        ))

    # Terminate on NaN

    cb.append(callbacks.TerminateOnNaN())

    return cb



DefaultConv2D = partial(Conv2D

                        , kernel_size=3

                        , strides=1

                        , activation=activation

                        , kernel_initializer=initializer

                        , padding='SAME')

DefaultMaxPool2D = partial(MaxPool2D

                           , pool_size=2

                           , strides=2)

DefaultDense = partial(Dense

                       , activation=activation

                       , kernel_initializer=initializer

                       , kernel_regularizer=get_regularizer())



def get_model(compiling=True, n_samples=n_train_samples, batch_size=batch_size

              , activation=activation, initializer=initializer, p=0.5):



    model = Sequential(

        [DefaultConv2D(filters=32, kernel_size=5, input_shape=(28,28,1))

         , BatchNormalization()

         , DefaultMaxPool2D()

         , DefaultConv2D(filters=64)

         , BatchNormalization()

         , DefaultConv2D(filters=64)

         , BatchNormalization()

         , DefaultMaxPool2D()

         , DefaultConv2D(filters=128)

         , BatchNormalization()

         , DefaultConv2D(filters=128)

         , BatchNormalization()

         , DefaultMaxPool2D()

         , Flatten()

         , DefaultDense(units=64)

         , Dropout(p)

         , DefaultDense(units=32)

         , Dropout(p)

         , Dense(units=10, activation='softmax')

        ]

    )

    optimizer = get_optimizer(n_samples=n_samples, batch_size=batch_size)

    if compiling:

        opt = get_optimizer()

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
verbose = 1

# Train model

model = get_model(n_samples=n_train_val_samples)

print(model.summary())

epochs = 100

cb = get_callbacks()

history = model.fit(tr_set, epochs = epochs

                    , steps_per_epoch = n_train_val_samples // batch_size

                    , validation_data = val_set

                    , callbacks = cb

                    , verbose = verbose)
def plot_performance(history):

    perf = pd.DataFrame(history.history)

    plt.style.use('seaborn')

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 14))

    for ax, (metric, val_metric, name) in zip(

        [ax1, ax2]

        , [[perf.loss, perf.val_loss, 'Loss'], [perf.accuracy, perf.val_accuracy, 'Accuracy']]

    ):

        ax.plot(perf.index, metric, label='train')

        ax.plot(perf.index, val_metric, label='validation')

        ax.set_ylabel(name)

        ax.set_title('Train and Validation '+name, size=16)

    handles, labels = ax1.get_legend_handles_labels()

    ax2.set_xlabel('Epoch')

    ax1.legend(loc='upper right')

    plt.show()

plot_performance(history)
X_plot = X_val[:5000]

y_plot = y_val[:5000]

y_pred = model.predict(X_plot)

y_pred = np.argmax(y_pred, axis=1)

y_actual = np.argmax(y_plot, axis=1)



plot_numbers(np.squeeze(X_plot), y_actual, y_pred)
plot_numbers(np.squeeze(X_plot), y_actual, y_pred, False)
# Output

y_pred = model.predict(test_set)

y_pred = np.argmax(y_pred, axis=1)

submission = reload('sample_submission.csv')

submission['Label'] = y_pred

submission.to_csv('submission.csv', index=False)
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# tf.config.experimental_connect_to_cluster(tpu)

# tf.tpu.experimental.initialize_tpu_system(tpu)

# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# VERBOSE = 1

# with tpu_strategy.scope():

#     NN = get_model(False)



# opt = get_optimizer()

# NN.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])