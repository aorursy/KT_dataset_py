import numpy as np

import pandas as pd

import tensorflow as tf
classes = 10

rows, cols, channel = 28, 28, 1

input_shape = (rows, cols, channel)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0



x_train = x_train.reshape(x_train.shape[0], *input_shape)

x_test = x_test.reshape(x_test.shape[0], *input_shape)
from tensorflow.keras import Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.models import Sequential
class LeNet(Model):

    def __init__(self, classes):

        super(LeNet, self).__init__()

        self.conv1 = Conv2D(6, kernel_size = (5, 5), padding = 'SAME', activation='relu')

        self.conv2 = Conv2D(16, kernel_size = (5,5), activation = 'relu')

        self.pool = MaxPooling2D(pool_size = (2,2))

        self.flatten = Flatten()

        self.dense1 = Dense(120, activation = 'relu')

        self.dense2 = Dense(84, activation = 'relu')

        self.dense3 = Dense(classes, activation='softmax')

    

    def call(self, inputs):

        x = self.pool(self.conv1(inputs))

        x = self.pool(self.conv2(x))

        x = self.flatten(x)

        x = self.dense3(self.dense2(self.dense1(x)))

        return x
model = LeNet(classes)

model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])




# We can call `model.summary()` only if the model was built before. 

# It is normally done automatically at the first use of the network,

# inferring the input shapes from the samples the network is given.

# For instance, the command below would build the network (then use it for prediction):

_ = model.predict(x_test[:10])



# Method to visualize the architecture of the network:

model.summary()
callbacks = [

    # Callback to interrupt the training if the validation loss (`val_loss`) stops improving for over 3 epochs:

    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),

    # Callback to log the graph, losses and metrics into TensorBoard (saving log files in `./logs` directory):

    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)]
history = model.fit(x_train, y_train,

                    batch_size=32, epochs=80, validation_data=(x_test, y_test), 

                    verbose=1,

                    callbacks=callbacks)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('MODEL ACCURACY')

plt.ylabel('Accuracy')

plt.xlabel('No. of epochs')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
def lenet(name = 'lenet'):

    model = Sequential(name = name)

    model.add(Conv2D(6, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = input_shape))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(16, kernel_size = (5,5), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    

    model.add(Flatten())

    model.add(Dense(120, activation = 'relu'))

    model.add(Dense(84, activation = 'relu'))

    model.add(Dense(classes, activation = 'softmax'))

    return model
from tensorflow.keras import optimizers



# Setting some variables to format the logs:

log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\n\033[94m', '\033[92m'

log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'

log_end_format = '\033[0m'
optimizers_list = {

    'sgd' : optimizers.SGD(),

    'momentum' : optimizers.SGD(momentum=0.9),

    'nag': optimizers.SGD(momentum=0.9, nesterov=True),

    'adagrad': optimizers.Adagrad(),

    'adadelta': optimizers.Adadelta(),

    'rmsprop': optimizers.RMSprop(),

    'adam': optimizers.Adam()

}
history_per_opti = dict()
print("Experiment: {0}start{1} (training logs = off)".format(log_begin_red, log_end_format))

for opti in optimizers_list:

    tf.random.set_seed (42)

    np.random.seed(42)

    

    model = lenet('lenet_{}'.format(opti))

    optimizer = optimizers_list[opti]

    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    

    print("\t> Training with {0}: {1}start{2}".format(

        opti, log_begin_red, log_end_format))

    history = model.fit(x_train, y_train, batch_size = 32, 

                       epochs = 20, validation_data = (x_test, y_test),

                       verbose = 1)

    history_per_opti[opti] = history

    print('\t> Training with {0}: {1}done{2}.'.format(

        opti, log_begin_green, log_end_format))

print("Experiment: {0}done{1}".format(log_begin_green, log_end_format))
import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns

fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col')

ax[0, 0].set_title("loss")

ax[0, 1].set_title("val-loss")

ax[1, 0].set_title("accuracy")

ax[1, 1].set_title("val-accuracy")



lines, labels = [], []

for optimizer_name in history_per_opti:

    history = history_per_opti[optimizer_name]

    ax[0, 0].plot(history.history['loss'])

    ax[0, 1].plot(history.history['val_loss'])

    ax[1, 0].plot(history.history['accuracy'])

    line = ax[1, 1].plot(history.history['val_accuracy'])

    lines.append(line[0])

    labels.append(optimizer_name)



fig.legend(lines,labels, loc='center right', borderaxespad=0.1)

plt.subplots_adjust(right=0.85)
x_train, y_train = x_train[:200], y_train[:200]

print('Training data: {}'.format(x_train.shape))

print('Testing data: {}'.format(x_test.shape))
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Conv2D, 

                                     MaxPooling2D, Dropout, BatchNormalization)
def lenet5_reg(name='lenet', input_shape = input_shape, use_dropout = False,

              use_batchnorm = False, regularizer = None):

    layers = []

    layers += [Conv2D(6, kernel_size = (5,5), padding = 'same', input_shape = 

                     input_shape, kernel_regularizer = regularizer)]

    if use_batchnorm:

        layers +=[BatchNormalization()]

    layers+=[Activation('relu'),

            MaxPooling2D(pool_size = (2,2))]

    if use_dropout:

        layers +=[Dropout(0.25)]

        

    layers += [Conv2D(16, kernel_size = (5,5),

                      kernel_regularizer = regularizer)]

    if use_batchnorm:

        layers +=[BatchNormalization()]

    layers+=[Activation('relu'),

            MaxPooling2D(pool_size = (2,2))]

    if use_dropout:

        layers +=[Dropout(0.25)]

    

    layers +=[Flatten()]

    

    layers +=[Dense(120, kernel_regularizer = regularizer)]

    if use_batchnorm:

        layers +=[BatchNormalization()]

    layers+=[Activation('relu')]

    if use_dropout:

        layers +=[Dropout(0.25)]

    

    layers += [Dense(84, kernel_regularizer = regularizer)]

    layers +=[Activation('relu')]

    layers += [Dense(classes, activation = 'softmax')]

    model = Sequential(layers, name = name)

    return model

    

        

    

        
configurations = {

    'none':         {'use_dropout': False, 'use_batchnorm': False, 'regularizer': None},

    'l1':           {'use_dropout': False, 'use_batchnorm': False, 'regularizer': tf.keras.regularizers.l1(0.01)},

    'l2':           {'use_dropout': False, 'use_batchnorm': False, 'regularizer': tf.keras.regularizers.l2(0.01)},

    'dropout':      {'use_dropout': True,  'use_batchnorm': False, 'regularizer': None},

    'bn':           {'use_dropout': False, 'use_batchnorm': True,  'regularizer': None}

}
history_per_instance = dict()

random_seed = 42

print("Experiment: {0}start{1} (training logs = off)".format(log_begin_red, log_end_format))

for config_name in configurations:

    tf.random.set_seed(random_seed)

    np.random.seed(random_seed)

    

    model = lenet5_reg('lenet_{}'.format(config_name), **configurations[config_name])

    model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    print('\t> Training with {0}: {1}start{2}'.format(

    config_name, log_begin_red, log_end_format))

    history = model.fit(x_train, y_train, batch_size = 32, epochs = 150, 

                       validation_data = (x_test, y_test), verbose = 0)

    history_per_instance[config_name] = history

    print('\t> Training with {0}: {1}done{2}.'.format(

        config_name, log_begin_green, log_end_format))

print("Experiment: {0}done{1}".format(log_begin_green, log_end_format))
fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col') # add parameter `sharey='row'` for a more direct comparison

ax[0, 0].set_title("loss")

ax[0, 1].set_title("val-loss")

ax[1, 0].set_title("accuracy")

ax[1, 1].set_title("val-accuracy")



lines, labels = [], []

for config_name in history_per_instance:

    history = history_per_instance[config_name]

    ax[0, 0].plot(history.history['loss'])

    ax[0, 1].plot(history.history['val_loss'])

    ax[1, 0].plot(history.history['accuracy'])

    line = ax[1, 1].plot(history.history['val_accuracy'])

    lines.append(line[0])

    labels.append(config_name)



fig.legend(lines,labels, loc='center right', borderaxespad=0.1)

plt.subplots_adjust(right=0.84)
for config_name in history_per_instance:

    best_val_acc = max(history_per_instance[config_name].history['val_accuracy']) * 100

    print('Max val-accuracy for model "{}": {:2.2f}%'.format(config_name, best_val_acc))