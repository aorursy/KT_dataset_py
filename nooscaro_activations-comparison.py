import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

import pickle
# importing dataset

infile = open('/kaggle/input/cifar10-preprocessed/data.pickle', 'rb')

data = pickle.load(infile)
x_train = np.rollaxis(data['x_train'], 1, 4)

y_train = data['y_train']



x_validation = np.rollaxis(data['x_validation'], 1, 4)

y_validation = data['y_validation']



x_test = np.rollaxis(data['x_test'], 1, 4)

y_test = data['y_test']
def test_model(activation, epochs=10):

    ## Creating the model

    model = models.Sequential(name=activation + "_test")

    # add convolutional layers

    model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3)))

    for i in range(0, 2):

        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation=activation))

    # add dense layers

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation=activation))

    model.add(layers.Dense(10))

    

    ## Compiling the model

    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

    

    ## Training the model

    history = model.fit(x_train, y_train, epochs=epochs, 

                    validation_data=(x_validation, y_validation))

    return history
def plot_results(ax, results):

    ax.plot(results.history['accuracy'], label='accuracy')

    ax.plot(results.history['val_accuracy'], label = 'val_accuracy')

    ax.set_xlabel('Epoch')

    ax.set_ylabel('Accuracy')

    ax.set_title(results.model.name + " train results")

    ax.set_ylim([0.5, 1])

    ax.set_yticks(np.arange(0, 1, step=0.05))

    ax.set_xticks(np.arange(0, 10, step=1))

    ax.legend(loc='upper right')
activations = [

    'linear',

    'relu',

    'softmax',

    'hard_sigmoid',

    'sigmoid',

    'swish',

    'tanh',

    'exponential',

    'elu',

    'selu'

]



activation_tests = [test_model(act) for act in activations]
# plot training statistics



nrows = 2

ncols = 5

fig, axes = plt.subplots(nrows, ncols, figsize=(25, 10))

fig.suptitle("Activation functions train results", fontsize=16)

for ax, result in zip(axes.flatten(), activation_tests):

    plot_results(ax, result)



plt.show()