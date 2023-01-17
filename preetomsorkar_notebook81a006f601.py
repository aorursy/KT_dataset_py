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
import tensorflow as tf

print(tf.__version__)
#### PACKAGE IMPORTS ####

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets, model_selection 

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras import regularizers

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

%matplotlib inline
#Sanity Check

iris_data = datasets.load_iris()

print(iris_data.keys())

print(iris_data['DESCR'])
def read_in_and_split_data(iris_data):

    """

    This function takes the Iris dataset as loaded by sklearn.datasets.load_iris(), and then 

    splits so that the training set includes 90% of the full dataset, with the test set 

    making up the remaining 10%.

    """

    data = iris_data['data']

    targets = iris_data['target']

    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size = 0.1)

    return train_data, test_data, train_targets, test_targets
# Run function to generate the test and training data.



iris_data = datasets.load_iris()

train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)
#Sanity Check

print(train_data.shape)

print(test_data.shape)

print(train_targets.shape)

print(test_targets.shape)
# Convert targets to a one-hot encoding



train_targets = tf.keras.utils.to_categorical(np.array(train_targets))

test_targets = tf.keras.utils.to_categorical(np.array(test_targets))
#Sanity Check

print(train_targets.shape)

print(test_targets.shape)
def get_regularised_model(input_shape, dropout_rate, weight_decay):

    """

    This function should build a regularised Sequential model according to the above specification. 

    The dropout_rate argument in the function is used to set the Dropout rate for all Dropout layers.

    L2 kernel regularisation (weight decay) is added using the weight_decay argument to 

    set the weight decay coefficient in all Dense layers that use L2 regularisation.

    Your function should return the model.

    """

    model = Sequential([

              Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Constant(1.), 

                    kernel_regularizer= regularizers.l2(weight_decay), input_shape = input_shape),

              Dense(128, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dense(128, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dropout(dropout_rate),

              Dense(128, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dense(128, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              BatchNormalization(),

              Dense(64, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dense(64, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dropout(dropout_rate),

              Dense(64, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dense(64, activation='relu', kernel_regularizer= regularizers.l2(weight_decay)),

              Dense(3, activation='softmax', kernel_regularizer= regularizers.l2(weight_decay))

              ])

    

    return model
# Instantiate the model, using a dropout rate of 0.3 and weight decay coefficient of 0.001



reg_model = get_regularised_model(train_data[0].shape, 0.3, 0.001)
#Sanity Check

reg_model.summary()
def compile_model(model):

    """

    This function takes in the model returned from my 'get_regularised_model' function, 

    and compiles it with an optimiser,loss function and metric.

    The model is compiled using the Adam optimiser (with learning rate set to 0.0001), 

    the categorical crossentropy loss function and accuracy as the only metric. 

    The function doesn't return anything; the model will be compiled in-place.

    """

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 

                  loss = 'categorical_crossentropy', metrics=['accuracy'])
def get_callbacks():

    """

    This function's job is to create and return a tuple (early_stopping, learning_rate_reduction) callbacks.

    The early stopping callback is used and monitors validation loss with the mode set to "min" and patience of 30.

    The learning rate reduction on plateaux is used with a learning rate factor of 0.2 and a patience of 20.

    """

    early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode = 'min')

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience=20)

    return early_stopping, learning_rate_reduction
#Train the regularised model with the callbacks.

compile_model(reg_model)

early_stopping, learning_rate_reduction = get_callbacks()

call_history = reg_model.fit(train_data, train_targets, epochs=800, validation_split=0.15,

                         callbacks=[early_stopping, learning_rate_reduction], verbose=2)
#This cell plots the epoch vs accuracy graph



try:

    plt.plot(call_history.history['accuracy'])

    plt.plot(call_history.history['val_accuracy'])

except KeyError:

    plt.plot(call_history.history['acc'])

    plt.plot(call_history.history['val_acc'])

plt.title('Accuracy vs. epochs')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='lower right')

plt.show() 
#Run this cell to plot the  loss vs epoch graph

plt.plot(call_history.history['loss'])

plt.plot(call_history.history['val_loss'])

plt.title('Loss vs. epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.show() 
# Evaluating the model on the test set



test_loss, test_acc = reg_model.evaluate(test_data, test_targets, verbose=0)

print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(test_loss, 100 * test_acc))