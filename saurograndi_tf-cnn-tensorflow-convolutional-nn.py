# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import warnings

warnings.filterwarnings('ignore')

tf.logging.set_verbosity(tf.logging.ERROR)

 

import matplotlib.pyplot as plt
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



Y = train["label"]

X = train.drop(labels = ["label"], axis = 1) 



scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)



X_test = scaler.fit_transform(test)
# Training Parameters

learning_rate = 0.001

num_steps = 2000

batch_size = 128



# Network Parameters

num_input = 784 # MNIST data input (img shape: 28*28)

num_classes = 10 # MNIST total classes (0-9 digits)

dropout = 0.25 # Dropout, probability to drop a unit
def conv_net(x_dict, n_classes, dropout, reuse, is_training):

    

    # Define a scope for reusing the variables

    with tf.variable_scope('ConvNet', reuse=reuse):

        # TF Estimator input is a dict, in case of multiple inputs

        x = x_dict['images']



        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)

        # Reshape to match picture format [Height x Width x Channel]

        # Tensor input becomes 4-D: [Batch Size, Height, Width, Channel]

        x = tf.reshape(x, shape=[-1, 28, 28, 1])



        # Convolution Layer with 32 filters and a kernel size of 5

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2

        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)



        # Convolution Layer 2 with 64 filters and a kernel size of 3

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2

        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)



        # Flatten the data to a 1-D vector for the fully connected layer

        fc1 = tf.contrib.layers.flatten(conv2)



        # Fully connected layer

        fc1 = tf.layers.dense(fc1, 1024)

        # Apply Dropout (if is_training is False, dropout is not applied)

        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)



        # Output layer, class prediction

        out = tf.layers.dense(fc1, n_classes)



    return out
# Define the model function (following TF Estimator Template)

def model_fn(features, labels, mode):

    

    # Build the neural network

    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)

    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    

    # Predictions

    pred_classes = tf.argmax(logits_test, axis=1)

    pred_probas = tf.nn.softmax(logits_test)

    

    # If prediction mode, early return

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 

        

    # Define loss and optimizer

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(

        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    

    # Evaluate the accuracy of the model

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    

    # TF Estimators requires to return a EstimatorSpec, that specify

    # the different ops for training, evaluating, ...

    estim_specs = tf.estimator.EstimatorSpec(

      mode=mode,

      predictions=pred_classes,

      loss=loss_op,

      train_op=train_op,

      eval_metric_ops={'accuracy': acc_op})



    return estim_specs
# Build the Estimator

model = tf.estimator.Estimator(model_fn)
# input_function for the Estimator    

def get_input_fn(mode):

    # We need two seperate branches because of differences in how we deal with the input data

    if mode == 'train':

        # When training, shuffling is OK, num_epochs denotes how many times to go over the training data

        return tf.estimator.inputs.numpy_input_fn(

            x={'images': X_train.astype('float64')}, y=Y_train.values,

            batch_size=batch_size, num_epochs=None, shuffle=True)

    elif mode == "evaluation":

        # When testing, don't shuffle

        # Default value for num_epochs is 1; we only want to go over the test set once

        return tf.estimator.inputs.numpy_input_fn(

            x={'images': X_val.astype('float64')}, y=Y_val.values,

            batch_size=batch_size, shuffle=False)

    elif mode == "prediction":

        # When testing, don't shuffle

        # Default value for num_epochs is 1; we only want to go over the test set once

        return tf.estimator.inputs.numpy_input_fn(

            x={'images': X_test.astype('float64')}, y=None,

            batch_size=batch_size, shuffle=False)
# Train

tf.logging.set_verbosity(tf.logging.INFO)

input_fn = get_input_fn('train')



model.train(input_fn, steps=num_steps)
# Evaluate

input_fn = get_input_fn('evaluation')

model.evaluate(input_fn)
# Predict

predictions = list(model.predict(get_input_fn('prediction')))
# Save test predictions to file

output = pd.DataFrame({'ImageId': [i for i in range(1, 1+len(predictions))],

                       'Label': predictions})

output.to_csv('submission.csv', index=False)