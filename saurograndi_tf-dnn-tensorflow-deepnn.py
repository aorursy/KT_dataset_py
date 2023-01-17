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

import logging

## Tensorflow produces a lot of warnings. We generally want to suppress them. The below code does exactly that.

logging.getLogger("tensorflow").setLevel(logging.ERROR)

 

import tensorflow as tf

import matplotlib.pyplot as plt
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



Y = train["label"]

X = train.drop(labels = ["label"], axis = 1) 



scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)



X_test = scaler.fit_transform(test)
# Parameters

learning_rate = 0.1

num_steps = 1000

batch_size = 128

display_step = 100

num_epochs = 200



# Network Parameters

n_hidden_1 = 256 # 1st layer number of neurons

n_hidden_2 = 256 # 2nd layer number of neurons

num_input = 784 # MNIST data input (img shape: 28*28)

num_classes = 10 # MNIST total classes (0-9 digits)
def get_input_fn(mode):

    # We need two seperate branches because of differences in how we deal with the input data

    if mode == 'train':

        # When training, shuffling is OK, num_epochs denotes how many times to go over the training data

        return tf.estimator.inputs.numpy_input_fn(

            x={'images': X_train.astype('float64')}, y=Y_train.values,

            batch_size=batch_size, num_epochs=num_epochs, shuffle=True)

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
# Define and adding the neural network to the TensorGraph

def neural_net(x_dict):

    x = x_dict['images']

    # Hidden fully connected layer with 256 neurons

    layer_1 = tf.layers.dense(x, n_hidden_1)

    # Hidden fully connected layer with 256 neurons

    layer_2 = tf.layers.dense(layer_1, n_hidden_2)

    # Output fully connected layer with a neuron for each class

    out_layer = tf.layers.dense(layer_2, num_classes)

    return out_layer
# Define the model function (following TF Estimator Template)

def model_fn(features, labels, mode):

    

    

    logits = neural_net(features)



    pred_classes = tf.argmax(logits, axis=1)

    pred_probas = tf.nn.softmax(logits)

    

    # If prediction mode, return early

    if mode == tf.estimator.ModeKeys.PREDICT:

        # return EstimatorSpec that would be only used for prediction

        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 

        

    # Else we are training and evaluating 

    # Define loss and optimizer

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(

        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())

    

    # Evaluate the accuracy of the model

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    

    # TF Estimators should return an EstimatorSpec

    # specifying the different ops for training, predicting

    estim_specs = tf.estimator.EstimatorSpec(

      mode=mode,

      predictions=pred_classes,

      loss=loss_op,

      train_op=train_op,

      eval_metric_ops={'accuracy': acc_op})



    return estim_specs
model = tf.estimator.Estimator(model_fn)



# Train the Model

# We do not need to start a tf.session!

input_fn = get_input_fn('train')

# The model can be trained using the Estimator's train function

model.train(input_fn, steps=num_steps)



# Remember, we need the evaluation input function when we test our model

# Again, we do not need to explicitly start a tf.session! 

evaluate_input_fn = get_input_fn('evaluation')

model.evaluate(evaluate_input_fn)


predictions = list(model.predict(get_input_fn('prediction')))
# Save test predictions to file

output = pd.DataFrame({'ImageId': [i for i in range(1, 1+len(predictions))],

                       'Label': predictions})

output.to_csv('submission.csv', index=False)