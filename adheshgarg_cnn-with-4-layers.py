# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import pandas as pd
test  = pd.read_csv('../input/test.csv',)
train = pd.read_csv('../input/train.csv', nrows = 40000)
valid = pd.read_csv('../input/train.csv',skiprows = 40000, nrows = 2000)

train_dataset = train.iloc[:,1:785]
train_labels  = train.iloc[:,0]

valid_dataset = valid.iloc[:,1:785]
valid_labels  = valid.iloc[:,0]

test_dataset = test.iloc[:,0:784]

train_dataset_np = (train_dataset.as_matrix()/255.0) - 1.0
train_labels_np = train_labels.as_matrix()

valid_dataset_np = (valid_dataset.as_matrix()/255.0) - 1.0
valid_labels_np = valid_labels.as_matrix()

test_np = (test_dataset.as_matrix()/255.0) - 1
test_dataset.head()

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5 # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        #x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x_dict, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x , 32, 5, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 32, 5, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.layers.dropout(conv2, rate=0.25)
        conv3 = tf.layers.conv2d(conv2, 64, 3, padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, 64, 3, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
        fc1 = tf.layers.dropout(fc1, rate=0.25)
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv4)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes, activation=tf.nn.softmax)
    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

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
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

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

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    train_dataset_np, train_labels_np,
    batch_size=batch_size, num_epochs=200, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=valid_dataset_np, y=valid_labels_np,
    batch_size=batch_size,num_epochs=200, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x=test_np,
    batch_size=batch_size,num_epochs=1, shuffle=False)
res = model.predict(input_fn2)
result = pd.Series(res, name = "Label")
submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), result], axis = 1)
submission.to_csv("Submission.csv", index = False)
