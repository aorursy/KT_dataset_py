# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

import numpy as np

import tensorflow as tf

import pickle

import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)



from PIL import Image



def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        dict = pickle.load(fo, encoding='bytes')

    return dict





batch_1 = unpickle('../input/data_batch_1')

batch_2 = unpickle('../input/data_batch_2')





metadata = unpickle('../input/batches.meta')

metadata
batch_2
# Training Parameters

learning_rate = 0.001

num_steps = 2000

batch_size = 128



num_input = 32 * 32 * 3

num_classes = 10

dropout = 0.25

def conv_net(x_dict, n_classes, dropout, reuse, is_training):

    with tf.variable_scope('ConvNet', reuse=reuse):

        # TF Estimator input is a dict, in case of multiple inputs

        x = x_dict['images']



        x = tf.reshape(x, [-1, 32, 32, 3])



        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.leaky_relu)

        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)



        # Convolution Layer with 64 filters and a kernel size of 3

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.leaky_relu)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2

        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)



        # Flatten the data to a 1-D vector for the fully connected layer

        fc1 = tf.contrib.layers.flatten(conv2)



        # Fully connected layer (in tf contrib folder for now)

        fc1 = tf.layers.dense(fc1, 1024)



        # Apply Dropout (if is_training is False, dropout is not applied)

        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)



        # Output layer, class prediction

        out = tf.layers.dense(fc1, n_classes)



        return out
# Define the model function (following TF Estimator Template)

def model_fn(features, labels, mode):

    # Build the neural network

    # Because Dropout have different behavior at training and prediction time, we

    # need to create 2 distinct computation graphs that still share the same weights.

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

        logits=logits_train, labels=tf.cast(labels, 'int32')))

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

input_fn = tf.estimator.inputs.numpy_input_fn(

    x={'images': batch_1[b'data'].astype('float32')},

    y=np.array(batch_1[b'labels']).astype('float32'),

    batch_size=batch_size,

    num_epochs=None,

    shuffle=True)

# Train the Model

model.train(input_fn, steps=10000)

n_images = 10

base = 200

test_images = batch_2[b'data'][base:base + n_images]

input_fn = tf.estimator.inputs.numpy_input_fn(

    x={'images': test_images.astype('float32')}, shuffle=False)



preds = list(model.predict(input_fn))



for i in range(n_images):

    from scipy.misc import toimage

    img = test_images[i].reshape([3, 32, 32]).transpose(1,2,0)

    plt.imshow(Image.fromarray(img))

    plt.show()

    

    print("Model prediction:", metadata[b'label_names'][preds[i]])


test_batch = unpickle('../input/test_batch')



# Evaluate the Model

# Define the input function for evaluating

input_fn = tf.estimator.inputs.numpy_input_fn(

    x={'images': np.array(test_batch[b'data']).astype('float32')}, y=np.array(test_batch[b'labels']).astype('float32'),

    batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method

model.evaluate(input_fn)