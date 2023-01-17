# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split
tf.logging.set_verbosity('INFO')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# print(train.shape)
train_label = train.pop('label')

print(train.info())
print(test.info())
print(train_label.value_counts())
#Normalization
train /= 255
test /= 255
#split data to train and validation set
train_features, val_features, train_label, val_label = train_test_split(
    train, train_label,
    train_size=0.75
)
#prepare input functions
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x' : train_features.values},
    y = train_label.values,
    num_epochs=None,
    shuffle=True
)
val_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x' : val_features.values},
    y = val_label.values,
    shuffle=False
)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x' : test.values}, shuffle=False)

def model_fn(features, labels, mode):

    '''
    this mode contains :
    conv -> max_pool -> conv -> max_pool->dense->dropout->softmax

    :param features: input features, has only one key 'x'
    :param labels:  with shape [num_samples, 10]
    :param mode: TRAIN EVAL PREDICT
    :return: EstimatorSpec
    '''
    inputs = tf.reshape(features['x'], [-1, 28,28,1])
    conv1 = tf.layers.conv2d(
        inputs = inputs,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )
    pool2 = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(
        inputs=pool2,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        training = mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(inputs=dropout, units=10)

    predicts = {
        'classid' : tf.argmax(logits, axis=1),
        'probabilities' : tf.nn.softmax(logits)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)

    losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(losses, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=losses, train_op=train_op)

    eval_metrics = {'accuracy' : tf.metrics.accuracy(labels, predicts['classid'])}
    return tf.estimator.EstimatorSpec(mode, loss=losses, eval_metric_ops=eval_metrics)

#create estimator
cnn_classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='mode'
)

#train and eval
cnn_classifier.train(train_input_fn, steps=1000)
eval_result = cnn_classifier.evaluate(val_input_fn)
print('eval accuracy is {}'.format(eval_result['accuracy']))
predict_result = cnn_classifier.predict(predict_input_fn)