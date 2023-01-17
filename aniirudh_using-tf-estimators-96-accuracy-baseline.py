import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from __future__ import absolute_import, division, print_function

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import numpy as np



tf.logging.set_verbosity(tf.logging.INFO)
# Load the training and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# separate out label into a different data frames

train_y = train['label']

train_x = train.drop(labels = ["label"],axis = 1)
# print the number of labels 

train_y.value_counts()
# Normalize the data

train_x = train_x / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (28 height x 28 width x 1 channel for gray)

train_x = train_x.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            zca_whitening=False,

                            width_shift_range = 0.1,

                            rotation_range = 10)



# this is not being used for now

datagen.fit(train_x)
def cnn_model(features, labels, mode):

    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])



    # 1st Convolutional Layer

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, 

                           kernel_size=[5, 5], padding="same",

                           activation=tf.nn.relu)



    # 1st Pooling Layer

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)



    # 2nd Convolutional Layer

    # This takes the output of previous pool layer as it's input

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, 

                           kernel_size=[5, 5],padding="same", 

                           activation=tf.nn.relu)

    

    # 2nd Pooling Layer

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



    # Before connecting the layer, we'll flatten the feature map

    dense = tf.layers.dense(inputs=tf.reshape(pool2, [-1, 7 * 7 * 64]), units=1024, activation=tf.nn.relu)

    

    # to improve the results apply dropout regularization to the layer to reduce overfitting

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)



    # Logits Layer

    logits = tf.layers.dense(inputs=dropout, units=10)



    predictions = {

          "classes": tf.argmax(input=logits, axis=1),

          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")

    }



    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)



    # Configure the Training Op (for TRAIN mode)

    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



    # Add evaluation metrics (for EVAL mode)

    eval_metric_ops = {

      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    }



    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
import numpy as np

import tensorflow as tf

import logging

from tensorflow.python.training import session_run_hook



class EarlyStoppingHook(session_run_hook.SessionRunHook):

    """Hook that requests stop at a specified step."""



    def __init__(self, monitor='val_loss', min_delta=0, patience=0,

                 mode='auto'):

        """

        """

        self.monitor = monitor

        self.patience = patience

        self.min_delta = min_delta

        self.wait = 0

        if mode not in ['auto', 'min', 'max']:

            logging.warning('EarlyStopping mode %s is unknown, '

                            'fallback to auto mode.', mode, RuntimeWarning)

            mode = 'auto'



        if mode == 'min':

            self.monitor_op = np.less

        elif mode == 'max':

            self.monitor_op = np.greater

        else:

            if 'acc' in self.monitor:

                self.monitor_op = np.greater

            else:

                self.monitor_op = np.less



        if self.monitor_op == np.greater:

            self.min_delta *= 1

        else:

            self.min_delta *= -1



        self.best = np.Inf if self.monitor_op == np.less else -np.Inf



    def begin(self):

        # Convert names to tensors if given

        graph = tf.get_default_graph()

        self.monitor = graph.as_graph_element(self.monitor)

        if isinstance(self.monitor, tf.Operation):

            self.monitor = self.monitor.outputs[0]



    def before_run(self, run_context):  # pylint: disable=unused-argument

        return session_run_hook.SessionRunArgs(self.monitor)



    def after_run(self, run_context, run_values):

        current = run_values.results



        if self.monitor_op(current - self.min_delta, self.best):

            self.best = current

            self.wait = 0

        else:

            self.wait += 1

            if self.wait >= self.patience:

                run_context.request_stop()
# Create the Estimator

classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="/tmp/model")



early_stopping_hook = EarlyStoppingHook(monitor='sparse_softmax_cross_entropy_loss/value', patience=10)



# Train the model

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_x},

                                                    y=train_y,

                                                    batch_size=64,

                                                    num_epochs=200,

                                                    shuffle=True)



classifier.train(input_fn=train_input_fn, steps=30000)
predict_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test},

                                                shuffle=False)



eval_results = classifier.predict(input_fn=predict_fn)
# predict results

result = []

for i in eval_results:

    result.append(i['classes'])



results = pd.Series(result,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)