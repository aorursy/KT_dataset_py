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
from IPython.display import HTML

HTML("""<iframe width="560" height="315" src="https://www.youtube.com/embed/pPN8d0E3900" frameborder="0" allowfullscreen></iframe>""")
HTML("""<iframe width="560" height="315" src="https://www.youtube.com/embed/2Kawrd5szHE" frameborder="0" allowfullscreen></iframe>""")
from __future__ import division, print_function, unicode_literals
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)
#tf.reset_default_graph()

from tensorflow.python.framework import ops

ops.reset_default_graph()
np.random.seed(42)

#tf.set_random_seed(42)

tf.random.set_seed(45)
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/")
from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

# mnist = mnist.load_data()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.

X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.

#Y_train = to_categorical(Y_train.astype('float32'))

#Y_test = to_categorical(Y_test.astype('float32'))
X_train.shape
n_samples = 5



plt.figure(figsize=(n_samples * 2, 3))

for index in range(n_samples):

    plt.subplot(1, n_samples, index + 1)

    sample_image = X_train[index].reshape(28, 28)

    plt.imshow(sample_image, cmap="binary")

    plt.axis("off")



plt.show()
Y_train[:n_samples]
#X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X") # original not working

tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
caps1_n_maps = 32

caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules

caps1_n_dims = 8
conv1_params = {

    "filters": 256,

    "kernel_size": 9,

    "strides": 1,

    "padding": "valid",

    "activation": tf.nn.relu,

}



conv2_params = {

    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters

    "kernel_size": 9,

    "strides": 2,

    "padding": "valid",

    "activation": tf.nn.relu

}
conv1 = tf.compat.v1.layers.conv2d(X, name="conv1", **conv1_params)

conv2 = tf.compat.v1.layers.conv2d(conv1, name="conv2", **conv2_params)
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],

                       name="caps1_raw")
def squash(s, axis=-1, epsilon=1e-7, name=None):

    #with tf.name_scope(name, default_name="squash"):

    with tf.name_scope(name):

        #squared_norm = tf.reduce_sum(tf.square(s), axis=axis,

        #                             keep_dims=True)

        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,

                                     keepdims=True)

        safe_norm = tf.sqrt(squared_norm + epsilon)

        squash_factor = squared_norm / (1. + squared_norm)

        unit_vector = s / safe_norm

        return squash_factor * unit_vector
caps1_output = squash(caps1_raw, name="caps1_output")
caps2_n_caps = 10

caps2_n_dims = 16
init_sigma = 0.1



#W_init = tf.random_normal(

W_init = tf.random.normal(

    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),

    stddev=init_sigma, dtype=tf.float32, name="W_init")

W = tf.Variable(W_init, name="W")
batch_size = tf.shape(X)[0]

W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
caps1_output_expanded = tf.expand_dims(caps1_output, -1,

                                       name="caps1_output_expanded")

caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,

                                   name="caps1_output_tile")

caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],

                             name="caps1_output_tiled")
W_tiled
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,

                            name="caps2_predicted")
caps2_predicted
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],

                       dtype=np.float32, name="raw_weights")
routing_weights = tf.nn.softmax(raw_weights, name="routing_weights")
weighted_predictions = tf.multiply(routing_weights, caps2_predicted,

                                   name="weighted_predictions")

weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,

                             name="weighted_sum")
caps2_output_round_1 = squash(weighted_sum, axis=-2,

                              name="caps2_output_round_1")
caps2_output_round_1
caps2_predicted
caps2_output_round_1
caps2_output_round_1_tiled = tf.tile(

    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],

    name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,

                      transpose_a=True, name="agreement")
raw_weights_round_2 = tf.add(raw_weights, agreement,

                             name="raw_weights_round_2")
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,

                                        name="routing_weights_round_2")

weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,

                                           caps2_predicted,

                                           name="weighted_predictions_round_2")

weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,

                                     axis=1, keepdims=True,

                                     name="weighted_sum_round_2")

caps2_output_round_2 = squash(weighted_sum_round_2,

                              axis=-2,

                              name="caps2_output_round_2")
caps2_output = caps2_output_round_2
def condition(input, counter):

    return tf.less(counter, 100)



def loop_body(input, counter):

    output = tf.add(input, tf.square(counter))

    return output, tf.add(counter, 1)



with tf.name_scope("compute_sum_of_squares"):

    counter = tf.constant(1)

    sum_of_squares = tf.constant(0)



    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])

    



with tf.compat.v1.Session() as sess:

    print(sess.run(result))
sum([i**2 for i in range(1, 100 + 1)])
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):

    with tf.name_scope(name):

        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,

                                     keepdims=keep_dims)

        return tf.sqrt(squared_norm + epsilon)
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_proba_argmax
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
y_pred
y = tf.compat.v1.placeholder(shape=[None], dtype=tf.int64, name="y")
m_plus = 0.9

m_minus = 0.1

lambda_ = 0.5
T = tf.one_hot(y, depth=caps2_n_caps, name="T")
with tf.compat.v1.Session():

    print(T.eval(feed_dict={y: np.array([0, 1, 2, 3, 9])}))
caps2_output
caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,

                              name="caps2_output_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),

                              name="present_error_raw")

present_error = tf.reshape(present_error_raw, shape=(-1, 10),

                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),

                             name="absent_error_raw")

absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),

                          name="absent_error")
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,

           name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
mask_with_labels = tf.compat.v1.placeholder_with_default(False, shape=(),

                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition

                                 lambda: y,        # if True

                                 lambda: y_pred,   # if False

                                 name="reconstruction_targets")
reconstruction_mask = tf.one_hot(reconstruction_targets,

                                 depth=caps2_n_caps,

                                 name="reconstruction_mask")
reconstruction_mask
caps2_output
reconstruction_mask_reshaped = tf.reshape(

    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],

    name="reconstruction_mask_reshaped")
caps2_output_masked = tf.multiply(

    caps2_output, reconstruction_mask_reshaped,

    name="caps2_output_masked")
caps2_output_masked
decoder_input = tf.reshape(caps2_output_masked,

                           [-1, caps2_n_caps * caps2_n_dims],

                           name="decoder_input")
n_hidden1 = 512

n_hidden2 = 1024

n_output = 28 * 28
with tf.name_scope("decoder"):

    hidden1 = tf.compat.v1.layers.dense(decoder_input, n_hidden1,

                              activation=tf.nn.relu,

                                        name="hidden1")

    hidden2 = tf.compat.v1.layers.dense(hidden1, n_hidden2,

                                        activation=tf.nn.relu,

                                        name="hidden2")

    decoder_output = tf.compat.v1.layers.dense(hidden2, n_output,

                                               activation=tf.nn.sigmoid,

                                               name="decoder_output")
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")

squared_difference = tf.square(X_flat - decoder_output,

                               name="squared_difference")

reconstruction_loss = tf.reduce_mean(squared_difference,

                                    name="reconstruction_loss")
alpha = 0.0005



loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
correct = tf.equal(y, y_pred, name="correct")

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
optimizer = tf.compat.v1.train.AdamOptimizer()

training_op = optimizer.minimize(loss, name="training_op")
init = tf.compat.v1.global_variables_initializer()

saver = tf.compat.v1.train.Saver()
len(X_train)
X_train[0:0+50]
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.08, random_state=42)
len(X_train)
len(X_val)
#y_train[c0:c0+batch_size]

X_val.shape

Y_val.shape
n_epochs = 10

batch_size = 50

restore_checkpoint = True



n_iterations_per_epoch = len(X_train) // batch_size

n_iterations_validation = len(X_val) // batch_size

best_loss_val = np.infty

checkpoint_path = "./my_capsule_network"



with tf.compat.v1.Session() as sess:

    if restore_checkpoint and tf.compat.v1.train.checkpoint_exists(checkpoint_path):

        saver.restore(sess, checkpoint_path)

    else:

        init.run()



    for epoch in range(n_epochs):

        b0 = 0

        c0 = 0

        for iteration in range(1, n_iterations_per_epoch + 1):

            X_batch, y_batch = X_train[b0:b0+batch_size], Y_train[b0:b0+batch_size]

            # Run the training operation and measure the loss:

            _, loss_train = sess.run(

                [training_op, loss],

                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),

                           y: y_batch,

                           mask_with_labels: True})

            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(

                      iteration, n_iterations_per_epoch,

                      iteration * 100 / n_iterations_per_epoch,

                      loss_train),

                  end="")

            b0+=batch_size



        # At the end of each epoch,

        # measure the validation loss and accuracy:

        loss_vals = []

        acc_vals = []

        for iteration in range(1, n_iterations_validation + 1):

            X_batch, y_batch = X_val[c0:c0+batch_size], Y_val[c0:c0+batch_size]

            loss_val, acc_val = sess.run(

                    [loss, accuracy],

                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),

                               y: y_batch})

            loss_vals.append(loss_val)

            acc_vals.append(acc_val)

            print("\rEvaluating the model: {}/{} ({:.1f}%) {}".format(

                      iteration, n_iterations_validation,

                      iteration * 100 / n_iterations_validation, c0),

                  end=" " * 10)

            c0+=batch_size

        loss_val = np.mean(loss_vals)

        acc_val = np.mean(acc_vals)

        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(

            epoch + 1, acc_val * 100, loss_val,

            " (improved)" if loss_val < best_loss_val else ""))



        # And save the model if it improved:

        if loss_val < best_loss_val:

            save_path = saver.save(sess, checkpoint_path)

            best_loss_val = loss_val
b0 = 0

n_iterations_test = len(X_test) // batch_size



with tf.compat.v1.Session() as sess:

    saver.restore(sess, checkpoint_path)



    loss_tests = []

    acc_tests = []

    for iteration in range(1, n_iterations_test + 1):

        X_batch, y_batch = X_test[b0:b0+batch_size], Y_test[b0:b0+batch_size]

        loss_test, acc_test = sess.run(

                [loss, accuracy],

                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),

                           y: y_batch})

        loss_tests.append(loss_test)

        acc_tests.append(acc_test)

        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(

                  iteration, n_iterations_test,

                  iteration * 100 / n_iterations_test),

              end=" " * 10)

        b0+=batch_size

    loss_test = np.mean(loss_tests)

    acc_test = np.mean(acc_tests)

    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(

        acc_test * 100, loss_test))
n_samples = 5



sample_images = X_test[:n_samples].reshape([-1, 28, 28, 1])



with tf.compat.v1.Session() as sess:

    saver.restore(sess, checkpoint_path)

    caps2_output_value, decoder_output_value, y_pred_value = sess.run(

            [caps2_output, decoder_output, y_pred],

            feed_dict={X: sample_images,

                       y: np.array([], dtype=np.int64)})
sample_images.shape
decoder_output_value.reshape([-1, 28, 28]).shape

#reconstructions.shape
sample_images = sample_images.reshape(-1, 28, 28)

reconstructions = decoder_output_value.reshape([-1, 28, 28])



plt.figure(figsize=(n_samples * 2, 3))

for index in range(n_samples):

    plt.subplot(1, n_samples, index + 1)

    plt.imshow(sample_images[index], cmap="binary")

    plt.title("Label:" + str(Y_test[index]))

    plt.axis("off")



plt.show()



plt.figure(figsize=(n_samples * 2, 3))

for index in range(n_samples):

    plt.subplot(1, n_samples, index + 1)

    plt.title("Predicted:" + str(y_pred_value[index]))

    plt.imshow(reconstructions[index], cmap="binary")

    plt.axis("off")

    

plt.show()
caps2_output_value.shape
def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):

    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25

    pose_parameters = np.arange(caps2_n_dims) # 0, 1, ..., 15

    tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])

    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps

    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]

    return tweaks + output_vectors_expanded
n_steps = 11



tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)

tweaked_vectors_reshaped = tweaked_vectors.reshape(

    [-1, 1, caps2_n_caps, caps2_n_dims, 1])
tweak_labels = np.tile(Y_test[:n_samples], caps2_n_dims * n_steps)



with tf.compat.v1.Session() as sess:

    saver.restore(sess, checkpoint_path)

    decoder_output_value = sess.run(

            decoder_output,

            feed_dict={caps2_output: tweaked_vectors_reshaped,

                       mask_with_labels: True,

                       y: tweak_labels})
tweak_reconstructions = decoder_output_value.reshape(

        [caps2_n_dims, n_steps, n_samples, 28, 28])
for dim in range(3):

    print("Tweaking output dimension #{}".format(dim))

    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))

    for row in range(n_samples):

        for col in range(n_steps):

            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)

            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")

            plt.axis("off")

    plt.show()