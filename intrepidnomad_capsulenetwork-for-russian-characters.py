# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#credit to Hinton for concepts and Geron for example implementation

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #for creating of neural networks
import imageio #processing of input images
import glob #processing of input images
import matplotlib # plotting
import matplotlib.pyplot as plt #plotting
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#defining norm function that can avoids zero value problems, credit to Geron
def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=keepdims)
        return tf.sqrt(squared_norm + epsilon)
#define normalize function for ensuring that input values of image matrixes are normalized from 0 to 1
def normalize(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))
#define safe squashing function, credit to Geron
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
#load images for processing, using letters2 for training set and letters for validation set, note the normalization of the images to ensure the 0 to 255 values are converted to 0 to 1 
# Note: I had to manually resize some images in the set to correct dimensions; some were 31x32 ect..
filelist_validation = glob.glob(r'..\classification-of-handwritten-letters\letters\*.png')
images_validation= [imageio.imread(fname) for fname in filelist_validation]
imagearray_validation = normalize(np.stack(images_validation,axis=0))
imagearray_validation = imagearray_validation.astype('float32')

filelist = glob.glob(r'..\classification-of-handwritten-letters\letters2\*.png')
images= [imageio.imread(fname) for fname in filelist]
imagearray = normalize(np.stack(images,axis=0))
imagearray = imagearray.astype('float32')

data_validation = pd.read_csv(r"..\classification-of-handwritten-letters\letters.csv")
data = pd.read_csv(r"..\classification-of-handwritten-letters\letters2.csv")



labels = data['label'].as_matrix()-1
labels_validation = data_validation['label'].as_matrix()-1

#randomize training set so as not to bias initial learning

idx = np.random.permutation(len(imagearray))
imagearray,labels = imagearray[idx], labels[idx]
#image dimensions
im_x = 32
im_y = 32

#window dimensions for convolution
winx = 9
winy = 9

#number of feature filters
num_filters = 256

# number of channels in image
num_channels = 4
#tensorflow placeholder for input images
X = tf.placeholder(shape=[None, im_x, im_y, num_channels], dtype=tf.float32, name="X")
#number of capsule channels and other capsule network parameters
Capsule1_Channels = 32
Capsules_per_Channel = 8*8
Total_Cap = Capsules_per_Channel * Capsule1_Channels
Capsule1_Dimensions = 8
#defining convolution parameters for two convolutional layers according to CAPSNET paper 
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": Capsule1_Channels * Capsule1_Dimensions, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}
#initialize the convolutional layers of the capsule network
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
# raw capsule network 1 input
caps1_raw = tf.reshape(conv2, [-1, Total_Cap, Capsule1_Dimensions],
                       name="caps1_raw")
# Capsule network 2 parameters, adjusted from 10 to 33 to adjust for 33 Russian characters vs 10 MNIST digits
Capsule2_NumCharacters = 33
Capsule2_Dimensions = 16
#initialize capsule network weights according to CAPSNET paper
init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, Total_Cap, Capsule2_NumCharacters, Capsule2_Dimensions, Capsule1_Dimensions),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")
#calculate the capsule network output
caps1_output = squash(caps1_raw, name="caps1_output")
#implement the initial round of routing by agreement to get agreement matrix

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")


caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, Capsule2_NumCharacters, 1, 1],
                             name="caps1_output_tiled")


caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")


raw_weights = tf.zeros([batch_size, Total_Cap, Capsule2_NumCharacters, 1, 1],
                       dtype=np.float32, name="raw_weights")



routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")


weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True,
                             name="weighted_sum")


caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")



caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, Total_Cap, 1, 1, 1],
    name="caps2_output_round_1_tiled")


agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")
#define tensorflow routing by agreement loop to implement subsequent rounds of routing by agreement and make number of rounds adjustable by changing value in tf.less function of the condition loop # currently runs 3 additional rounds for 4 ( 1+3 ) total
def condition(raw_weights_loop,agreement_loop,caps2_output_loop, counter):
    return tf.less(counter, 4)

def loop_body(raw_weights_loop,agreement_loop,caps2_output_loop, counter):
    raw_weights_loop = tf.add(raw_weights_loop, agreement_loop,
                             name="raw_weights_loop")
    routing_weights_loop = tf.nn.softmax(raw_weights_loop,
                                        axis=2,
                                        name="routing_weights_loop")
    weighted_predictions_loop = tf.multiply(routing_weights_loop,
                                           caps2_predicted,
                                           name="weighted_predictions_loop")
    weighted_sum_loop = tf.reduce_sum(weighted_predictions_loop,
                                     axis=1, keepdims=True,
                                     name="weighted_sum_loop")
    caps2_output_loop = squash(weighted_sum_loop,
                              axis=-2,
                              name="caps2_output_loop")
    caps2_output_loop_tiled = tf.tile(
    caps2_output_loop, [1, Total_Cap, 1, 1, 1],
    name="caps2_output_loop_tiled")
    agreement_loop = tf.matmul(caps2_predicted, caps2_output_loop_tiled,
                      transpose_a=True, name="agreement_loop")
    return  raw_weights_loop,agreement_loop,caps2_output_loop, tf.add(counter,1)
with tf.name_scope("weights_loop"):
    counter = tf.constant(1)
    raw_weights_loop = raw_weights
    agreement_loop = agreement
    caps2_output_loop=caps2_output_round_1
    result = tf.while_loop(condition, loop_body, [raw_weights_loop,agreement_loop,caps2_output_loop,counter],swap_memory=True)
#use output of  capsule2 network to calculate label probabilities and select the most likely label candidate
y_proba = safe_norm(caps2_output_loop, axis=-2, name="y_proba")

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_proba_argmax

y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
#placeholder for training labels when training the network
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
#define margin loss parameters given in CAPSNET paper
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

#create matrix using labels y for use in calculating margin loss
T = tf.one_hot(y, depth=Capsule2_NumCharacters, name="T")
#calculate norm of capsule network 2 output
caps2_output_norm = safe_norm(caps2_output_loop, axis=-2, keepdims=True,
                              name="caps2_output_norm")
# calculate present and absence error for incorrect guesses of presence or absence of a character in images
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, Capsule2_NumCharacters),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, Capsule2_NumCharacters),
                          name="absent_error")
# loss matrix for input batch
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")
#calculate margin loss according to paper
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
# masking for reconstruction loss 
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=Capsule2_NumCharacters,
                                 name="reconstruction_mask")
#mask the capsule network output to define decoder input
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, Capsule2_NumCharacters, 1, 1],
    name="reconstruction_mask_reshaped")


caps2_output_masked = tf.multiply(
    caps2_output_loop, reconstruction_mask_reshaped,
    name="caps2_output_masked")


decoder_input = tf.reshape(caps2_output_masked,
                           [-1, Capsule2_NumCharacters * Capsule2_Dimensions],
                           name="decoder_input")
#define decoder network parameters
n_hidden1 = 512
n_hidden2 = 1024
n_output = im_x * im_y * num_channels
#implement decoder with tensorflow
with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

#compare original image input to decoder output to calculate reconstruction loss
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")
#define alpha parameter to tune inclusion of reconstruction relative to margin loss
alpha = 0.0005
#calculate hybrid loss
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
#define correctness and accuracy
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#define optimizer as ADAM optimizer as used in paper, tell tensorflow to minimize loss using the optimizer
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")
#initialize and run training
init = tf.global_variables_initializer()
saver = tf.train.Saver()


n_epochs = 30
batch_size = 10
restore_checkpoint = True

n_iterations_per_epoch = 594#letters2 / batch_size
n_iterations_validation = 165 #letters / batch_size
best_loss_val = np.infty
checkpoint_path = os.getenv("TMP") + "/color/my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch = imagearray[(iteration-1)*batch_size:iteration*batch_size]
            y_batch  = labels[(iteration-1)*batch_size:iteration*batch_size]
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, im_x, im_y, num_channels]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")
        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch = imagearray_validation[(iteration-1)*batch_size:iteration*batch_size]
            y_batch  = labels_validation[(iteration-1)*batch_size:iteration*batch_size]
            loss_val, acc_val, prediction, real = sess.run(
                    [loss, accuracy,y_pred,y],
                    feed_dict={X: X_batch.reshape([-1, im_x, im_y, num_channels]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print(prediction, real)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val