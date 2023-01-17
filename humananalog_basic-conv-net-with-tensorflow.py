import numpy as np

import pandas as pd

import tensorflow as tf



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = (16.0, 8.0)   # change default figure size
import csv as csv



image_width = 28

image_height = 28

num_pixels = image_width * image_height



# Unfortunately, using the entire data set exceeds Kaggle's memory limit.

#num_examples = 42000

num_examples = 2000



data_X = np.zeros((num_examples, num_pixels))

data_y = np.zeros(num_examples, dtype=np.int)



with open("../input/train.csv", "rt") as f:

    reader = csv.reader(f)

    header = next(reader)

    

    for j, row in enumerate(reader):

        if j == num_examples:

            break

        for (i, col) in enumerate(row):

            if i == 0:

                data_y[j] = int(col)

            else:

                data_X[j][i - 1] = float(col) / 255

                

print("data_X is %d bytes" % data_X.nbytes)

print("data_y is %d bytes" % data_y.nbytes)
fig = plt.figure(figsize=(5, 5))

num_horz = 4

num_vert = 4

for i in range(num_horz * num_vert):

    ax = fig.add_subplot(num_vert, num_horz, i + 1, xticks=[], yticks=[])

    image_data = (data_X[i] * 255).reshape(image_width, image_height)

    ax.imshow(image_data, cmap=cm.Greys_r, interpolation='none')
def onehot(labels, num_outputs):

    m = labels.shape[0]

    y = np.zeros((m, num_outputs))

    for i, label in enumerate(labels):

        y[i][int(label)] = 1.0

    return y
np.random.seed(666)

indices = np.random.permutation(len(data_X))



num_train = int(0.6 * num_examples)

num_val = int(0.2 * num_examples)

num_test = num_examples - num_val - num_train



X_train = data_X[indices[:num_train]]

y_train = data_y[indices[:num_train]]

X_val   = data_X[indices[num_train:-num_test]]

y_val   = data_y[indices[num_train:-num_test]]

X_test  = data_X[indices[-num_test:]]

y_test  = data_y[indices[-num_test:]]



X_mean = np.zeros((1, num_pixels))

X_std = np.zeros((1, num_pixels))



y_train_labels = y_train

y_val_labels   = y_val

y_test_labels  = y_test



y_train = onehot(y_train, 10)

y_val   = onehot(y_val, 10)

y_test  = onehot(y_test, 10)
%xdel data_X

%xdel data_y
X_mean = np.mean(X_train, axis=0, keepdims=True)

X_std = np.std(X_train, axis=0, keepdims=True)



X_train -= X_mean

X_val   -= X_mean

X_test  -= X_mean
fig = plt.figure(figsize=(2, 2))

ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

ax.set_frame_on(False)

ax.set_axis_off()

image_data = (X_mean * 255).reshape(image_width, image_height)

ax.imshow(image_data, cmap=cm.Greys_r, interpolation='none')
import math



tf.set_random_seed(7777)

tf.reset_default_graph()



# Init the weights with a small amount of noise.

def weight_variable(name, shape):

    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())    



# Use a slightly positive initial bias to avoid dead ReLUs.

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



# Convolution with stride 1 and enough zero padding to keep width/height the same.

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



# Plain old max pooling over 2x2 blocks.

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# The input data.

x = tf.placeholder(tf.float32, [None, 784], name="x-input")

y = tf.placeholder(tf.float32, [None, 10], name="y-input")



with tf.name_scope("hyperparameters"):

    learning_rate = tf.placeholder(tf.float32, name="learning-rate")

    reg_lambda = tf.placeholder(tf.float32, name="regularization")



# Reshape x into a 4d tensor. -1 is because we don't know the number of examples

# yet. 28x28 is the width and height, 1 is the number of color channels.

x_image = tf.reshape(x, [-1, 28, 28, 1])



with tf.name_scope("conv1_1"):

    W_conv1_1 = weight_variable("W_conv1_1", [3, 3, 1, 32])

    b_conv1_1 = bias_variable([32])

    h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)



with tf.name_scope("conv1_2"):

    W_conv1_2 = weight_variable("W_conv1_2", [3, 3, 32, 32])

    b_conv1_2 = bias_variable([32])

    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)



with tf.name_scope("pool1"):

    h_pool1 = max_pool_2x2(h_conv1_2)



with tf.name_scope("conv2_1"):

    W_conv2_1 = weight_variable("W_conv2_1", [3, 3, 32, 64])

    b_conv2_1 = bias_variable([64])

    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)



with tf.name_scope("conv2_2"):

    W_conv2_2 = weight_variable("W_conv2_2", [3, 3, 64, 64])

    b_conv2_2 = bias_variable([64])

    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)



with tf.name_scope("pool2"):

    h_pool2 = max_pool_2x2(h_conv2_2)



    # Reshape the output into a vector that we can pass to the FC layer.

    h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])



with tf.name_scope("fc3"):

    W_fc3 = weight_variable("W_fc3", [7 * 7 * 64, 128])

    b_fc3 = bias_variable([128])

    h_fc3 = tf.nn.relu(tf.matmul(h_flat, W_fc3) + b_fc3)



with tf.name_scope("fc3-dropout"):

    # Apply dropout to reduce overfitting.

    keep_prob = tf.placeholder(tf.float32, name="dropout-probability")

    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)



with tf.name_scope("fc4"):

    W_fc4 = weight_variable("W_fc4", [128, 128])

    b_fc4 = bias_variable([128])

    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)



with tf.name_scope("fc4-dropout"):

    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)



with tf.name_scope("fc5"):

    W_fc5 = weight_variable("W_fc5", [128, 10])

    b_fc5 = bias_variable([10])

    y_pred = tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)



# Softmax, so use cross entropy loss.

with tf.name_scope("loss-function"):

    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))

    

    # L2 regularization for the fully connected parameters.

    regularizers = (tf.nn.l2_loss(W_fc3) + tf.nn.l2_loss(b_fc3) +

                    tf.nn.l2_loss(W_fc4) + tf.nn.l2_loss(b_fc4) +

                    tf.nn.l2_loss(W_fc5) + tf.nn.l2_loss(b_fc5))

    loss += 5e-4 * regularizers



# Use ADAM (instead of plain gradient descent).

with tf.name_scope('train'):

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)



# The accuracy op computes the % correct on a dataset with labels. 

with tf.name_scope("accuracy"):

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# For doing inference on the test set without labels.

with tf.name_scope("inference"):

    inference = tf.argmax(y_pred, 1)



init = tf.initialize_all_variables()



sess = tf.InteractiveSession()
index_in_epoch = 0

epochs_completed = 0



# Based on code from learn/datasets/mnist.py

def next_batch(batch_size):

    global index_in_epoch, epochs_completed, X_train, y_train, y_train_labels

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # Epoch completed?

    if index_in_epoch > len(X_train):

        # Shuffle the data.

        perm = np.arange(len(X_train))

        np.random.shuffle(perm)

        X_train = X_train[perm]

        y_train = y_train[perm]

        y_train_labels = y_train_labels[perm]



        # Start next epoch.

        start = 0

        index_in_epoch = batch_size

        epochs_completed += 1



    end = index_in_epoch

    return X_train[start:end], y_train[start:end]
import math

import time



def train_and_validate(max_steps, batch_size, print_every, verbose=True, acceptable_loss=0.001, smooth_loss=False):

    sess.run(init)



    loss_history = []

    loss_avg = 0

    smooth_steps = 20



    # Used by next_batch()

    global index_in_epoch, epochs_completed

    index_in_epoch = 0

    epochs_completed = 0



    for step in range(max_steps):

        start_time = time.time()



        # Get the next mini-batch of training data.

        batch_xs, batch_ys = next_batch(100)

        feed_dict = { x: batch_xs, y: batch_ys, learning_rate: lr, reg_lambda: reg, keep_prob: 0.5 }



        # Run the network.

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)



        duration = time.time() - start_time



        # If enabled, we calculate the average loss over the last X steps, since

        # the loss can be a bit jittery when using stochastic gradient descent.

        if smooth_loss:

            loss_avg += loss_value

            if step % smooth_steps == 0:

                if step > 0: loss_avg /= smooth_steps

                loss_history.append(loss_avg)

                loss_avg = 0

        else:

            loss_history.append(loss_value)

        

        # Print the loss once every so many steps.

        if (step % print_every == 0) and verbose:

            print("    step: %4d, epoch: %2d, loss: %.3f (%.3f sec)" % \

                      (step, epochs_completed, loss_value, duration))



        # Stop the gradient descent if a user-specified loss is reached.

        if loss_value <= acceptable_loss or math.isnan(loss_value):

            print("    Loss is below acceptable limit, ending training after %d steps" % step)

            break



    if verbose:

        print("Final loss: %f" % loss_value)



    # Calculate cross validation score.

    score = sess.run(accuracy, feed_dict={x: X_val, y: y_val, keep_prob: 1.0})

    if verbose:

        print("Validation score: %g" % score)



    return loss_history, loss_value, score
# Hyperparameters used in the grid search

learning_rates = [0.001, 0.003, 0.01]

reg_lambdas = [0, 0.1, 0.3]            # regularization strength



from itertools import product

grid = list(product(learning_rates, reg_lambdas))



verbose = True
# How many random searches to perform

max_search = 10



grid = []

for i in range(max_search):

    lr = 10**np.random.uniform(-5, -2)

    reg = 10**np.random.uniform(-3, 1)

    grid.append((lr, reg))

    

verbose = False
grid = [(0.0003, 0.0)]

verbose = True
print("Training %d examples" % X_train.shape[0])



scores = []

start_time = time.time()

loss_history = {}



for i, params in enumerate(grid):

    lr, reg = params

    

    if verbose:

        print("*** learning rate: %g, regularization: %g" % (lr, reg))

    

    hist, loss_value, score = train_and_validate(max_steps=500, batch_size=100, print_every=50,\

                                                 verbose=verbose, acceptable_loss=0.001, smooth_loss=False)



    if not verbose:

        print("score: %0.6f, loss: %0.6f, rate: %0.6f, reg: %0.6f (%d/%d)" % (score, loss_value, lr, reg, i+1, max_search))



    key = "learn: %g, reg: %g" % (lr, reg)

    loss_history[key] = hist

    scores.append(score)



    if verbose:

        print()



print("Best validation score: %g" % np.max(scores))

print("Best parameters:", grid[np.argmax(scores)])

print("Time: %f sec" % (time.time() - start_time))
start_time = time.time()



# Calculate accuracy on training data. This should be a good score, 

# but not *too* good or we're overfitting.

print(sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1.0}))



duration = time.time() - start_time

print("duration %f sec" % duration)



# Calculate accuracy on validation data.

print(sess.run(accuracy, feed_dict={x: X_val, y: y_val, keep_prob: 1.0}))



# Calculate accuracy on test data. We do have labels for these examples, 

# but the examples themselves were not used to train the network.

print(sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))
def confusion_matrix(target, predicted):

    assert(target.shape == predicted.shape)

    

    num_classes = len(np.unique(target))

    confusion = np.zeros((num_classes, num_classes))

    

    for i in range(len(target)):      

        confusion[target[i], predicted[i]] += 1

    

    return confusion



def plot_confusion_matrix(conf):

    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.binary)

    plt.xticks(range(conf.shape[1]))

    plt.yticks(range(conf.shape[0]))

    plt.xlabel("predicted label")

    plt.ylabel("true label")

    plt.grid(False)

    plt.colorbar()

    

pred = sess.run(inference, feed_dict={x: X_test, keep_prob: 1.0})



conf = confusion_matrix(y_test_labels, pred)

print(conf)



plot_confusion_matrix(conf)
for k, v in loss_history.items():

    plt.plot(v, label=k)

plt.ylabel("Loss")

plt.xlabel("Time")

plt.legend()

plt.grid(True)
# Free up memory we don't need anymore.

%xdel X_train

%xdel X_test

%xdel X_val
# Change this to "if True" if you want to run this part of the script.

if False:

    import csv as csv



    num_test_examples = 28000

    X_kaggle = np.zeros((num_test_examples, num_pixels))



    with open("../input/test.csv", "rt") as f:

        reader = csv.reader(f)

        header = next(reader)



        for j, row in enumerate(reader):

            for (i, col) in enumerate(row):

                X_kaggle[j][i - 1] = float(col) / 255



    X_kaggle -= X_mean



    print("X_kaggle is %d bytes" % X_kaggle.nbytes)
if False:

    pred = sess.run(inference, feed_dict={x: X_kaggle, keep_prob: 1.0})

    print(pred)

    

    with open("prediction.txt", "wt") as f:

        f.write('"ImageId","Label"\n')

        for i in range(num_test_examples):

            f.write('%d,"%d"\n' % (i+1, pred[i]))    