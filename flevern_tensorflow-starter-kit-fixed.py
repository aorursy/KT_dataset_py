# These are all the modules we'll be using later

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from PIL import Image

import PIL

import pandas as pd

import numpy as np

import cv2

import os

import glob

import re

import tensorflow as tf



# Constants

TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'

IMAGE_SIZE = 150

CHANNELS = 3



# Sample sizes

TRAINING_SIZE = 1600

VALIDATION_SIZE = 400

TESTING_SIZE = 100



# CNN parameters

BATCH_SIZE = 16

NUM_HIDDEN = 32

DROPOUT = 0.75

LEARNING_RATE = 0.0001

NUM_STEPS = 1001
def norm_image(img):

    """

    Normalize PIL image

    

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch

    """

    img_y, img_b, img_r = img.convert('YCbCr').split()

    

    img_y_np = np.asarray(img_y).astype(float)



    img_y_np /= 255

    img_y_np -= img_y_np.mean()

    img_y_np /= img_y_np.std()

    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),

                    np.abs(np.percentile(img_y_np, 99.0))])

    img_y_np = img_y_np / scale

    img_y_np = np.clip(img_y_np, -1.0, 1.0)

    img_y_np = (img_y_np + 1.0) / 2.0

    

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)



    img_y = Image.fromarray(img_y_np)



    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    

    img_nrm = img_ybr.convert('RGB')

    

    return img_nrm
def resize_image(img, size):

    """

    Resize PIL image

    

    Resizes image to be square with sidelength size. Pads with black if needed.

    """

    # Resize

    n_x, n_y = img.size

    if n_y > n_x:

        n_y_new = size

        n_x_new = int(size * n_x / n_y + 0.5)

    else:

        n_x_new = size

        n_y_new = int(size * n_y / n_x + 0.5)



    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)



    # Pad the borders to create a square image

    img_pad = Image.new('RGB', (size, size), (128, 128, 128))

    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)

    img_pad.paste(img_res, ulc)



    return img_pad
def natural_key(string_):

    """

    Define sort key that is integer-aware

    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
train_cats = np.array(sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key))

train_dogs = np.array(sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key))



test_all = np.array(sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')), key=natural_key))
def read_images(images):

    """ Load image data into a useful structure. """

    count = len(images)

    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)



    for i, image_file in enumerate(images):

        # Normalize and resize the image

        img = Image.open(image_file)

        img_nrm = norm_image(img)

        img_res = resize_image(img_nrm, IMAGE_SIZE)

        # Store it with a useful format

        img_data = np.array(img_res, dtype=np.float32) 

        data[i] = img_data

    return data
np.random.seed(133)

def randomize(dataset, labels=None):

    permutation = np.random.permutation(len(dataset))

    shuffled_dataset = dataset[permutation]

    if labels is not None:

        shuffled_labels = labels[permutation]

        return shuffled_dataset, shuffled_labels

    return shuffled_dataset
# Training dataset

train_all = np.append(randomize(train_cats), randomize(train_dogs))



half_train_size = int(TRAINING_SIZE / 2)

mid_train = int(len(train_all) / 2)



train_images = np.append(train_all[:half_train_size], train_all[mid_train:mid_train+half_train_size])

train_dataset = read_images(train_images)

train_labels = np.append(np.ones(half_train_size), np.zeros(half_train_size))

train_labels = (np.arange(2) == train_labels[:,None]).astype(np.float32)



# Validation dataset

valid_all = np.append(train_all[half_train_size:mid_train], train_all[mid_train+half_train_size:])

valid_labels_all = np.append(np.ones(mid_train - half_train_size), np.zeros(mid_train - half_train_size))

valid_images, valid_images_labels = randomize(valid_all, valid_labels_all)

valid_dataset = read_images(valid_images[:VALIDATION_SIZE])

valid_labels = valid_images_labels[:VALIDATION_SIZE]

valid_labels = (np.arange(2) == valid_labels[:,None]).astype(np.float32)



# Testing dataset

test_images = test_all[:TESTING_SIZE]

test_dataset = read_images(test_images)



print("Train shape:", train_dataset.shape, train_labels.shape)

print("Valid shape:", valid_dataset.shape, valid_labels.shape)

print("Test shape:", test_dataset.shape)
plt.imshow(train_dataset[0,:,:,:], interpolation='nearest')

plt.figure()

plt.imshow(train_dataset[1000,:,:,:], interpolation='nearest')

plt.figure()

plt.imshow(valid_dataset[0,:,:,:], interpolation='nearest')

plt.figure()

plt.imshow(valid_dataset[1,:,:,:], interpolation='nearest')
graph = tf.Graph()



with graph.as_default():

    # Input data

    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 2))

    tf_valid_dataset = tf.constant(valid_dataset)

    tf_test_dataset = tf.constant(test_dataset)



    # Variables 

    HALF_HIDDEN = int(NUM_HIDDEN / 2)

    kernel_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, HALF_HIDDEN], dtype=tf.float32, stddev=1e-1), name='weights_conv1')

    biases_conv1 = tf.Variable(tf.constant(0.0, shape=[HALF_HIDDEN], dtype=tf.float32), trainable=True, name='biases_conv1')

    kernel_conv2 = tf.Variable(tf.truncated_normal([3, 3, HALF_HIDDEN, HALF_HIDDEN], dtype=tf.float32, stddev=1e-1), name='weights_conv2')

    biases_conv2 = tf.Variable(tf.constant(0.0, shape=[HALF_HIDDEN], dtype=tf.float32), trainable=True, name='biases_conv2')

    kernel_conv3 = tf.Variable(tf.truncated_normal([3, 3, HALF_HIDDEN, NUM_HIDDEN], dtype=tf.float32, stddev=1e-1), name='weights_conv3')

    biases_conv3 = tf.Variable(tf.constant(0.0, shape=[NUM_HIDDEN], dtype=tf.float32), trainable=True, name='biases_conv3')

  

    fc1w = tf.Variable(tf.truncated_normal([11552, NUM_HIDDEN], dtype=tf.float32, stddev=1e-1), name='weights') 

    fc1b = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN], dtype=tf.float32), trainable=True, name='biases')

    fc2w = tf.Variable(tf.truncated_normal([NUM_HIDDEN, 2], dtype=tf.float32, stddev=1e-1), name='weights')

    fc2b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32), trainable=True, name='biases')

 

    def model(data):

        parameters = []

        with tf.name_scope('conv1_1') as scope:

            conv = tf.nn.conv2d(data, kernel_conv1, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases_conv1)

            conv1_1 = tf.nn.relu(out, name=scope)

            conv1_1 = tf.nn.dropout(conv1_1, DROPOUT)

            parameters += [kernel_conv1, biases_conv1]

         

        # pool1

        pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.name_scope('conv2_1') as scope:

            conv = tf.nn.conv2d(pool1, kernel_conv2, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases_conv2)

            conv2_1 = tf.nn.relu(out, name=scope)

            conv2_1 = tf.nn.dropout(conv2_1, DROPOUT)

            parameters += [kernel_conv2, biases_conv2]

         

        # pool2

        pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.name_scope('conv3_1') as scope:

            conv = tf.nn.conv2d(pool2, kernel_conv3, [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, biases_conv3)

            conv3_1 = tf.nn.relu(out, name=scope)

            conv3_1 = tf.nn.dropout(conv3_1, DROPOUT)

            parameters += [kernel_conv3, biases_conv3]

         

        # pool3

        pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

         

        # fc1

        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(pool3.get_shape()[1:])) # except for batch size (the first one), multiple the dimensions

            pool3_flat = tf.reshape(pool3, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)

            fc1 = tf.nn.relu(fc1l)

            parameters += [fc1w, fc1b]



        # fc3

        with tf.name_scope('fc3') as scope:

            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)

            parameters += [fc2w, fc2b]

            

        return fc2l

  

    # Loss function

    logits = model(tf_train_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    

    # Optimizer

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

  

    # Predictions for the training, validation, and test data.

    train_prediction = tf.nn.softmax(logits)

    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))

    test_prediction = tf.nn.softmax(model(tf_test_dataset))
def accuracy(predictions, labels):

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
# Create a TensorFlow session

with tf.Session(graph=graph) as sess:



    # Initialize the variables

    tf.initialize_all_variables().run()



    # Training loop

    for step in range(NUM_STEPS):

        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

        batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]

        batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 50 == 0:

            print("Minibatch loss at step", step, ":", l)

            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

            

    result = test_prediction.eval()
# TODO