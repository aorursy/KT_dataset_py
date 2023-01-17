import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import tensorflow as tf

#Variables initialization



# image number to output

IMAGE_TO_DISPLAY = 20

# split for train and validation set

VALIDATION_SIZE = 2000

num_channels = 1 # grayscale



# hyperparameters

lr = 1e-4

training_iters = 1000

batch_size = 100

n_inputs = 28 # MNIST data input (img shape: 28*28)

patch_size = 5

depth = 32

n_steps = 28 # time steps

#n_hidden_units = 128 # neurons in hidden layer

n_classes = 10 # MNIST classes (0-9 digits)
# Data Preperation 

train = pd.read_csv('../input/train.csv')

#print (train.head())

test  = pd.read_csv("../input/test.csv")

#print(test)

images = train.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)

print('data size: (%g, %g)' % images.shape)



# Transforming 784 values of one image to 28*28 size

image_size = images.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square (28x28)

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))



#Test data preperation##

images_test = test.iloc[:,1:].values

images_test = images.astype(np.float)

images_test = np.multiply(images_test, 1.0 / 255.0)
#Display image

def display(img):

    # (784) => (28x28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# display an image

display(images[IMAGE_TO_DISPLAY])
# print information about image size, label of image-to-display and number of labels

labels_flat = train[[0]].values.ravel()

print('labels_flat ({0})'.format(len(labels_flat)))

print ('label of image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))



labels_count = np.unique(labels_flat).shape[0]

print('number of labels => {0}'.format(labels_count))
# convert class labels from scalars to one-hot vectors

# 0 => [1 0 0 0 0 0 0 0 0 0]

# 1 => [0 1 0 0 0 0 0 0 0 0]

# ...

# 9 => [0 0 0 0 0 0 0 0 0 1]



def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('labels({0[0]},{0[1]})'.format(labels.shape))

print ('labels vector for image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))
# split data into training & validation sets

validation_dataset = images[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_dataset = images[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]



print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', validation_dataset.shape, validation_labels.shape)

    
#Reformat datasets 

def reformat(dataset):

  dataset = dataset.reshape(

    (-1, image_width, image_height, num_channels)).astype(np.float32)



  return dataset

train_dataset = reformat(train_dataset)

valid_dataset = reformat(validation_dataset)

test_dataset = reformat(images_test)

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, validation_labels.shape)

print('Test set', test_dataset.shape)
# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, shape=(batch_size, n_inputs, n_inputs, num_channels)) # 28x28

ys = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])
####Methods####

def compute_accuracy(v_xs, v_ys):

    global prediction

    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result



def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)



def conv2d(x, W):

    # stride [1, x_movement, y_movement, 1]

    # Must have strides[0] = strides[3] = 1

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    # stride [1, x_movement, y_movement, 1]

    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

## conv1 layer ##

W_conv1 = weight_variable([patch_size,patch_size,num_channels,depth]) # patch 5x5, in size 1, out size 32

b_conv1 = bias_variable([depth])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32

h_pool1 = max_pool_2x2(h_conv1) # output size 14x14x32
## conv2 layer ##

W_conv2 = weight_variable([patch_size,patch_size,depth,depth]) # patch 5x5, in size 32, out size 32

b_conv2 = bias_variable([depth])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x32

h_pool2 = max_pool_2x2(h_conv2) # output size 7x7x32
## fully connected layer 1 ##

shape = h_pool2.get_shape().as_list()

print("shape",shape)

W_fc1 = weight_variable([shape[1] * shape[2] * shape[3], 1024])

b_fc1 = bias_variable([1024])

# [n_samples, 7, 7, 32] ->> [n_samples, 7*7*32]

h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



## fully connected layer 2 ##

W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# the error between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),

                                              reduction_indices=[1]))       # loss

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)



sess = tf.Session()

# important step

sess.run(tf.initialize_all_variables())



for i in range(training_iters):

    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]

    batch_labels = train_labels[offset:(offset + batch_size), :]

    

    sess.run(train_step, feed_dict={xs: batch_data, ys: batch_labels, keep_prob: 0.5})

    if i % 50 == 0:

        #print('Training accuracy',compute_accuracy(validation_dataset, validation_labels))

        print('Validation accuracy',compute_accuracy(valid_dataset[0:batch_size], validation_labels[0:batch_size]))
# the error between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),

                                              reduction_indices=[1]))       # loss

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)



sess = tf.Session()

# important step

sess.run(tf.initialize_all_variables())



for i in range(training_iters):

    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]

    batch_labels = train_labels[offset:(offset + batch_size), :]

    

    sess.run(train_step, feed_dict={xs: batch_data, ys: batch_labels, keep_prob: 0.5})

    if i % 50 == 0:

        #print('Training accuracy',compute_accuracy(validation_dataset, validation_labels))

        print('Validation accuracy',compute_accuracy(valid_dataset[0:batch_size], validation_labels[0:batch_size]))