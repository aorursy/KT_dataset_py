import numpy as np

import pandas as pd

import tensorflow as tf

import os, random, cv2

import math



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



# image number to output

IMAGE_TO_DISPLAY = 4



IMAGE_SIZE = 64

IMG_ROW = 64

IMG_COL = 64

IMG_CHA = 3



LEARNING_RATE = 1e-10

TRAINING_ITERATIONS = 500

#EXTRA_ITERATIONS = 4000



DROPOUT = .5

BATCH_SIZE = 50

#EXTRA_BATCH_SIZE = 400



TRAIN_SIZE = 100 #Set to 25000 for entire set, validation is cut from here

TEST_SIZE = 50 #Set to 12500 for entire set

VALIDATION_SIZE = 50



FILTER_SIZE = 5 #Produces NxN filters



FILTER_NUM_1 = 32 #Number of filters at given layer

FILTER_NUM_2 = 32

FILTER_NUM_3 = 64

FILTER_NUM_4 = 64

FILTER_NUM_5 = 128

FILTER_NUM_6 = 128



train_images_dir = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]

train_dogs_dir =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats_dir =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images_dir =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]



random.shuffle(train_images_dir)
def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if (img.shape[0] >= img.shape[1]): # height is greater than width

       resizeto = (IMAGE_SIZE, int (round (IMAGE_SIZE * (float (img.shape[1])  / img.shape[0]))));

    else:

       resizeto = (int (round (IMAGE_SIZE * (float (img.shape[0])  / img.shape[1]))), IMAGE_SIZE);

    

    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)

    img3 = cv2.copyMakeBorder(img2, 0, IMAGE_SIZE - img2.shape[0], 0, IMAGE_SIZE - img2.shape[1], cv2.BORDER_CONSTANT, 0)

        

    return img3[:,:,::-1]  # turn into rgb format



def prep_data(images):

    count = len(images)

    data = np.ndarray((count, IMAGE_SIZE, IMAGE_SIZE, IMG_CHA), dtype=np.float32)

    labels = np.zeros((len(images),2))



    for i, image_file in enumerate(images):

        image = read_image(image_file);

        image_data = np.array (image, dtype=np.float32);

        image_data[:,:,0] = (image_data[:,:,0].astype(float)) / 255-.5

        image_data[:,:,1] = (image_data[:,:,1].astype(float)) / 255-.5

        image_data[:,:,2] = (image_data[:,:,2].astype(float)) / 255-.5

        if('/dog' in image_file):

            labels[i,0] = 1

        else:

            labels[i,1] = 1

        

        data[i] = image_data; # image_data.T

        if i%250 == 0 or (i+1)==count: print('Processed {} of {}'.format(i, count))    

    data = np.resize(data.T, [len(images),(IMG_ROW*IMG_COL), 3])

    return data, labels
train, labels = prep_data(train_images_dir[:TRAIN_SIZE])

test, test_labels = prep_data(test_images_dir[:TEST_SIZE])

# split data into training & validation

validation_images = train[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = train[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]
# xavier initialization

def weight_variable(shape,n_in,n_out):

    initial = tf.truncated_normal(shape, stddev=math.sqrt(2. / (n_in + n_out)))

    return tf.Variable(initial)



def bias_variable(shape,n_in,n_out):

    initial = tf.constant(math.sqrt(2. / (n_in + n_out)), shape=shape)

    return tf.Variable(initial)



# convolution

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# input & output of NN



# images

x = tf.placeholder('float', shape=[None, train.shape[1], train.shape[2]], )

# labels

y_ = tf.placeholder('float', shape=[None, 2])



def lrelu(x):

  return tf.nn.relu(x) - .01 * tf.nn.relu(-x)



image_size = train.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
# first convolutional layer

W_conv1 = weight_variable([FILTER_SIZE, FILTER_SIZE, IMG_CHA, FILTER_NUM_1], IMG_CHA, FILTER_NUM_1)

b_conv1 = bias_variable([FILTER_NUM_1], IMG_CHA, FILTER_NUM_1)



image = tf.reshape(x, [-1, image_width, image_height, IMG_CHA])



h_conv1 = lrelu(conv2d(image, W_conv1) + b_conv1)

h_pool1 = h_conv1
# second convolutional layer

W_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM_1, FILTER_NUM_2], FILTER_NUM_1, FILTER_NUM_2)

b_conv2 = bias_variable([FILTER_NUM_2], FILTER_NUM_1, FILTER_NUM_2)



h_conv2 = lrelu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
# third convolutional layer

W_conv3 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM_2, FILTER_NUM_3], FILTER_NUM_2, FILTER_NUM_3)

b_conv3 = bias_variable([FILTER_NUM_3], FILTER_NUM_2, FILTER_NUM_3)



h_conv3 = lrelu(conv2d(h_pool2, W_conv3) + b_conv3)

h_pool3 = h_conv3
# fourth convolutional layer

W_conv4 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM_3, FILTER_NUM_4], FILTER_NUM_3, FILTER_NUM_4)

b_conv4 = bias_variable([FILTER_NUM_4], FILTER_NUM_3, FILTER_NUM_4)



h_conv4 = lrelu(conv2d(h_pool3, W_conv4) + b_conv4)

h_pool4 = max_pool_2x2(h_conv4)
# fifth convolutional layer

W_conv5 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM_4, FILTER_NUM_5], FILTER_NUM_4, FILTER_NUM_5)

b_conv5 = bias_variable([FILTER_NUM_5], FILTER_NUM_4, FILTER_NUM_5)



h_conv5 = lrelu(conv2d(h_pool4, W_conv5) + b_conv5)

h_pool5 = h_conv5
# sixth convolutional layer

W_conv6 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_NUM_5, FILTER_NUM_6], FILTER_NUM_5, FILTER_NUM_6)

b_conv6 = bias_variable([FILTER_NUM_6], FILTER_NUM_5, FILTER_NUM_6)



h_conv6 = lrelu(conv2d(h_pool5, W_conv6) + b_conv6)

h_pool6 = max_pool_2x2(h_conv6)

# densely connected layer

W_fc1 = weight_variable([int(image_width/8 * image_height/8 * FILTER_NUM_6), 1024], FILTER_NUM_6, 1024)

b_fc1 = bias_variable([1024], FILTER_NUM_6, 1024)



h_pool6_flat = tf.reshape(h_pool6, [-1, int(image_width/8 * image_height/8 * FILTER_NUM_6)])



h_fc1 = lrelu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
# dropout

keep_prob = tf.placeholder('float')

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer for deep net

W_fc2 = weight_variable([1024, 2], 1024, 2)

b_fc2 = bias_variable([2], 1024, 2)



y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# cost function

cross_entropy = -tf.reduce_sum(y_*tf.log(y))



# optimisation function

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)



# evaluation

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)
epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all training data have been already used, it is reordered randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return train_images[start:end], train_labels[start:end]
# start TensorFlow session

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()



sess.run(init)
# visualisation variables

train_accuracies = []

validation_accuracies = []

x_range = []



display_step=1



for i in range(TRAINING_ITERATIONS):



    #get new batch

    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step

    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:

        

        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 

                                                  y_:batch_ys, 

                                                  keep_prob: 1.0})

        if(VALIDATION_SIZE):

            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 

                                                            y_: validation_labels[0:BATCH_SIZE], 

                                                            keep_prob: 1.0})                                  

            print('training_accuracy / validation_accuracy / epoch=> %.2f / %.2f / %i for step %d'%(train_accuracy, validation_accuracy, epochs_completed, i))

            

            validation_accuracies.append(validation_accuracy)

            

        else:

             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))

        train_accuracies.append(train_accuracy)

        x_range.append(i)

        

        # increase display_step, max 1000

        if i%(display_step*10) == 0 and i and display_step < 1000:

            display_step *= 10

    # train on batch

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

    