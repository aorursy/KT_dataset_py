# Importing python libraries

import pickle

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import random

import cv2

import skimage.morphology as morp

from skimage.filters import rank

from sklearn.utils import shuffle

import csv

import os

import tensorflow as tf

from tensorflow.contrib.layers import flatten

from sklearn.metrics import confusion_matrix



# is it using the GPU?

print(tf.test.gpu_device_name())



# Show current TensorFlow version

tf.__version__
# Load the data with pickle

training_file = '../input/projet-trafficsign/test.p'

validation_file = '../input/projet-trafficsign/valid.p'

testing_file = '../input/projet-trafficsign/test.p'



with open(training_file, mode='rb') as f:

    train = pickle.load(f)

with open(validation_file, mode='rb') as f:

    valid = pickle.load(f)

with open(testing_file, mode='rb') as f:

    test = pickle.load(f)    

# Mapping ClassID to traffic sign names

signames_file = '../input/signames/signnames.csv'

signs = []

with open(signames_file, 'r') as csvfile:

    signnames = csv.reader(csvfile, delimiter=',')

    next(signnames,None)

    for row in signnames:

        signs.append(row[0])

    csvfile.close()
X_train, y_train = train['features'], train['labels']

X_valid, y_valid = valid['features'], valid['labels']

X_test, y_test = test['features'], test['labels']





# Number of training examples

n_train = X_train.shape[0]

# Number of testing examples

n_test = X_test.shape[0]

# Number of validation examples.

n_validation = X_valid.shape[0]



# What's the shape of an traffic sign image?

image_shape = X_train[0].shape



# How many unique classes/labels there are in the dataset.

n_classes = len(np.unique(y_train))

print("Number of training examples: ", n_train)

print("Number of testing examples: ", n_test)

print("Number of validation examples: ", n_validation)

print("Image data shape =", image_shape)

print("Number of classes =", n_classes)



def list_images(dataset_X, dataset_y, ylabel="", cmap=None):

    """

    Display a list of images in a single figure with matplotlib.

        Parameters:

            images: An np.array compatible with plt.imshow.

            lanel (Default = No label): A string to be used as a label for each image.

            cmap (Default = None): Used to display gray images.

    """

    plt.figure(figsize=(15, 16))

    for i in range(6):

        plt.subplot(1, 6, i+1)

        indx = random.randint(0, len(dataset_X))

        #Use gray scale color map if there is only one channel

        cmap = 'gray' if len(dataset_X[indx].shape) == 2 else cmap

        plt.imshow(dataset_X[indx], cmap = cmap)

        plt.xlabel(signs[dataset_y[indx]])

        plt.ylabel(ylabel)

        plt.xticks([])

        plt.yticks([])

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.show()



    
# Plotting sample examples

list_images(X_train, y_train, "Training example")

list_images(X_test, y_test, "Testing example")

list_images(X_valid, y_valid, "Validation example")
def histogram_plot(dataset, label):

    """

    Plots a histogram of the input data.

        Parameters:

            dataset: Input data to be plotted as a histogram.

            label: A string to be used as a label for the histogram.

    """

    hist, bins = np.histogram(dataset, bins=n_classes)

    width = 0.7 * (bins[1] - bins[0])

    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, hist, align='center', width=width)

    plt.xlabel(label)

    plt.ylabel("Image count")

    plt.show()
# Plotting histograms of the count of each sign

histogram_plot(y_train, "Training examples")

histogram_plot(y_test, "Testing examples")

histogram_plot(y_valid, "Validation examples")
#Shuffling Data

X_train, y_train = shuffle(X_train, y_train)
#Grayscaling Image

def gray_scale(image):

    """

    Convert images to gray scale.

        Parameters:

            image: An np.array compatible with plt.imshow.

    """

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Sample images after greyscaling

gray_images = list(map(gray_scale, X_train))

list_images(gray_images, y_train, "Gray Scale image", "gray")
#Local Histogram Equalization

def local_histo_equalize(image):

    """

    Apply local histogram equalization to grayscale images.

        Parameters:

            image: A grayscale image.

    """

    kernel = morp.disk(30)

    img_local = rank.equalize(image, selem=kernel)

    return img_local
# Sample images after Local Histogram Equalization

equalized_images = list(map(local_histo_equalize, gray_images))

list_images(equalized_images, y_train, "Equalized Image", "gray")
#Normalization

def image_normalize(image):

    """

    Normalize images to [0, 1] scale.

        Parameters:

            image: An np.array compatible with plt.imshow.

    """

    image = np.divide(image, 255)

    return image
# Sample images after normalization

n_training = X_train.shape

normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))

for i, img in enumerate(equalized_images):

    normalized_images[i] = image_normalize(img)

list_images(normalized_images, y_train, "Normalized Image", "gray")

normalized_images = normalized_images[..., None]
def preprocess(data):

    """

    Applying the preprocessing steps to the input data.

        Parameters:

            data: An np.array compatible with plt.imshow.

    """

    gray_images = list(map(gray_scale, data))

    equalized_images = list(map(local_histo_equalize, gray_images))

    n_training = data.shape

    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))

    for i, img in enumerate(equalized_images):

        normalized_images[i] = image_normalize(img)

    normalized_images = normalized_images[..., None]

    return normalized_images
class LaNet:  



    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):

        # Hyperparameters

        self.mu = mu

        self.sigma = sigma



        # Layer 1 (Convolutional): Input = 32x32x1. Output = 28x28x6.

        self.filter1_width = 5

        self.filter1_height = 5

        self.input1_channels = 1

        self.conv1_output = 6

        # Weight and bias

        self.conv1_weight = tf.Variable(tf.truncated_normal(

            shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output),

            mean = self.mu, stddev = self.sigma))

        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))

        # Apply Convolution

        self.conv1 = tf.nn.conv2d(x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias

        

        # Activation:

        self.conv1 = tf.nn.relu(self.conv1)

        

        # Pooling: Input = 28x28x6. Output = 14x14x6.

        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        

        # Layer 2 (Convolutional): Output = 10x10x16.

        self.filter2_width = 5

        self.filter2_height = 5

        self.input2_channels = 6

        self.conv2_output = 16

        # Weight and bias

        self.conv2_weight = tf.Variable(tf.truncated_normal(

            shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),

            mean = self.mu, stddev = self.sigma))

        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))

        # Apply Convolution

        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_bias

        

        # Activation:

        self.conv2 = tf.nn.relu(self.conv2)

        

        # Pooling: Input = 10x10x16. Output = 5x5x16.

        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        

        # Flattening: Input = 5x5x16. Output = 400.

        self.fully_connected0 = flatten(self.conv2)

        

        # Layer 3 (Fully Connected): Input = 400. Output = 120.

        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = self.mu, stddev = self.sigma))

        self.connected1_bias = tf.Variable(tf.zeros(120))

        self.fully_connected1 = tf.add((tf.matmul(self.fully_connected0, self.connected1_weights)), self.connected1_bias)

        

        # Activation:

        self.fully_connected1 = tf.nn.relu(self.fully_connected1)

    

        # Layer 4 (Fully Connected): Input = 120. Output = 84.

        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = self.mu, stddev = self.sigma))

        self.connected2_bias = tf.Variable(tf.zeros(84))

        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)

        

        # Activation.

        self.fully_connected2 = tf.nn.relu(self.fully_connected2)

    

        # Layer 5 (Fully Connected): Input = 84. Output = 43.

        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = self.mu, stddev = self.sigma))

        self.output_bias = tf.Variable(tf.zeros(43))

        self.logits =  tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)



        # Training operation

        self.one_hot_y = tf.one_hot(y, n_out)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)

        self.loss_operation = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        self.training_operation = self.optimizer.minimize(self.loss_operation)



        # Accuracy operation

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))

        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



        # Saving all variables

        self.saver = tf.train.Saver()

    

    def y_predict(self, X_data, BATCH_SIZE=64):

        num_examples = len(X_data)

        y_pred = np.zeros(num_examples, dtype=np.int32)

        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):

            batch_x = X_data[offset:offset+BATCH_SIZE]

            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 

                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})

        return y_pred

    

    def evaluate(self, X_data, y_data, BATCH_SIZE=64):

        num_examples = len(X_data)

        total_accuracy = 0

        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):

            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

            accuracy = sess.run(self.accuracy_operation, 

                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })

            total_accuracy += (accuracy * len(batch_x))

        return total_accuracy / num_examples
#Model Training and Evaluation

x = tf.placeholder(tf.float32, (None, 32, 32, 1))

y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers

keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers
# Validation set preprocessing

X_valid_preprocessed = preprocess(X_valid)
EPOCHS = 30

BATCH_SIZE = 64

DIR = 'Saved_Models'
LeNet_Model = LaNet(n_out = n_classes)

model_name = "LeNet"



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    num_examples = len(y_train)

    print("Training ...")

    print()

    for i in range(EPOCHS):

        normalized_images, y_train = shuffle(normalized_images, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):

            end = offset + BATCH_SIZE

            batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]

            sess.run(LeNet_Model.training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})

            

        validation_accuracy = LeNet_Model.evaluate(X_valid_preprocessed, y_valid)

        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))

    LeNet_Model.saver.save(sess, os.path.join(DIR, model_name))

    print("Model saved")
class VGGnet:  



    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):

        # Hyperparameters

        self.mu = mu

        self.sigma = sigma



        # Layer 1 (Convolutional): Input = 32x32x1. Output = 32x32x32.

        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = self.mu, stddev = self.sigma))

        self.conv1_b = tf.Variable(tf.zeros(32))

        self.conv1   = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b



        # ReLu Activation.

        self.conv1 = tf.nn.relu(self.conv1)



        # Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.

        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = self.mu, stddev = self.sigma))

        self.conv2_b = tf.Variable(tf.zeros(32))

        self.conv2   = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b



        # ReLu Activation.

        self.conv2 = tf.nn.relu(self.conv2)



        # Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.

        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)



        # Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.

        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = self.mu, stddev = self.sigma))

        self.conv3_b = tf.Variable(tf.zeros(64))

        self.conv3   = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b



        # ReLu Activation.

        self.conv3 = tf.nn.relu(self.conv3)



        # Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.

        self.conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = self.mu, stddev = self.sigma))

        self.conv4_b = tf.Variable(tf.zeros(64))

        self.conv4   = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b



        # ReLu Activation.

        self.conv4 = tf.nn.relu(self.conv4)



        # Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.

        self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv) # dropout



        # Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.

        self.conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = self.mu, stddev = self.sigma))

        self.conv5_b = tf.Variable(tf.zeros(128))

        self.conv5   = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b



        # ReLu Activation.

        self.conv5 = tf.nn.relu(self.conv5)



        # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.

        self.conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = self.mu, stddev = self.sigma))

        self.conv6_b = tf.Variable(tf.zeros(128))

        self.conv6   = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b



        # ReLu Activation.

        self.conv6 = tf.nn.relu(self.conv6)



        # Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.

        self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv) # dropout



        # Flatten. Input = 4x4x128. Output = 2048.

        self.fc0   = flatten(self.conv6)



        # Layer 10 (Fully Connected): Input = 2048. Output = 128.

        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = self.mu, stddev = self.sigma))

        self.fc1_b = tf.Variable(tf.zeros(128))

        self.fc1   = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b



        # ReLu Activation.

        self.fc1    = tf.nn.relu(self.fc1)

        self.fc1    = tf.nn.dropout(self.fc1, keep_prob) # dropout



        # Layer 11 (Fully Connected): Input = 128. Output = 128.

        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = self.mu, stddev = self.sigma))

        self.fc2_b  = tf.Variable(tf.zeros(128))

        self.fc2    = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b



        # ReLu Activation.

        self.fc2    = tf.nn.relu(self.fc2)

        self.fc2    = tf.nn.dropout(self.fc2, keep_prob) # dropout



        # Layer 12 (Fully Connected): Input = 128. Output = n_out.

        self.fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean = self.mu, stddev = self.sigma))

        self.fc3_b  = tf.Variable(tf.zeros(n_out))

        self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b



        # Training operation

        self.one_hot_y = tf.one_hot(y, n_out)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)

        self.loss_operation = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

        self.training_operation = self.optimizer.minimize(self.loss_operation)



        # Accuracy operation

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))

        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



        # Saving all variables

        self.saver = tf.train.Saver()

        

    def y_predict(self, X_data, BATCH_SIZE=64):

        num_examples = len(X_data)

        y_pred = np.zeros(num_examples, dtype=np.int32)

        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):

            batch_x = X_data[offset:offset+BATCH_SIZE]

            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 

                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})

        return y_pred

    

    def evaluate(self, X_data, y_data, BATCH_SIZE=64):

        num_examples = len(X_data)

        total_accuracy = 0

        sess = tf.get_default_session()

        for offset in range(0, num_examples, BATCH_SIZE):

            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

            accuracy = sess.run(self.accuracy_operation, 

                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })

            total_accuracy += (accuracy * len(batch_x))

        return total_accuracy / num_examples

#VGGNet Model

VGGNet_Model = VGGnet(n_out = n_classes)

model_name = "VGGNet"



# Validation set preprocessing

X_valid_preprocessed = preprocess(X_valid)

one_hot_y_valid = tf.one_hot(y_valid, 43)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    num_examples = len(y_train)

    print("Training...")

    print()

    for i in range(EPOCHS):

        normalized_images, y_train = shuffle(normalized_images, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):

            end = offset + BATCH_SIZE

            batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]

            sess.run(VGGNet_Model.training_operation, 

            feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})



        validation_accuracy = VGGNet_Model.evaluate(X_valid_preprocessed, y_valid)

        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))

    VGGNet_Model.saver.save(sess, os.path.join(DIR, model_name))

    print("Model saved")
# Test set preprocessing

X_test_preprocessed = preprocess(X_test)
with tf.Session() as sess:

    VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))

    y_pred = VGGNet_Model.y_predict(X_test_preprocessed)

    test_accuracy = sum(y_test == y_pred)/len(y_test)

    print("Test Accuracy = {:.1f}%".format(test_accuracy*100))
cm = confusion_matrix(y_test, y_pred)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

cm = np.log(.0001 + cm)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Log of normalized Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()