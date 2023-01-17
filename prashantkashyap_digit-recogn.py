# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

### Import all the modules

from sklearn.utils import resample

from zipfile import ZipFile

import gzip

import os

from urllib.request import urlretrieve

import csv

import numpy as np

from PIL import Image

import tensorflow as tf

from scipy.misc import imsave

from sklearn.model_selection import train_test_split



import tensorflow as tf

from random import randint

import math

from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

from numpy import prod

#import pre_processing 



# Network Parameters

IMAGE_SIZE = 28

# All the pixels in the image (28 * 28 = 784)

features_count = 784 # MNIST data input (img shape: 28*28)

# All the labels

labels_count = 10 # MNIST total classes (0-9 digits)

# Change if have a memory restrictions

batch_size = 4096

# TODO: Find the best parameters for each configuration

epochs = 10

keep_probability = 1

# Model save path

save_model_path = './cnn_digit_recognition'

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

WORK_DIRECTORY = '../input'

TRAIN_IMAGE = '../input/train-image/'

TEST_IMAGE = '../input/test-image/'

NUM_CHANNELS = 1

PIXEL_DEPTH = 255

NUM_LABELS = 10



print ('Successfully Import of library ')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





# Utility Functions

# Download script to download datasets if not already

"""

code to download the nmist data set if not already downloded Data set will be downloded at "currnt dirctory/mnist" dirctory unless we modify the WORK_DIRECTORY.

"""

def download_dataset(filename):

  print('downloading',filename)

  """Download the data from Yann's website, unless it's already here."""

  if not tf.gfile.Exists(WORK_DIRECTORY):

    tf.gfile.MakeDirs(WORK_DIRECTORY)

    print('...')

  filepath = os.path.join(WORK_DIRECTORY, filename)

  if not tf.gfile.Exists(filepath):

    filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)

    with tf.gfile.GFile(filepath) as f:

      size = f.size()

    print('Successfully downloaded', filename, size, 'bytes.')

  return filepath



"""

Extract routine to extract data and labels from dowloded files and save it into numpy array

"""

def extract_data(filename, num_images):

  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].

  """

  print('Extracting', filename)

  with gzip.open(filename) as bytestream:

    bytestream.read(16)

    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)

    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH

    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)

    

    # Load image data as 1 dimensional array

    data = np.array(data, dtype=np.float32).flatten()

    # reshape the image into [image imdex, image_size*image_size*channels]

    data = data.reshape(num_images,IMAGE_SIZE*IMAGE_SIZE*1)

    return data





def extract_labels(filename, num_images):

  """Extract the labels into a vector of int64 label IDs."""

  print('Extracting', filename)

  with gzip.open(filename) as bytestream:

    bytestream.read(8)

    buf = bytestream.read(1 * num_images)

    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    labels = labels.reshape(num_images,1)

  return labels



# Call download routing to download the mnist data files

def get_raw_data():

    """

    : return: will return np array of train_data, train_labels, test_data, test_labels 

              from mnist data set

              

    """

    print('get_data')

    train_data_filename = download_dataset('train-images-idx3-ubyte.gz')

    train_labels_filename = download_dataset('train-labels-idx1-ubyte.gz')

    test_data_filename = download_dataset('t10k-images-idx3-ubyte.gz')

    test_labels_filename = download_dataset('t10k-labels-idx1-ubyte.gz')



    # Extract train_data_filename and save into np arrays upto 60000 images.

    train_data = extract_data(train_data_filename, 60000)

#    display(train_data.shape)

    # Extract train_labels_filename and save into np arrays upto 60000 images.

    train_labels = extract_labels(train_labels_filename, 60000)

#    display(train_labels.shape)



    # Extract test_data_filename and save into np arrays upto 10000 images.

    test_data = extract_data(test_data_filename, 10000)

    # Extract test_labels_filename and save into np arrays upto 10000 images.

    test_labels = extract_labels(test_labels_filename, 10000)



#    display(test_data[0:])

    # Wait until you see that all features and labels have been uncompressed.

    print('All features and labels uncompressed.')

   # data_save(train_data, train_labels, test_data, test_labels)

    

    return train_data, train_labels, test_data, test_labels



"""

Save the digit in .jpg format in TRAIN-IMAGE and TEST-IMAGE directory 

Also creating a .csv file to save lables

"""

def data_save(train_data, train_labels, test_data, test_labels):

    if not os.path.isdir(TRAIN_IMAGE):

       os.makedirs(TRAIN_IMAGE)



    if not os.path.isdir(TEST_IMAGE):

       os.makedirs(TEST_IMAGE)



    # process train data

    with open("mnist/train-labels.csv", 'w') as csvFile:

      writer = csv.writer(csvFile, delimiter=',', quotechar='"')

      for i in range(len(train_data)):

        img = train_data[i]

        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)

       # print (img.shape)

        imsave(TRAIN_IMAGE + str(i) + ".jpg", img)

        writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])

    print('training-data saving done')

    # repeat for test data

    with open("mnist/test-labels.csv", 'w') as csvFile:

      writer = csv.writer(csvFile, delimiter=',', quotechar='"')

      for i in range(len(test_data)):

        img = test_data[i]

        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE)

        imsave(TEST_IMAGE + str(i) + ".jpg", img)

        writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])

    print('test-data saving done')

    

### Label encoding routine.

def label_encoding(x):

    y = np.zeros((len(x), 10))

    for i in range(len(x)):

        y[i,x[i]] = 1

    return y



"""

This routine is to get the training, cross validation and test data as well as encoded label 

from mnist dataset

"""

def get_processed_data():

    print ('data processing')

    train_data, train_labels, test_data, test_labels = get_raw_data()

    train_label_encoded = label_encoding(train_labels)

    test_label_encoded = label_encoding(test_labels)

    # Get randomized datasets for training and validation

    train_features, valid_features, train_labels, valid_labels = train_test_split(

        train_data,

        train_label_encoded,

        test_size=0.05,

        random_state=0)

#    display(train_features)



    print('Training features and labels randomized and split.')

#    display(test_label_encoded)

    return train_features, valid_features, train_labels, valid_labels, test_data, test_label_encoded

def get_data():

    return get_processed_data()
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):

    """

    Apply convolution then max pooling to x_tensor

    :param x_tensor: TensorFlow Tensor

    :param conv_num_outputs: Number of outputs for the convolutional layer

    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer

    :param conv_strides: Stride 2-D Tuple for convolution

    :param pool_ksize: kernal size 2-D Tuple for pool

    :param pool_strides: Stride 2-D Tuple for pool

    : return: A tensor that represents convolution and max pooling of x_tensor

    """

    # TODO: Implement Function

    dimension = x_tensor.get_shape().as_list()

    shape = list(conv_ksize + (dimension[-1],) + (conv_num_outputs,))

    #print(shape)

    filter_weights = tf.Variable(tf.truncated_normal(shape,0,0.1)) # (height, width, input_depth, output_depth)

    filter_bias = tf.Variable(tf.zeros(conv_num_outputs))

    padding = 'SAME'

    #print(list((1,)+conv_strides+(1,)))

    #print(filter_weights)

    conv_layer = tf.nn.conv2d(x_tensor, filter_weights, list((1,)+conv_strides+(1,)), padding)

    conv_layer = tf.nn.bias_add(conv_layer, filter_bias)

    

    conv_layer = tf.nn.relu(conv_layer)

    

    conv_layer = tf.nn.max_pool(

        conv_layer,

        ksize=[1] + list(pool_ksize) + [1],

        strides=[1] + list(pool_strides) + [1],

        padding='SAME')

    

    return conv_layer

def flatten_image(x_tensor):

    """

    Flatten x_tensor to (Batch Size, Flattened Image Size)

    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.

    : return: A tensor of size (Batch Size, Flattened Image Size).

    """

    # TODO: Implement Function

    #return None

    dimension = x_tensor.get_shape().as_list()  

    print(dimension)

    x =  tf.reshape(x_tensor,[-1,prod(dimension[1:])])

    print(x.get_shape().as_list())

    return x
def fully_connected(x_tensor, num_outputs):

    """

    Apply a fully connected layer to x_tensor using weight and bias

    : x_tensor: A 2-D tensor where the first dimension is batch size.

    : num_outputs: The number of output that the new tensor should be.

    : return: A 2-D tensor where the second dimension is num_outputs.

    """

    # TODO: Implement Function

    dimension = x_tensor.get_shape().as_list()

    shape = list( (dimension[-1],) + (num_outputs,))

    print(x_tensor.get_shape())

    weight = tf.Variable(tf.truncated_normal(shape,0,0.1))

    bias = tf.Variable(tf.zeros(num_outputs))

    print (weight.get_shape(), bias.get_shape())

    return tf.nn.relu(tf.add(tf.matmul(x_tensor,weight), bias))

 
def output(x_tensor, num_outputs):

    """

    Apply a output layer to x_tensor using weight and bias

    : x_tensor: A 2-D tensor where the first dimension is batch size.

    : num_outputs: The number of output that the new tensor should be.

    : return: A 2-D tensor where the second dimension is num_outputs.

    """

    # TODO: Implement Function

    dimension = x_tensor.get_shape().as_list()

    shape = list( (dimension[-1],) + (num_outputs,))

    print(shape)

    weight = tf.Variable(tf.truncated_normal(shape,0,0.01))

    bias = tf.Variable(tf.zeros(num_outputs))



    return tf.add(tf.matmul(x_tensor,weight), bias)

def convolutional_net(x, keep_prob):

    """

    Create a convolutional neural network model

    : x: Placeholder tensor that holds image data.

    : keep_prob: Placeholder tensor that hold dropout keep probability.

    : return: Tensor model that represents logits.

    """

    #    Play around with different number of outputs, kernel size and stride

    # Function Definition from Above:

    model = conv2d_maxpool(x, conv_num_outputs=18, conv_ksize=(4,4), conv_strides=(1,1), pool_ksize=(8,8), pool_strides=(1,1))

    model = tf.nn.dropout(model, keep_prob)    



    model = flatten_image(model)

    #    Play around with different number of outputs

    model = fully_connected(model,200)

    

    model = tf.nn.dropout(model, keep_prob)

    

    #    Set this to the number of classes

    #   output(x_tensor, num_outputs)

    model = output(model,10)

    return model



##############################

## Build the Neural Network ##

##############################

def cnn_model():

    """

    return: features, labels, keep_prob, logits, cost, accuracy, optimizer of 

              convolutinal neural network model

    """

    # Remove previous weights, bias, inputs, etc..

    tf.reset_default_graph()

    # Inputs

    image_shape = (IMAGE_SIZE,IMAGE_SIZE,1)

    features = tf.placeholder(tf.float32, [None] + list(image_shape), name="features")

    labels = tf.placeholder(tf.float32, (None, labels_count), name="label")

    keep_prob = tf.placeholder(tf.float32, name="keep_probability")



    # Model

    logits = convolutional_net(features, keep_prob)



    # Name logits Tensor, so that is can be loaded from disk after training

    logits = tf.identity(logits, name='logits')



    # Loss and Optimizer

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer().minimize(cost)



    # Accuracy

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    

    return features, labels, keep_prob, logits, cost, accuracy, optimizer

def show_stats(session, feature_batch, label_batch, cost, accuracy, features, labels, keep_prob):

    """

    show information about loss and validation accuracy

    : session: Current TensorFlow session

    : feature_batch: Batch of Numpy image data

    : label_batch: Batch of Numpy label data

    : cost: TensorFlow cost function

    : accuracy: TensorFlow accuracy function

    : features: placeholder for features

    : labels: placeholder for labels

    : keel_prob: placeholder for keep_probability

    """

    #features, labels, keep_prob, logits, cost, accuracy, optimizer = cnn_model()

    loss = session.run(cost, feed_dict={features:feature_batch, labels:label_batch, keep_prob:1.0})

    valid_acc = session.run(accuracy, feed_dict={

                features: valid_features,

                labels: valid_labels,

                keep_prob: 1.})

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(

                loss,

                valid_acc))
def train_model():

    validation_accuracy = 0.0

    log_batch_step = 10

    batches = []

    loss_batch = []

    train_acc_batch = []

    valid_acc_batch = []

    

    features, labels, keep_prob, logits, cost, accuracy, optimizer = cnn_model()



    # Feed dicts for training, validation

    train_feed_dict = {features: train_features, labels: train_labels, keep_prob:keep_probability}

    valid_feed_dict = {features: valid_features, labels: valid_labels, keep_prob:keep_probability}

 

    with tf.Session() as sess:

        # Initializing the variables

        sess.run(tf.global_variables_initializer())

        batch_count = int(math.ceil(len(train_features)/batch_size))

        # Training cycle

        for epoch_i in range(epochs):

            avg_cost = 0

            # Progress bar

            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # display((batch_count))

            # The training cycle

            for batch_i in batches_pbar:

                # Get a batch of training features and labels

                batch_start = batch_i*batch_size

                batch_features = train_features[batch_start:batch_start + batch_size]

                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss

                _, l = sess.run(

                        [optimizer, cost],

                        feed_dict={features: batch_features, labels: batch_labels, keep_prob: keep_probability})

                show_stats(sess, batch_features, batch_labels, 

                            cost, accuracy, features, labels, keep_prob)

                  

                avg_cost += l / batch_size

                        

                # Log every 50 batches

                if not batch_i % log_batch_step:

                    # Calculate Training and Validation accuracy

                    training_accuracy = sess.run(accuracy, feed_dict=train_feed_dict)

                    validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches

                    previous_batch = batches[-1] if batches else 0

                    batches.append(log_batch_step + previous_batch)

                    loss_batch.append(l)

                    train_acc_batch.append(training_accuracy)

                    valid_acc_batch.append(validation_accuracy)

            print ("Epoch:", (epoch_i+1), "cost =", "{:.5f}".format(avg_cost))

            # Check accuracy against Validation data

            validation_accuracy = sess.run(accuracy, feed_dict=valid_feed_dict)

            print('Validation accuracy at {}'.format(validation_accuracy))



        # Save Model

        saver = tf.train.Saver()

        save_path = saver.save(sess, save_model_path)



    # plot the cost and validation accuracy

    loss_plot = plt.subplot(211)

    loss_plot.set_title('Loss')

    loss_plot.plot(batches, loss_batch, 'g')

    loss_plot.set_xlim([batches[0], batches[-1]])

    acc_plot = plt.subplot(212)

    acc_plot.set_title('Accuracy')

    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')

    acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')

    acc_plot.set_ylim([0, 1.0])

    acc_plot.set_xlim([batches[0], batches[-1]])

    acc_plot.legend(loc=4)

    plt.tight_layout()

    plt.show()

def test_model():

    """

    Test the saved model against the test dataset

    """

    loaded_graph = tf.Graph()



    with tf.Session(graph=loaded_graph) as sess:

        # Load model

        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        loader.restore(sess, save_model_path)



        # Get Tensors from loaded model

        loaded_x = loaded_graph.get_tensor_by_name('features:0')

        loaded_y = loaded_graph.get_tensor_by_name('label:0')

        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_probability:0')

        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        

        test_accuracy = sess.run(loaded_acc, feed_dict={loaded_x: test_data, loaded_y: test_label, loaded_keep_prob: 1.0})

        

        print('Test Accuracy is {}'.format(test_accuracy))

def test_random_image(data):

    # test ramdom image

    loaded_graph = tf.Graph()

    top_n_predictions = 3



    with tf.Session(graph=loaded_graph) as session:

    # Load model

        loader = tf.train.import_meta_graph(save_model_path + '.meta')

        loader.restore(session, save_model_path)

        # Get Tensors from loaded model

        loaded_x = loaded_graph.get_tensor_by_name('features:0')

        loaded_y = loaded_graph.get_tensor_by_name('label:0')

        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_probability:0')

        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')

        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')



        num = randint(0, data.shape[0])

        img = data[num]

        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE,1)

        classification = session.run(tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions), feed_dict={loaded_x: [img], loaded_keep_prob: 1.0})

        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)

        plt.show()

        print ('Top {} prediction :' .format(top_n_predictions))

        print ('{}% ' .format(classification[0]*100) )

        print ('predicted value {}' .format(classification[1]))

 
train_features, valid_features, train_labels, valid_labels, test_data, test_label = get_data()

train_features = train_features.reshape(train_features.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

valid_features = valid_features.reshape(valid_features.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

test_data = test_data.reshape(test_data.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

train_model()

#### Call test function to test the accuracy of model against trained model

test_model()
### Call function to predict any random image from test data 

test_random_image(test_data)