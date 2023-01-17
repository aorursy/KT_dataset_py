#Allows using operating system dependent functionality
import os
#Provides support for multidimensional arrays and high-level mathematical functions
import numpy as np
 #Provides support for data manipulation and analysis
import pandas as pd
#Provides MATLAB-like plotting framework (based on numpy)
import matplotlib.pyplot as plt
#Provides data visualisation interface for drawing and graphing (based on matplotlib)
import seaborn as sns
#Provides a framework for fast numerical computing, machine learning and neural networks
import tensorflow as tf
#Provides a random permutation cross-validator. Yields indices to split data into training and test sets
from sklearn.model_selection import ShuffleSplit
#Scales/standartizes features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
#Encodes labels with value between 0 and n_classes-1
from sklearn.preprocessing import LabelEncoder
#Encodes categorical integer features using a one-hot aka one-of-K scheme
from sklearn.preprocessing import OneHotEncoder
#Chekcing input directory
print(os.listdir("../input"))
# Learning rate
LEARNING_RATE = 0.001
# Number of training iterations to run
STEPS = 25000 
#Validation data size
VALIDATION_SIZE = 1000 
#Types of labels for digits 1 through 10
LABELS = 10
#Width/height of the image
DIMENSION = 28
#Number of colors in the image (greyscale)
COLOUR_CHANNELS = 1
#Batch size
BATCH_SIZE = 100
#Convolutional Kernel size
CONVOLUTION_PATCH = 5
#Number of Convolutional Kernels
DEPTH = 32
#Number of hidden neurons in the fully connected layer
HIDDEN = 1024
#Read cvs files into pandas dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#Look at top 5 rows of train dataframe
train.head()
#Remove the labels as a numpy array from the dataframe
labels = np.array(train.pop('label'))
#Encode labels
labels = LabelEncoder().fit_transform(labels)[:, None]
#Look at encoded labels
print(labels.shape)
print('\n')
print(labels[:5])
#Apply OneHot encoding
labels = OneHotEncoder().fit_transform(labels).todense()
#Look at one hot encoded labels
print(labels.shape)
print('\n')
print(labels[:5])
#Convert the dataframe to a numpy array
train = StandardScaler().fit_transform(np.float32(train.values)) 
#Look at scaled train dataframe
print(train.shape)
print('\n')
print(train[:5])
#Reshape the data into 42000 28x28 arrays (Number of images, height, width, colour channels)
train = train.reshape(-1, DIMENSION, DIMENSION, COLOUR_CHANNELS) 
#Look at reshaped train dataframe
print(train.shape)
print('\n')
print(train[:1])
#Split data into training and validation sets
ts_data, valids_data = train[:-VALIDATION_SIZE], train[-VALIDATION_SIZE:]
ts_labels, valids_labels = labels[:-VALIDATION_SIZE], labels[-VALIDATION_SIZE:]
#Initialize the input data with placeholders.
tf_data = tf.placeholder(tf.float32, shape=(None, DIMENSION, DIMENSION, COLOUR_CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABELS))
#Initialise the biases
#Convolutional layer 1 biases with depth of DEPTH
b1 = tf.Variable(tf.zeros([DEPTH]))
#Convolutional layer 2 biases (twice the depth of the first convolutional layer)
b2 = tf.Variable(tf.constant(1.0, shape=[2 * DEPTH]))
#Hidden/Dense layer biases with HIDDEN of hidden nodes
b3 = tf.Variable(tf.constant(1.0, shape=[HIDDEN]))
#Output layer biases
b4 = tf.Variable(tf.constant(1.0, shape=[LABELS]))

#Initialise the weights with patch size of CONVOLUTION_PATCH
#Convolutional layer 1 weights
w1 = tf.Variable(tf.truncated_normal([CONVOLUTION_PATCH, CONVOLUTION_PATCH, COLOUR_CHANNELS, DEPTH], stddev=0.1))
#Convolutional layer 2 weights (twice the depth of the first convolutional layer)
w2 = tf.Variable(tf.truncated_normal([CONVOLUTION_PATCH, CONVOLUTION_PATCH, DEPTH, 2 * DEPTH], stddev=0.1))
#Hidden/Dense layer weights
w3 = tf.Variable(tf.truncated_normal([DIMENSION // 4 * DIMENSION // 4 * 2 * DEPTH, HIDDEN], stddev=0.1))
#Output layer biaweightsses
w4 = tf.Variable(tf.truncated_normal([HIDDEN, LABELS], stddev=0.1))
#Assemble the layers
def logits(input):
    #Convolutional layer 1
    mnist_classifier = tf.nn.conv2d(input, w1, [1, 1, 1, 1], padding='SAME')
    mnist_classifier = tf.nn.max_pool(mnist_classifier, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    mnist_classifier = tf.nn.relu(mnist_classifier + b1)
    #Convolutional layer 2
    mnist_classifier = tf.nn.conv2d(mnist_classifier, w2, [1, 1, 1, 1], padding='SAME')
    mnist_classifier = tf.nn.max_pool(mnist_classifier, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    mnist_classifier = tf.nn.relu(mnist_classifier + b2)
    #Hidden layer
    mnist_classifier = tf.reshape(mnist_classifier, (-1, DIMENSION // 4 * DIMENSION // 4 * 2 * DEPTH))
    mnist_classifier = tf.nn.relu(tf.matmul(mnist_classifier, w3) + b3)
    return tf.matmul(mnist_classifier, w4) + b4

#Prediction:
tf_pred = tf.nn.softmax(logits(tf_data))
#Get the loss by using Categorical Cross Entropy Loss function for training the model.
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits(tf_data), labels=tf_labels))
#Get accuracy 
tf_acc = 100 * tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_labels, 1))))

#Set up one of the following (or other) optimisers

#tf_opt = tf.train.RMSPropOptimizer(LEARNING_RATE)
tf_opt = tf.train.AdamOptimizer(LEARNING_RATE)

grads_and_vars = tf_opt.compute_gradients(tf_loss)
tf_step = tf_opt.minimize(tf_loss)
#Open tensorflow session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
#Initialise error log
error_log = [(0, 0, 10)] 
#Run the session
ss = ShuffleSplit(n_splits=STEPS, train_size=BATCH_SIZE)
#Get the number of splitting iterations in the cross-validator
ss.get_n_splits(ts_data, ts_labels)
#Go through the data for STEPS of steps
for step, (id, _) in enumerate(ss.split(ts_data,ts_labels), start=1):
    #Get single image and its label at id of id and set it to fd
    fd = {tf_data:ts_data[id], tf_labels:ts_labels[id]}
    #Run the session for single image with the chosen optimiser
    session.run(tf_step, feed_dict=fd)
    #Every 500 steps do this
    if step%1000 == 0:
        #Get validation images and labels and assign them to fd
        fd = {tf_data:valids_data, tf_labels:valids_labels}
        #Get model loss and accuracy from the validation dataset
        validation_loss, validation_accuracy = session.run([tf_loss, tf_acc], feed_dict=fd)
        #Save model loss and accuracy at each 500 step to history
        error_log.append((step, validation_loss, validation_accuracy))
        #Print step and accuracy
        print('Step %i \t Valid. Acc. = %f'%(step, validation_accuracy), end='\n')
#Unzip the history list into 3 tuples
steps, loss, accuracy = zip(*error_log)
#Create a figure and set plot size
fig = plt.figure(figsize=(16,6))
#Add first subplot
sub1 = fig.add_subplot(221)
#Plot the data on first subplot
sub1.plot(steps,loss, 'o-')
#Set title for first subplot
sub1.set_title('Validation Loss')
#Set x label for first subplot
sub1.set_xlabel('Steps')
#Set y label for first subplot
sub1.set_ylabel('Log(Loss)')
#Add second subplot
sub2 = fig.add_subplot(222)
#Plot the data on second subplot
sub2.plot(steps, accuracy, '.-')
#Set x label for second subplot
sub2.set_xlabel('Steps')
#Set y label for second subplot
sub2.set_ylabel('Accuracy')
#Set title for second subplot
sub2.set_title('Accuracy')
#Make sure plots don't overlap
plt.tight_layout()
#Show plots
plt.show()
# Convert the test dataframe to a numpy array
test_data = StandardScaler().fit_transform(np.float32(test.values)) 
#Reshape test_data into 42000 28x28 matricies
test_data = test_data.reshape(-1, DIMENSION, DIMENSION, COLOUR_CHANNELS) 
#Make a prediction about the test labels
test_pred = session.run(tf_pred, feed_dict={tf_data:test_data})
test_labels = np.argmax(test_pred, axis=1)
#Show the structure of the predictions
print(test_pred.shape)
print(test_labels.shape)
print(test_pred[:2])
print(test_labels[:2])
#Plotting an example
image = 34
plt.axis('off')
plt.imshow(test_data[image,:,:,0])
plt.show()
print("Prediction: %i"%test_labels[image])
#Save predictions to a dataframe
predictions = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
#Write the datafram to a csv file
predictions.to_csv('predictions2.csv', index=False)
#Show sample of the submission dataframe
predictions.head(5)
