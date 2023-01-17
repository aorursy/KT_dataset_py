# Andrea Minardi 21.03.2016
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
print('Libraries imported')
dataset = np.genfromtxt('../input/train.csv', delimiter=',', skip_header=1, dtype='float32')  
print ('Dataset shape:', dataset.shape)
# Extract image as example
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# display image
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(28, 28)    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    
# output image     
display(dataset[7000,1:785])
# Working on dataset

labels = dataset[:,0]  # labels
data = dataset[:,1:]  # data

mean = np.mean(data)
dev_st = np.std(data)

print ('Data Shape: ',data.shape)
print ('Data Mean: ',np.mean(data))
print ('Data Standard deviation:', np.std(data))

data = data - mean
data = data / dev_st

print ('Normalized data:')
print ('Data shape: ',data.shape)
print ('Data Media: ',np.mean(data))
print ('Data Standard deviation:', np.std(data))
# Setting data and validation subset

# Start point and end point
train_subset_start = 13000
train_subset_end = 15000
validation_subset = 1000

# Train and validation Creation
train = data[train_subset_start:train_subset_end, :]
validation = data[train_subset_end + 1 : train_subset_end + validation_subset + 1, :]

# Labels creation
train_labels = labels[train_subset_start:train_subset_end]
validation_labels = labels[train_subset_end + 1 : train_subset_end + validation_subset + 1]

print ('Training: ', train.shape, train_labels.shape)
print ('Validation: ', validation.shape, validation_labels.shape)

# Labels in matrix format
matrix_train_labels = (np.arange(10) == train_labels[:,None]).astype(np.float32)
matrix_validation_labels = (np.arange(10) == validation_labels[:,None]).astype(np.float32)
print ('Matrix_train_labels: ',matrix_train_labels.shape)
print ('Matrix_validation_labels: ',matrix_validation_labels.shape)
# Utility functions

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

print ('Functions defined')
graph = tf.Graph()   
with graph.as_default():
    
    # Input and output
    y_ = tf.constant(matrix_train_labels)         # Labels matrix    
    x = tf.constant(train)                        # Data input
    x_validation = tf.constant(validation)        # Validation input
    
    # Matrix Initialising
    W1 = tf.Variable(tf.truncated_normal([10, 10, 1, 16], stddev=0.1))
    b1 = bias_variable([16])
    
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1))
    b2 = bias_variable([32])
    
    W = weight_variable([7*7*32, 10])
    b = bias_variable([10])

    # NN Model
    def model(data):
        image = tf.reshape(data, [-1,28,28,1])
        h1 = tf.nn.relu(conv2d(image, W1) + b1)
        h1 = max_pool_2x2(h1)
        h2 = tf.nn.relu(conv2d(h1, W2)+b2)
        h2 = max_pool_2x2(h2)
        h2 = tf.reshape(h2, [-1, 7*7*32])
        return tf.matmul(h2, W) + b

    # Training computation.
    logits = model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))   # cross entropy
    
    # Optimizer.
    optimizer =tf.train.AdagradOptimizer(0.015).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    validation_logits = model(validation)
    validation_prediction = tf.nn.softmax(validation_logits)
    
print ('NN Layers Initialised')
# steps
start_steps = 0
num_steps = 20

# accuracy calculus
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels,1)) / predictions.shape[0])

# NN operating
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialised')

  for step in range(start_steps+1, start_steps + num_steps):
    _, l, predictions, validations = session.run([optimizer, loss, train_prediction, validation_prediction])
    if (step % 2 == 0):
        print('Loss at step %d: %f' % (step, l))
        print('Training accuracy: %.1f%%' % accuracy(predictions, matrix_train_labels))
        print('Validation accuracy: %.1f%%' % accuracy(validations, matrix_validation_labels))
  print('End')

session.close() 