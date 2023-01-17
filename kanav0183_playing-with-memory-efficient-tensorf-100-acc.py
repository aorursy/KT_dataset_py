# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
img = np.load('../input/Sign-language-digits-dataset/X.npy')
label = np.load('../input/Sign-language-digits-dataset/Y.npy')
plt.imshow(img[2000])
img[0].shape

x = tf.placeholder(tf.float32, shape=[None, 64,64], name='X')

x_image = tf.reshape(x, [-1, 64, 64, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
def new_conv_layer(input, num_input_channels, filter_size, num_filters, name,stride=[1, 1, 1, 1]):
    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=stride, padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights
def new_pool_layer(input,name,stride=[1,2,2,1],ksize=[1,2,2,1]):
  with tf.variable_scope(name) as scope:
    layer = tf.nn.max_pool(input,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')
    return layer
  
def new_relu_layer(input,name):
  with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
        
        return layer

def new_fc_layer(input, num_inputs, num_outputs, name):

  with tf.variable_scope(name) as scope:

      # Create new weights and biases.
      weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
      biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

      # Multiply the input and weights, and then add the bias-values.
      layer = tf.matmul(input, weights) + biases

      return layer


  
  
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=3, num_filters=6, name ="conv1")

layer_pool1 = new_pool_layer(layer_conv1,'pool1')

layer_relu1  =new_relu_layer(layer_pool1,'relu1')

layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=3, num_filters=32, name ="conv2")

layer_pool2 = new_pool_layer(layer_conv2,'pool2')

layer_relu2  =new_relu_layer(layer_pool2,'relu2')

layer_conv3, weights_conv3 = new_conv_layer(input=layer_relu2, num_input_channels=32, filter_size=3, num_filters=64, name ="conv3")

layer_pool3 = new_pool_layer(layer_conv3,'pool3')


layer_relu3  =new_relu_layer(layer_pool3,'relu3')

num_features = layer_relu3.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu3, [-1, num_features])

layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128,name='fc1')

### Batch Normalisation

layer_bn = tf.contrib.layers.batch_norm(layer_fc1 ,center=True, scale=True)

layer_relu4 = new_relu_layer(layer_bn, name="relu3")

layer_fc2 = new_fc_layer(input=layer_relu4, num_inputs=128, num_outputs=10, name="fc2")
with tf.variable_scope('softmax'):
  y_pred = tf.nn.softmax(layer_fc2)
  y_pred_cls = tf.argmax(y_pred, axis=1)
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
with tf.name_scope('opt'):
  optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")
# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()
num_epochs = 20
batchSize = 100
import time
def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
         '\tTraining Loss: {:.3f}'.format(trainLoss),
         '\tValidation Loss: {:.3f}'.format(validationLoss),
         '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))
from sklearn.model_selection import train_test_split
with tf.Session () as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(num_epochs):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(img, label,\
                                                      test_size = 0.2)   
        # training loss
        for i in range(0, len(train_x), 100):
            trainLoss, _= sess.run([cost, optimizer], feed_dict = {
                x: train_x[i: i+batchSize],
                y_true: train_y[i: i+batchSize]
            })
            
        # validation loss
        valAcc, valLoss = sess.run([accuracy, cost], feed_dict ={
            x: val_x,
            y_true: val_y,})
        
        
        # print out
        printResult(epoch, num_epochs, trainLoss, valLoss, valAcc)

x = tf.placeholder(tf.float32, shape=[None, 64,64], name='X')

x_image = tf.reshape(x, [-1, 64, 64, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
layer, weights = new_conv_layer(input=x_image, num_input_channels=1, filter_size=3, num_filters=96, name ="conv1",stride=[1,4,4,1])
layer  =new_relu_layer(layer,'relu1')

layer, weights = new_conv_layer(input=layer, num_input_channels=96, filter_size=3, num_filters=96, name ="conv1",stride=[1,2,2,1])
layer  =new_relu_layer(layer,'relu1')

layer = new_pool_layer(layer,'pool1',)

#batch norm
layer, weights = new_conv_layer(input=layer, num_input_channels=96, filter_size=5, num_filters=256, name ="conv1")
layer =new_relu_layer(layer,'relu1')
layer = new_pool_layer(layer,'pool1',ksize=[1,3,3,1])
#BAtch norm
layer = tf.contrib.layers.batch_norm(layer ,center=True, scale=True)



layer, weights = new_conv_layer(input=layer, num_input_channels=256, filter_size=5, num_filters=384, name ="conv1",)
layer =new_relu_layer(layer,'relu1')

layer ,weights= new_conv_layer(input=layer, num_input_channels=384, filter_size=5, num_filters=384, name ="conv1",)
layer  =new_relu_layer(layer,'relu1')

layer, weights = new_conv_layer(input=layer, num_input_channels=384, filter_size=5, num_filters=256, name ="conv1")
layer =new_relu_layer(layer,'relu1')
layer = new_pool_layer(layer,'pool1',ksize=[1,3,3,1])
#batch norm
layer = tf.contrib.layers.batch_norm(layer ,center=True, scale=True)


num_features = layer.get_shape()[1:4].num_elements()
layer= tf.reshape(layer, [-1, num_features])

layer= new_fc_layer(layer, num_inputs=num_features, num_outputs=4096,name='fc1')

layer = tf.layers.dropout(layer,rate=0.5)

layer= new_relu_layer(layer, name="relu3")

layer = new_fc_layer(input=layer, num_inputs=4096, num_outputs=4096, name="fc2")
layer = tf.layers.dropout(layer,rate=0.5)

layer = new_fc_layer(input=layer, num_inputs=4096, num_outputs=10, name="fc2")
#layer = tf.layers.dropout(layer,rate=0.5)

#layer = new_fc_layer(input=layer, num_inputs=128, num_outputs=10, name="fc2")

with tf.variable_scope('softmax'):
  y_pred = tf.nn.softmax(layer)
  y_pred_cls = tf.argmax(y_pred, axis=1)
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('opt'):
  optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")
# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()
num_epochs = 20
batchSize = 100
import time
def printResult(epoch, numberOfEpoch, trainLoss, validationLoss, validationAccuracy):
    print("Epoch: {}/{}".format(epoch+1, numberOfEpoch),
         '\tTraining Loss: {:.3f}'.format(trainLoss),
         '\tValidation Loss: {:.3f}'.format(validationLoss),
         '\tAccuracy: {:.2f}%'.format(validationAccuracy*100))
from sklearn.model_selection import train_test_split
with tf.Session () as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(num_epochs):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(img, label,\
                                                      test_size = 0.2)   
        # training loss
        for i in range(0, len(train_x), 100):
            trainLoss, _= sess.run([cost, optimizer], feed_dict = {
                x: train_x[i: i+batchSize],
                y_true: train_y[i: i+batchSize]
            })
            
        # validation loss
        valAcc, valLoss = sess.run([accuracy, cost], feed_dict ={
            x: val_x,
            y_true: val_y,})
        
        
        # print out
        printResult(epoch, num_epochs, trainLoss, valLoss, valAcc)
def fire(inputs,squeezeTo,expandTo):
    h = squeeze(inputs,squeezeTo)
    h = expand(h,expandTo)
    h = tf.clip_by_norm(h,NORM) # Very important
    activations.append(h)

def squeeze(inputs,squeezeTo):
    with tf.name_scope('squeeze'):
        inputSize = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1,1,inputSize,squeezeTo]))
        h = tf.nn.relu(tf.nn.conv2d(inputs,w,[1,1,1,1],'SAME'))        
    return h

def expand(inputs,expandTo):
    with tf.name_scope('expand'):
        squeezeTo = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1,1,squeezeTo,expandTo]))
        h1x1 = tf.nn.relu(tf.nn.conv2d(inputs,w,[1,1,1,1],'SAME'))
        w = tf.Variable(tf.truncated_normal([3,3,squeezeTo,expandTo]))
        h3x3 = tf.nn.relu(tf.nn.conv2d(inputs,w,[1,1,1,1],'SAME'))
        h = tf.concat([h1x1,h3x3],3)
    return h
x = tf.placeholder(tf.float32, shape=[None, 64,64], name='X')

x_image = tf.reshape(x, [-1, 64, 64, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer, weights = new_conv_layer(input=x_image, num_input_channels=1, filter_size=3, num_filters=96, name ="conv1",stride=[1,4,4,1])
layer  =new_relu_layer(layer,'relu1')

layer, weights = new_conv_layer(input=layer, num_input_channels=96, filter_size=3, num_filters=96, name ="conv1",stride=[1,2,2,1])
layer  =new_relu_layer(layer,'relu1')

layer = new_pool_layer(layer,'pool1',)


# Squeeze layer
h = squeeze(layer,16)
layer = expand(h,64)
#*****************************************************************************************************************************************
# fire_squeeze, weights = new_conv_layer(input=layer, num_input_channels=96, filter_size=1, num_filters=16, name ="conv1",stride=[1,1,1,1])
# fire_squeeze  = new_relu_layer(fire_squeeze,'relu1')

# fire_expand1, weights = new_conv_layer(input=fire_squeeze, num_input_channels=16, filter_size=3, num_filters=64, name ="conv1",stride=[1,1,1,1])
# fire_expand1  = new_relu_layer(fire_expand1,'relu3')

# fire_expand2, weights = new_conv_layer(input=fire_squeeze, num_input_channels=16, filter_size=1, num_filters=64, name ="conv1",stride=[1,1,1,1])
# fire_expand2  = new_relu_layer(fire_expand2,'relu4')

# layer, weights = tf.concat([fire_expand1,fire_expand2],axis=3)
#*****************************************************************************************************************************************



#batch norm
layer, weights = new_conv_layer(input=layer, num_input_channels=128, filter_size=5, num_filters=256, name ="conv1")
layer =new_relu_layer(layer,'relu1')
layer = new_pool_layer(layer,'pool1',ksize=[1,3,3,1])
#BAtch norm
layer = tf.contrib.layers.batch_norm(layer ,center=True, scale=True)



layer, weights = new_conv_layer(input=layer, num_input_channels=256, filter_size=5, num_filters=384, name ="conv1",)
layer =new_relu_layer(layer,'relu1')

layer ,weights= new_conv_layer(input=layer, num_input_channels=384, filter_size=5, num_filters=384, name ="conv1",)
layer  =new_relu_layer(layer,'relu1')

layer, weights = new_conv_layer(input=layer, num_input_channels=384, filter_size=5, num_filters=256, name ="conv1")
layer =new_relu_layer(layer,'relu1')
layer = new_pool_layer(layer,'pool1',ksize=[1,3,3,1])
#batch norm
layer = tf.contrib.layers.batch_norm(layer ,center=True, scale=True)


num_features = layer.get_shape()[1:4].num_elements()
layer= tf.reshape(layer, [-1, num_features])

layer= new_fc_layer(layer, num_inputs=num_features, num_outputs=4096,name='fc1')

layer = tf.layers.dropout(layer,rate=0.5)

layer= new_relu_layer(layer, name="relu3")

layer = new_fc_layer(input=layer, num_inputs=4096, num_outputs=4096, name="fc2")
layer = tf.layers.dropout(layer,rate=0.5)

layer = new_fc_layer(input=layer, num_inputs=4096, num_outputs=10, name="fc2")
#layer = tf.layers.dropout(layer,rate=0.5)

#layer = new_fc_layer(input=layer, num_inputs=128, num_outputs=10, name="fc2")

with tf.variable_scope('softmax'):
  y_pred = tf.nn.softmax(layer)
  y_pred_cls = tf.argmax(y_pred, axis=1)
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('opt'):
  optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")
# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()
num_epochs = 20
batchSize = 100
import time
from sklearn.model_selection import train_test_split
with tf.Session () as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(num_epochs):
        # training data & validation data
        train_x, val_x, train_y, val_y = train_test_split(img, label,\
                                                      test_size = 0.2)   
        # training loss
        for i in range(0, len(train_x), 100):
            trainLoss, _= sess.run([cost, optimizer], feed_dict = {
                x: train_x[i: i+batchSize],
                y_true: train_y[i: i+batchSize]
            })
            
        # validation loss
        valAcc, valLoss = sess.run([accuracy, cost], feed_dict ={
            x: val_x,
            y_true: val_y,})
        
        
        # print out
        printResult(epoch, num_epochs, trainLoss, valLoss, valAcc)
