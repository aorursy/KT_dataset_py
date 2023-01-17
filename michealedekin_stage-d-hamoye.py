# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print ("Training data: {},{}".format(train_images.shape, train_labels.shape))
print( "Test data: {}, {}" .format(test_images.shape, test_labels.shape))
class_labels = np.unique(train_labels)
print( "There are {} classes in the dataset. They are: {}" .format(len(class_labels),
class_labels))
plt.figure(figsize=( 8 , 5 ))
plt.subplot( 1 , 3 , 1 )
plt.imshow(train_images[ 0 ])
plt.title( "Label: {}" .format(train_labels[ 0 ]))
plt.subplot( 1 , 3 , 2 )
plt.imshow(train_images[ 2500 ])
plt.title( "Label: {}" .format(train_labels[ 2500 ]))
plt.subplot( 1 , 3 , 3 )
plt.imshow(test_images[ 12 ])
plt.show() 
train_images = train_images / 255.0
test_images = test_images / 255.0 
x_train = train_images[ 0 : 50000 ]
x_val = train_images[ 50000 :]
y_train = train_labels[ 0 : 50000 ]
y_val = train_labels[ 50000 :] 
print( "x_train: {}" .format(x_train.shape))
print( "x_val: {}" .format(x_val.shape))
print( "y_train: {}" .format(y_train.shape))
print( "y_val: {}" .format(y_val.shape))
new_dimension = np.prod(train_images.shape[ 1 :])
x_train = x_train.reshape(x_train.shape[ 0 ], new_dimension)
x_val = x_val.reshape(x_val.shape[ 0 ], new_dimension)
test_images = test_images.reshape(test_images.shape[ 0 ], new_dimension) 
print( "x_train: {}" .format(x_train.shape))
print( "x_val: {}" .format(x_val.shape))
print( "test_images: {}" .format(test_images.shape))
from tensorflow.keras.utils import to_categorical
no_labels = 10
y_train = to_categorical(y_train, no_labels)
y_val = to_categorical(y_val, no_labels)
y_test = to_categorical(test_labels, no_labels)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
X = tf.placeholder(tf.float32, [ None , new_dimension])
Y = tf.placeholder(tf.float32, [ None , no_labels]) 

# create model architecture
def multilayer_perceptron(x, no_classes, first_layer_neurons= 256 , second_layer_neurons= 128 ) : 
 
    # first layer
 first_weight = tf.Variable(tf.random_uniform([new_dimension, first_layer_neurons]))
 first_bias = tf.Variable(tf.zeros([first_layer_neurons]))
 first_layer_output = tf.nn.relu(tf.add(tf.matmul(x, first_weight), first_bias))

    # second layer
 second_weight = tf.Variable(tf.random_uniform([first_layer_neurons,second_layer_neurons]))
 second_bias = tf.Variable(tf.zeros([second_layer_neurons]))
 second_layer_output = tf.nn.relu(tf.add(tf.matmul(first_layer_output, second_weight),
 second_bias))
    
    # output layer
 final_weight = tf.Variable(tf.random_uniform([second_layer_neurons, no_classes]))
 final_bias = tf.Variable(tf.zeros([no_classes]))
 logits = tf.add(tf.matmul(second_layer_output, final_weight), final_bias)
 
 return logits 

logits = multilayer_perceptron(X, no_labels)
learning_rate = 0.01
#we define the loss and optimiser for the network
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimiser.minimize(loss_op) 
#initialise the variables
init = tf.global_variables_initializer()

epochs = 20
batch_size = 1000
iteration = len(x_train) // batch_size
#train model
with tf.Session() as session:
  session.run(init)
  for epoch in range(epochs):
    average_cost = 0
    start, end = 0, batch_size
    for i in range(iteration):
      batch_x, batch_y = x_train[start: end], y_train[start: end]
      _, loss = session.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
      start += batch_size
      end += batch_size 
      #average loss
      average_cost += loss/iteration
    print("Epoch========{}".format(epoch))
    #evaluate model
  prediction = tf.nn.softmax(logits)
  ground_truth = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(ground_truth, "float"))
  print("Accuracy: {}".format(accuracy.eval({X: test_images, Y: y_test})))
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#Build the model object
model = Sequential()
# Build the input and the hidden layers
model.add(Dense(256, activation='relu', input_shape=(new_dimension,)))
model.add(Dense(128, activation='relu'))
# Build the output layer
model.add(Dense(no_labels, activation='softmax'))

model.compile(optimizer= 'adam' , loss=tf.keras.losses.categorical_crossentropy,
 metrics=[ 'accuracy' ])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs= 20 ,
batch_size= 1000 )
test_loss, test_accuracy = model.evaluate(test_images, y_test)
print( 'Test loss: {}' .format(test_loss))
print( 'Test accuracy: {}' .format(test_accuracy)) 
plt.figure()
plt.plot(history.history[ 'loss' ], 'blue' )
plt.plot(history.history[ 'val_loss' ], 'red' )
plt.legend([ 'Training loss' , 'Validation Loss' ])
plt.xlabel( 'Epochs' )
plt.ylabel( 'Loss' )
plt.title( 'Loss Curves - before regularisation' )

from tensorflow.keras.layers import Dropout
reg_model = Sequential()
reg_model.add(Dense(256, activation='relu', input_shape=(new_dimension,)))
reg_model.add(Dropout(0.4))
reg_model.add(Dense(128, activation='relu'))
reg_model.add(Dropout(0.4))
reg_model.add(Dense(no_labels, activation='softmax'))
reg_model.compile(optimizer= 'adam' , loss=tf.keras.losses.categorical_crossentropy,
 metrics=[ 'accuracy' ])
reg_history = reg_model.fit(x_train, y_train, validation_data=(x_val, y_val),
 epochs= 20 , batch_size= 1000 )
test_loss, test_accuracy = reg_model.evaluate(test_images, y_test)
print( 'Test loss: {}' .format(test_loss))
print( 'Test accuracy: {}' .format(test_accuracy)) 
plt.figure()
plt.plot(reg_history.history[ 'loss' ], 'blue' )
plt.plot(reg_history.history[ 'val_loss' ], 'red' )
plt.legend([ 'Training loss' , 'Validation Loss' ])
plt.xlabel( 'Epochs' )
plt.ylabel( 'Loss' )
plt.title( 'Loss Curves - after regularisation' )
