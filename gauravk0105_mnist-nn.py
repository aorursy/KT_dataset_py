import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

from matplotlib import pyplot
# importing mnist data

# Training data
df_train = pd.read_csv('../input/mnist_train.csv')

# Test data
df_test = pd.read_csv('../input/mnist_test.csv')

# containing 60000 images each of shape (28*28*1) 
# row 1 -- [category, (28*28) i.e. image pixels in a entire row]
print(df_train.shape)
#Validation split from training data
df_features = df_train.iloc[:, 1:785]     # pixels of image (28*28*1)
df_label = df_train.iloc[:, 0]            # label associated with the image

# splitting training dataset into train and cross validation data
# so that we can use training data to build several models with different different parameters (learning rate[alpha], adam optimization parameter[beta])
# i.e. to tune model on different hyperparamter
# and validation data to select the best model out of them which give better result on validation dataset.
X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)
# Normalization : Very Important (basically to speed up gradient descent an algorithm used to train model)
#  generally speeds up learning and leads to faster convergence.
# So in image best way to do normalization is to divide each pixel by 255 so that their standard deviation become zero  
X_train = X_train.values.astype('float32')/255.
X_cv = X_cv.values.astype('float32')/255.

# applying normalization similarily on test data 
X_test = df_test.iloc[:, 1:785]
y_test = df_test.iloc[:, 0]
X_test = X_test.values.astype('float32')/255.


# Convert labels to One Hot Encoded 
# in case of label 5 -- [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# it will be helpful in case of vectorization approach which keras inbuilt uses and so helpful in doing vectorized calculation in one go
y_train = keras.utils.to_categorical(y_train, 10)
y_cv = keras.utils.to_categorical(y_cv, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# print the shape

# shape of training data
print ("Shape of X_train: ", X_train.shape)
print ("Shape of Y_train: ", y_train.shape)

# shape of validation data

print ("Shape of X_cv: ", X_cv.shape)
print ("Shape of Y_cv: ", y_cv.shape)

# Shape of test data

print ("Shape of X_test: ", X_test.shape)
print ("Shape of Y_test: ", y_test.shape)
# display image

image_X_train=X_train.reshape(48000, 28*28*1)
plt.imshow(np.resize(image_X_train[4], (28, 28)))
plt.imshow(np.resize(image_X_train[331], (28, 28)))
plt.imshow(np.resize(image_X_train[2991], (28, 28)))
plt.imshow(np.resize(image_X_train[35547], (28, 28)))
plt.imshow(np.resize(image_X_train[6534], (28, 28)))
# model : ADAM optimizater + 4 hidden + softmax output + learning rate (alpha=0.1) + dropout 

# Input Parameters
n_input = 784 # number of features which is input layer
n_hidden_1 = 300 # number of neuron in hidden layer 1 
n_hidden_2 = 100 # number of neuron in hidden layer 2 
n_hidden_3 = 100 # number of neuron in hidden layer 3
n_hidden_4 = 200 # number of neuron in hidden layer 4
num_digits = 10  # number of node in output layer which is softmax layer

# Insert Hyperparameters
learning_rate = 0.01  # hyperparameter in Gradient desent algorithm (represents the steps will take in order to converge)
training_epochs = 30  # one iteration through training set
batch_size = 100  # dividing training data in batches for faster convergence
dropout_factor = 0.3 # produce an effect of regularization (so it bascially shutdown few neurons in each of the hidden layer randomly for each of
                     # the training example. So let consider for a patricular training sample t1, for that specific we are working with few number of neurons
                     # meaning that we are having less number of parameter to train which led to finding out less complex hypothesis which will probably be
                     # less prone to overfitting
'''
 for a 1 training example
 we are having 784 features/nodes in input layer  
                                                 [feature weights], [bias] 
 hidden layer 1 -> 300 nodes, so # of weights are = (300*784 + 300*1)
 hidden layer 2 -> 100 nodes, so # of weights are = (100*300 + 100*1)
 hidden layer 3 -> 100 nodes, so # of weights are = (100*100 + 100*1)
 hidden layer 4 -> 200 nodes, so # of weights are = (200*100 + 200*1)
 
 output layer -> 10 nodes, so # of weights are = (10*200 + 10*1)
'''            

'''
Why relu?: because in sigmoid or in tanH activation function for higher value the derivative (slope) become zero and so learning becomes slow.
'''

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu',name = "Hidden_Layer_1")(Inp)
x = Dropout(dropout_factor)(x)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dropout(dropout_factor)(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dropout(dropout_factor)(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

model = Model(Inp, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = batch_size, verbose=1, epochs = training_epochs, validation_data=(X_cv, y_cv))
n_input = 784 # number of features which is input layer
n_hidden_1 = 300 # number of neuron in hidden layer 1 
n_hidden_2 = 100 # number of neuron in hidden layer 2 
n_hidden_3 = 100 # number of neuron in hidden layer 3
n_hidden_4 = 200 # number of neuron in hidden layer 4
num_digits = 10  # number of node in output layer which is softmax layer

# Insert Hyperparameters
learning_rate = 0.1 # hyperparameter in gradient desent algorithm i.e.  b0 = b0 - alpha*(derivative of error w.r.to b0) 
                # denotes how big the steps we are taking to reach towards the point of minima
training_epochs = 30 # i.e. one iteration through training set 
batch_size = 100  # dividing training data in batches for faster convergence
dropout_factor = 0.3 # produce an effect of regularization

# input layer (784 nodes)
Inp = Input(shape=(784,))

# hidden layer 1 (300 nodes)  
x = Dense(n_hidden_1, activation='relu',name = "Hidden_Layer_1")(Inp)
x = Dropout(dropout_factor)(x)

# hidden layer 1 (100 nodes)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dropout(dropout_factor)(x)

# hidden layer 1 (100 nodes)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dropout(dropout_factor)(x)

# hidden layer 1 (200 nodes)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)

# output layer (10 nodes)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

model = Model(Inp, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = batch_size, verbose=1, epochs = training_epochs, validation_data=(X_cv, y_cv))
'''
    with alpha = 0.01                                           with alpha = 0.1

loss: 0.0328 - acc: 0.9901                                  loss: 0.0326 - acc: 0.9904
val_loss: 0.0766 - val_acc: 0.9828                          val_loss: 0.0672 - val_acc: 0.9846
'''
# plot loss during training
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train') 
pyplot.plot(history.history['val_acc'], label='val') 
pyplot.legend()
pyplot.show()
score, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(score, test_acc)
import random
image_idx = random.randint(1,10000)-1
first_image = X_test[image_idx]
pixels = first_image.reshape((28, 28))
pyplot.imshow(pixels, cmap='gray')
pyplot.show()

# Model predication output
predictions = model.predict(X_test, batch_size=200)
print("Predication Output on test image: ", predictions[image_idx].argmax(axis=0))
'''
Model Summary

Model Used: model: ADAM optimizater + 4 hidden + softmax output + learning rate (alpha=0.1) + dropout

with Hyperparameter
     alpha = 0.1, 
     batch size = 100
     epochs: 30
     dropout_factor = 0.3
       

     On Training Data: loss: 0.0326 - acc: 0.9904 
     On validation Data: val_loss: 0.0672 - val_acc: 0.9846
     On Test Data: loss: 0.0835998178840884 - acc: 0.9812
'''