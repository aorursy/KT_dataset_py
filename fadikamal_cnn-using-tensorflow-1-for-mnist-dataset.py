#pip install tensorflow-gpu==1.15  # GPU
import tensorflow as tf 



tf.__version__
#import the important modules, libraries, and frameworks

import numpy as np 



import matplotlib.pyplot as  plt



import os 



import cv2



import pandas as pd 
#import the training and test sets

train_df = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

test_df = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
#print the shpae of training and test data sets

print(train_df.shape)

print(test_df.shape)
#Explore the dataset

train_df.head()
#split the training dataset into  features --> tr_x and labels --> tr_y

tr_x = train_df[train_df.columns[train_df.columns != 'label']]

tr_y = train_df[train_df.columns[train_df.columns == 'label']]

print("The shape of training features:{0}\nThe shape of labels:{1}".format(tr_x.shape,tr_y.shape))
#Convert the training features and labels into numpy array to feed the CNN During training

tr_x = tr_x.values



tr_y = tr_y.values
#split the testing dataset into  features --> test_x and labels --> test_y

test_x = test_df[test_df.columns[test_df.columns != 'label']]

test_y = test_df[test_df.columns[test_df.columns == 'label']]

print("The shape of testing features:{0}\nThe shape of labels:{1}".format(test_x.shape,test_y.shape))
#Convert the training features and labels into numpy array to feed the CNN during testing

test_x = test_x.values



test_y = test_y.values
#Define the input and output placeholders

x = tf.placeholder(tf.float32, [None, 28*28])

y = tf.placeholder(tf.float32, [None, 10])



#Apply 32 convolutions of window-size 5*5

w1 = tf.Variable(tf.random_normal([5,5,1,32]))

b1 = tf.Variable(tf.random_normal([32]))



#Then Apply 32 more  convolutions of window-size 5*5

w2 = tf.Variable(tf.random_normal([5,5,32,64]))

b2 = tf.Variable(tf.random_normal([64]))



#Then we introduced a fully-connected l ayer

w3 = tf.Variable(tf.random_normal([7*7*64,1024]))

b3 = tf.Variable(tf.random_normal([1024]))



#Finaly, we define the variables for a fully-connected linear layer

w_out = tf.Variable(tf.random_normal([1024,10]))

b_out = tf.Variable(tf.random_normal([10]))
#lets Create dic to hold our  parameters to can get it after updated

parameters = {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3, "WO": w_out, "bO": b_out}
#Create a convolutional layer

def conv_layer(x, w, b):

    conv = tf.nn.conv2d(x, w, strides = [1,1,1,1], padding = 'SAME')

    conv_with_b = tf.nn.bias_add(conv, b)

    conv_out = tf.nn.relu(conv_with_b)

    return conv_out
#Create a max-pool layer

def maxpool_layer(conv, k = 2):

    return tf.nn.max_pool(conv, ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'SAME')
def model(x):

    #Reshape the features in the form(m,height,wideth,channels), and m represents the number of training examples

    x_reshaped = tf.reshape(x, shape = [-1, 28, 28, 1])

    

    #Construct the first layer of convolution and max-pooling

    conv_out1 = conv_layer(x_reshaped, w1, b1)

    maxpool_out1 = maxpool_layer(conv_out1)

    

    #Construct the second layer of convolution and max-pooling

    conv_out2 = conv_layer(maxpool_out1, w2, b2)

    maxpool_out2 = maxpool_layer(conv_out2)

    

    #Finally, Construct the final fully connected layer

    ##1.First flatten the output from the second layer

    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, w3.get_shape().as_list()[0]])

    

    ##2.Create the linear part of the fully connected layer

    linear_part = tf.add(tf.matmul(maxpool_reshaped, w3), b3)

    

    ##3.Create the non-linear part of the fully connected layer 

    ##in other word,applay the activation function on the linear part

    nonlinear_part = tf.nn.relu(linear_part)

    

    #Get the the output ten classses

    output = tf.add(tf.matmul(nonlinear_part, w_out), b_out)

    

    return output
#Construct the model

model_op = model(x)



#Define the classification loss function

cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model_op, labels = y))



#Define the training optimizer to minimize the loss function

train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)



#Define the Calculation to get the accuracy of the model

##1.first define the correct predictions between the model and the ground truth

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))



##2.Then Define the accuracy fomula which is the number of examples correctly classified over the total number of examples

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#Encoding the output using one_hot() function

def one_hot(labels):

    labels_ = np.zeros((60000, 10))

    labels_[np.arange(60000), labels] = 1

    labels_ = np.array(labels_)

    return labels_
#Normalizing the features in the training and testing data

tr_x = tr_x /255

test_x = test_x /255
def model_train(parameters):

    sess = tf.Session()

    

    #important to initialize the variables in order to use it 

    sess.run(tf.global_variables_initializer())

    

    onehot_labels = one_hot(tr_y)

    batch_size = 256

    #Loop through 1000 epochs

    for i in range(0, 1000):    

        #Train the network in batchse

        for j in range(0, 60000, batch_size):

            batch_features = tr_x[j:j+batch_size, :]

            batch_onehot_labels = onehot_labels[j:j+batch_size, :]

            sess.run(train_op, feed_dict = {x: batch_features, y: batch_onehot_labels})

            cost_ = sess.run(cost, feed_dict = {x: batch_features, y: batch_onehot_labels})

            accuracy_ = sess.run(accuracy, feed_dict = {x: batch_features, y: batch_onehot_labels})

            

            if j % 2048 == 0:

                print("At j:{0}, the accuracy:{1}".format(j, accuracy_))

        print("Reached epoch",i ,"cost J = ", cost_)

        

    # lets save the parameters in a variable

    parameters = sess.run(parameters)

    print("\n\nParameters have been trained!") 

    return parameters,sess
parameters,sess = model_train(parameters)