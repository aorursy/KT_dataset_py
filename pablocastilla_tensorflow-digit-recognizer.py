import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn

import tensorflow as tf

from subprocess import check_output

from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split
df = pd.read_csv("../input/train.csv")

# Split data into training set and validation set

y_train = df.ix[:,0].values

x_train = df.ix[:,1:].values



#One Hot encoding of labels.

x_train = x_train.reshape(42000,28, 28)

y_train = to_categorical(y_train)



x_train, x_test, y_train, y_test = train_test_split( x_train, y_train, test_size=0.20, random_state=42)

print(x_train.shape, y_train.shape,x_test.shape,y_test.shape)
tf.reset_default_graph()



# Parameters

learning_rate = 0.001

training_epochs = 1

batch_size = 128  # Decrease batch size if you don't have enough memory

display_step = 1



n_input = 784  # MNIST data input (img shape: 28*28)

n_classes = 10  # MNIST total classes (0-9 digits)



n_hidden_layer = 256 # layer number of features



# Store layers weight & bias

weights = {

    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),

    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))

}

biases = {

    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),

    'out': tf.Variable(tf.random_normal([n_classes]))

}



# tf Graph input

x = tf.placeholder("float", [None, 28, 28])

y = tf.placeholder("float", [None, n_classes])



x_flat = tf.reshape(x, [-1, n_input])



# Hidden layer with RELU activation

layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])

layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation

logits = tf.matmul(layer_1, weights['out']) + biases['out']



# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)



# Initializing the variables

init = tf.global_variables_initializer()



# Launch the graph

with tf.Session() as sess:

    sess.run(init)

    # Training cycle

    for epoch in range(training_epochs):

        total_batch = int(x_train.shape[0]/batch_size)

        # Loop over all batches

        for i in range(total_batch):

            batch_x = x_train[batch_size:(i+1)*batch_size]

            batch_y = y_train[batch_size:(i+1)*batch_size]

            

            # Run optimization op (backprop) and cost op (to get loss value)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # Display logs per epoch step

        if epoch % display_step == 0:

            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

            print("Epoch:", '%04d' % (epoch+1), "cost=", \

                "{:.9f}".format(c))

    print("Optimization Finished!")



    # Test model

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Decrease test_size if you don't have enough memory

    test_size = 256

    print("Accuracy:", accuracy.eval({x: x_test[:test_size], y: y_test[:test_size]}))