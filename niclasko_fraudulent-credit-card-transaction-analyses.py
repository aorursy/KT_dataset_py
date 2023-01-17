import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from IPython.display import clear_output, Image, display, HTML

from matplotlib import pyplot
# Read data

data = pd.read_csv("../input/creditcard.csv")

auto_encoder_variables = [s for s in data.columns if "V" in s]



# Print first 10

data.loc[1:10,auto_encoder_variables]
# Training Parameters

learning_rate = 0.01

num_steps = 100

batch_size = 100

display_step = 10



# Network Parameters

num_input = len(auto_encoder_variables)

num_hidden_1 = len(auto_encoder_variables) # 1st layer num features

num_hidden_2 = 2 # 2nd layer num features (the latent dim)



# tf Graph input

X = tf.placeholder("float", [None, num_input])



weights = {

    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),

    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),

    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),

    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),

}

biases = {

    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),

    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),

    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),

    'decoder_b2': tf.Variable(tf.random_normal([num_input])),

}



# Building the encoder

def encoder(x):

    # Encoder Hidden layer with sigmoid activation #1

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),

                                   biases['encoder_b1']))

    # Encoder Hidden layer with sigmoid activation #2

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),

                                   biases['encoder_b2']))

    return layer_2





# Building the decoder

def decoder(x):

    # Decoder Hidden layer with sigmoid activation #1

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),

                                   biases['decoder_b1']))

    # Decoder Hidden layer with sigmoid activation #2

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),

                                   biases['decoder_b2']))

    return layer_2



# Construct model

encoder_op = encoder(X)

decoder_op = decoder(encoder_op)



# Prediction

y_pred = decoder_op

# Targets (Labels) are the input data.

y_true = X



# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()



encoded = None



# Start Training

# Start a new TF session

with tf.Session() as sess:

    # Run the initializer

    sess.run(init)

    # Training

    for i in range(1, num_steps+1):

        x_batch = data[auto_encoder_variables].sample(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)

        _, l = sess.run([optimizer, loss], feed_dict={X: x_batch})

        # Display logs per step

        if i % display_step == 0 or i == 1:

            print('Step %i: Minibatch Loss: %f' % (i, l))

    

    # Encode data using auto-encoder neural network

    encoded = pd.DataFrame(

        sess.run(encoder_op, feed_dict={X: data[auto_encoder_variables]}),

        columns = ["X", "Y"]

    )
# Print 10 first encoded variables

encoded.loc[1:10,:]
pyplot.scatter(

    encoded["X"],

    encoded["Y"],

    c=data["Class"],

    s=0.005

)



pyplot.show()