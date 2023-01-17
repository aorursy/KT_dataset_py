import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Import MNIST data

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

x_train = np.array(mnist_train.iloc[:, 1:]).reshape(-1, 28*28) / 255

# y_train = np.array(mnist_train.iloc[:, 0])

x_test = np.array(mnist_test.iloc[:, 1:]).reshape(-1, 28*28) / 255

# y_test = np.array(mnist_test.iloc[:, 0])

n_train_samples = x_train.shape[0]



# Network Parameters

num_hidden_1 = 256 # 1st layer num features

num_hidden_2 = 128 # 2nd layer num features (the latent dim)

num_input = 784 # MNIST data input (img shape: 28*28)



# tf Graph input (only pictures)

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

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    # layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    

    # Encoder Hidden layer with sigmoid activation #2

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    return layer_2



# Building the decoder

def decoder(x):

    # Decoder Hidden layer with sigmoid activation #1

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    # layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    

    # Decoder Hidden layer with sigmoid activation #2

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    return layer_2



# Construct model

embedding = encoder(X)

recon_x = decoder(embedding)



# Prediction

y_pred = recon_x

# Targets (Labels) are the input data.

y_true = X



# Define loss and optimizer, minimize the squared error

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

learning_rate = tf.placeholder("float32")

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)



print('Autoencoder is constructed.')
# Training Parameters

learning_rate_AE = 0.01

max_epoch = 10000

# max_epoch = 50000

batch_size = 256



display_step = 1000



# Start Training

# Start a new TF session

with tf.Session() as sess:

    # Initialize the variables (i.e. assign their default value)

    sess.run(tf.global_variables_initializer())



    # Training

    for i in range(max_epoch):

        # Prepare a batch

        tmp_index = np.random.permutation(n_train_samples)

        batch_x = x_train[tmp_index[:batch_size], :]

        # Run optimization op (backprop) and cost op (to get loss value)

        _, l = sess.run([optimizer, loss], feed_dict={X:batch_x, learning_rate:learning_rate_AE})

        # Display logs per step

        if i % display_step == 0:

            print('Step %i: Minibatch Loss: %f' % (i, l))



    # Testing

    # Encode and decode images from test set and visualize their reconstruction.

    n = 4

    img_orig = np.empty((28 * n, 28 * n))

    img_recon = np.empty((28 * n, 28 * n))

    for i in range(n):

        # MNIST test set

        batch_x = x_test[i*n:(i+1)*n, :]

        # Encode and decode the digit image

        g = sess.run(recon_x, feed_dict={X:batch_x})



        # Display original images

        for j in range(n):

            # Draw the original digits

            img_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])

        # Display reconstructed images

        for j in range(n):

            # Draw the reconstructed digits

            img_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])



    print("Original Images")

    plt.figure(figsize=(n, n))

    plt.imshow(img_orig, origin="upper", cmap="gray")

    plt.show()



    print("Reconstructed Images")

    plt.figure(figsize=(n, n))

    plt.imshow(img_recon, origin="upper", cmap="gray")

    plt.show()