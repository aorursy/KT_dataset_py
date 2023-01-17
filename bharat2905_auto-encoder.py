%matplotlib inline
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
img = mnist.train.images[2]
plt.imshow(img.reshape((28,28)),cmap='Greys_r')
#ARCHITECTURE
learning_rate=0.01

inputs=tf.placeholder(tf.float32,(None,28,28,1),name="inputs")

targets=tf.placeholder(tf.float32,(None,28,28,1),name="targets")



conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')

conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')

conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')

upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)

decoded = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)

cost = tf.reduce_mean(loss)

opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
sess = tf.Session()
epochs = 20

batch_size = 100

noise_factor = 0.5

sess.run(tf.global_variables_initializer())

for e in range(epochs):

    for ii in range(mnist.train.num_examples//batch_size):

        batch = mnist.train.next_batch(batch_size)

        imgs = batch[0].reshape((-1, 28, 28, 1))

        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)

        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs: noisy_imgs,

                                                         targets: imgs})



        print("Epoch: {}/{}...".format(e+1, epochs),

              "Training loss: {:.4f}".format(batch_cost))
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))

in_imgs = mnist.test.images[:10]

noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)

noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(decoded, feed_dict={inputs: noisy_imgs.reshape((10, 28, 28, 1))})



for images, row in zip([noisy_imgs, reconstructed], axes):

    for img, ax in zip(images, row):

        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)





fig.tight_layout(pad=0.1)


fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))

in_imgs = mnist.test.images[:10]

reconstructed = sess.run(decoded, feed_dict={inputs: in_imgs.reshape((10, 28, 28, 1))})



for images, row in zip([in_imgs, reconstructed], axes):

    for img, ax in zip(images, row):

        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)





fig.tight_layout(pad=0.1)