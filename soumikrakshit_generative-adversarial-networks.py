import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go
from tqdm import tqdm
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
len(mnist.train.images)
def generator(noise, reuse = None):
    with tf.variable_scope('gen', reuse = reuse):
        hidden1 = tf.layers.dense(inputs = noise, units = 128)
        hidden1 = tf.maximum(0.01 * hidden1, hidden1) # LeakyReLu Activation
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
        hidden2 = tf.maximum(0.01 * hidden2, hidden2) # LeakyReLu Activation
        output = tf.layers.dense(inputs = hidden2, units = 784, activation = tf.nn.tanh)
        return output
def discriminator(x, reuse = None):
    with tf.variable_scope('dis', reuse = reuse):
        hidden1 = tf.layers.dense(inputs = x, units = 128)
        hidden1 = tf.maximum(0.01 * hidden1, hidden1) # LeakyReLu Activation
        hidden2 = tf.layers.dense(inputs = hidden1, units = 128)
        hidden2 = tf.maximum(0.01 * hidden2, hidden2) # LeakyReLu Activation
        logits = tf.layers.dense(inputs = hidden2, units = 1)
        output = tf.sigmoid(logits)
        return output, logits
real_images = tf.placeholder(tf.float32, shape = [None, 784])
z = tf.placeholder(tf.float32, shape = [None, 100])
G = generator(z)
disc_output_real, disc_logits_real = discriminator(real_images)
disc_output_fake, disc_logits_fake = discriminator(G, reuse = True)
def loss_function(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
disc_real_loss = loss_function(disc_logits_real, tf.ones_like(disc_logits_real) * 0.9)
disc_fake_loss = loss_function(disc_logits_fake, tf.zeros_like(disc_logits_fake))
disc_loss = disc_real_loss + disc_fake_loss
gen_loss = loss_function(disc_logits_fake, tf.ones_like(disc_logits_fake))
learning_rate = 0.001
batch_size = 128
epochs = 1000
tvars = tf.trainable_variables()
disc_vars = [var for var in tvars if 'dis' in var.name]
gen_vars = [var for var in tvars if 'gen' in var.name]
disc_train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(disc_loss, var_list = disc_vars)
gen_train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(gen_loss, var_list = gen_vars)
init = tf.global_variables_initializer()
samples, gen_loss_hist, disc_loss_hist = [], [], []
with tf.Session() as sess:
    sess.run(init)
    for epoch in tqdm(range(1, epochs + 1)):
        num_batches = mnist.train.num_examples // batch_size
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1
            batch_noise = np.random.uniform(-1, 1, size = (batch_size, 100))
            _, _ = sess.run([disc_train_op, gen_train_op], feed_dict = {
                real_images : batch_images,
                z : batch_noise
            })
        if epoch == 1 or epoch % 100 == 0:
            gl, dl = sess.run([gen_loss, disc_loss], feed_dict = {
                real_images : batch_images,
                z : batch_noise
            })
            gen_loss_hist.append(gl)
            disc_loss_hist.append(dl)
            print("Epoch: " + str(epoch) + ", Generator Loss: " + str(gen_loss_hist[-1]) + ", Discriminator Loss: " + str(disc_loss_hist[-1]))
        sample_noise = np.random.uniform(-1, 1, size = (1, 100))
        generated_sample = sess.run(G, feed_dict = { z : sample_noise })
        samples.append(generated_sample)
data = [go.Scatter(x = np.array(list(range(len(gen_loss_hist)))) * 100, y = gen_loss_hist, name = "Generator Loss"),
        go.Scatter(x = np.array(list(range(len(disc_loss_hist)))) * 100, y = disc_loss_hist, name = "Discriminator Loss")]
layout = dict(title = "Generator and Discriminator Loss", xaxis = dict(title = "Epoch"), yaxis = dict(title = "Loss"))
iplot(dict(data = data, layout = layout), filename = "GAN-loss")
def sample_batch(batch_size, data):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    return np.asarray(data_shuffle)
def display_images(x, rows, columns, title, figsize):
    fig, axes = plt.subplots(rows, columns, figsize = figsize)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    fig.suptitle(title, fontsize = 18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap = 'binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
display_images(sample_batch(12, np.array(samples)), 2, 6, 'Generated Images', (18, 5))