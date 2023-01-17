import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv('../input/fashion-mnist_train.csv')

df.head()
# first 500 sneakers

sneakers = df.loc[df['label'] == 7].iloc[:500, 1:].values.copy()

sneakers = sneakers / 255.0

# first 500 shirts

shirt = df.loc[df['label'] == 6].iloc[:500, 1:].values.copy()

shirt = shirt / 255.0



sneakers = sneakers.reshape((-1, 28, 28, 1))

shirt = shirt.reshape((-1, 28, 28, 1))
# example print image

fig = plt.figure(figsize = (9, 3))

plt.subplot(1, 3, 1)

plt.imshow(sneakers[0, :, :, :].reshape((28, 28)))

plt.title('sneaker 1')

plt.subplot(1, 3, 2)

plt.imshow(shirt[0, :, :, :].reshape((28, 28)))

plt.title('shirt 1')

plt.subplot(1, 3, 3)

plt.imshow(sneakers[1, :, :, :].reshape((28, 28)))

plt.title('sneaker 2')

plt.show()
def generate_sample(samples):

    # take first 16 samples

    idx = [i for i in range(16)]

    fig, axes = plt.subplots(4, 4, sharex = True, sharey = True, figsize = (5,5))



    for ii, ax in zip(idx, axes.flatten()):

        ax.imshow(samples[ii,:].reshape((28, 28)), aspect = 'equal')

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)



    plt.subplots_adjust(wspace = 0, hspace = 0)

    plt.show()
# compress size

encoding_dim = 256

# 28 * 28

image_size = 784



def huber_loss(logits, labels, max_gradient = 1.0):

    err = tf.abs(labels - logits)

    mg = tf.constant(max_gradient)

    lin = mg * (err - 0.5 * mg)

    quad = 0.5 * err * err

    return tf.where(err < mg, quad, lin)



# multi-perceptron encoder

def generator(z, name, reuse = False, training = True):

    with tf.variable_scope(name, reuse = reuse):

        conv1 = tf.layers.conv2d(z, 16, (3, 3), padding = 'same', activation = tf.nn.relu)

        maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding = 'same')

        conv2 = tf.layers.conv2d(maxpool1, 8, (3, 3), padding = 'same', activation = tf.nn.relu)

        maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2, 2), padding = 'same')

        conv3 = tf.layers.conv2d(maxpool2, 8, (3, 3), padding = 'same', activation = tf.nn.relu)

        encoded = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding = 'same')

        upsample1 = tf.image.resize_nearest_neighbor(encoded, (7, 7))

        conv4 = tf.layers.conv2d(upsample1, 8, (3, 3), padding = 'same', activation = tf.nn.relu)

        upsample2 = tf.image.resize_nearest_neighbor(conv4, (14, 14))

        conv5 = tf.layers.conv2d(upsample2, 8, (3, 3), padding = 'same', activation = tf.nn.relu)

        upsample3 = tf.image.resize_nearest_neighbor(conv5, (28, 28))

        conv6 = tf.layers.conv2d(upsample3, 16, (3, 3), padding = 'same', activation = tf.nn.relu)

        logits = tf.layers.conv2d(conv6, 1, (3, 3), padding = 'same', activation = None)

        return tf.nn.sigmoid(logits)

        

def discriminator(z, name, reuse = False):

    with tf.variable_scope(name, reuse = reuse):

        x1 = tf.layers.conv2d(z, 8, 5, strides = 2, padding = 'same')

        relu1 = tf.nn.relu(x1)

        x2 = tf.layers.conv2d(relu1, 64, 5, strides = 2, padding = 'same')

        relu2 = tf.nn.relu(x2)

        x3 = tf.layers.conv2d(relu2, 128, 5, strides = 2, padding = 'same')

        relu3 = tf.nn.relu(x3)

        flat = tf.reshape(relu3, (-1, 4 * 4 * 128))

        return tf.layers.dense(flat, 1)
class discoGAN:

    # set learning rate here

    def __init__(self, learning_rate = 1e-6):

        # first domain

        self.X = tf.placeholder(tf.float32, (None, 28, 28, 1))

        # second domain

        self.Y = tf.placeholder(tf.float32, (None, 28, 28, 1))

        g_AB_model = generator(self.X, 'generator_AB')

        g_BA_model = generator(self.Y, 'generator_BA')

        self.g_out_AB = generator(self.X, 'generator_AB', reuse = True, training = False)

        self.g_out_BA = generator(self.Y, 'generator_BA', reuse = True, training = False)

        g_huber_A = generator(g_AB_model, 'generator_BA', reuse = True)

        g_huber_B = generator(g_BA_model, 'generator_AB', reuse = True)

        l_const_a = tf.reduce_mean(huber_loss(g_huber_A, self.X))

        l_const_b = tf.reduce_mean(huber_loss(g_huber_B, self.Y))

        d_logits_real_A = discriminator(self.Y, 'discriminator_A')

        d_logits_fake_A = discriminator(g_AB_model, 'discriminator_A', reuse = True)

        d_logits_real_B = discriminator(self.X, 'discriminator_B')

        d_logits_fake_B = discriminator(g_BA_model, 'discriminator_B', reuse = True)

        

        # ian goodfellow cost function, policy based

        d_loss_real_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real_A, 

                                                                               labels = tf.ones_like(d_logits_real_A)))

        d_loss_fake_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake_A, 

                                                                               labels = tf.zeros_like(d_logits_fake_A)))

        # maximise generatorAB output

        self.g_loss_AB = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake_A, 

                                                                                labels = tf.ones_like(d_logits_real_A)))

        

        d_loss_real_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real_B, 

                                                                               labels = tf.ones_like(d_logits_real_B)))

        d_loss_fake_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake_B, 

                                                                               labels = tf.zeros_like(d_logits_fake_B)))

        # maximise generatorBA output

        self.g_loss_BA = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake_B, 

                                                                                labels = tf.ones_like(d_logits_real_B)))

        

        self.d_loss = d_loss_real_A + d_loss_fake_A + d_loss_real_B + d_loss_fake_B

        self.g_loss = self.g_loss_AB + self.g_loss_BA + l_const_a + l_const_b

        

        t_vars = tf.trainable_variables()

        d_vars_A = [var for var in t_vars if var.name.startswith('discriminator_A')]

        d_vars_B = [var for var in t_vars if var.name.startswith('discriminator_B')]

        g_vars_AB = [var for var in t_vars if var.name.startswith('generator_AB')]

        g_vars_BA = [var for var in t_vars if var.name.startswith('generator_BA')]

        

        self.d_train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.d_loss, var_list = d_vars_A + d_vars_B)

        self.g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.999, beta2 = 0.999).minimize(self.g_loss, var_list = g_vars_AB + g_vars_BA)
def train(model, X, Y, batch, epoch):

    LOSS_D, LOSS_G = [], []

    for i in range(epoch):

        g_loss, d_loss = 0, 0

        for k in range(0, (X.shape[0] // batch) * batch, batch):

            batch_x = X[k: k + batch, :, :, :]

            batch_y = Y[k: k + batch, :, :, :]

            _, lossd = sess.run([model.d_train_opt, model.d_loss], 

                                feed_dict = {model.X: batch_x, model.Y: batch_y})

            _, lossg = sess.run([model.g_train_opt, model.g_loss], 

                                feed_dict = {model.X: batch_x, model.Y: batch_y})

            g_loss += lossg

            d_loss += lossd

        g_loss /= (X.shape[0] // batch)

        d_loss /= (X.shape[0] // batch)

        print("Epoch {}/{}".format(i + 1, EPOCH), 

              "Discriminator Loss: {}".format(d_loss), 

              "Generator Loss: {}".format(g_loss))

        LOSS_G.append(g_loss); LOSS_D.append(d_loss)

        

        if (i + 1) % 10 == 0:

            # 16 pictures because our sample function above only accept 16 samples

            batch_x = X[:16, :, :, :]

            batch_y = Y[:16, :, :, :]

            outputs = sess.run(model.g_out_AB, feed_dict = {model.X: batch_x})

            print('GENERATOR A')

            generate_sample(outputs)

            print('GENERATOR B')

            outputs = sess.run(model.g_out_BA, feed_dict = {model.Y: batch_y})

            generate_sample(outputs)

    

    epoch = [i for i in range(len(LOSS_D))]

    plt.plot(epoch,LOSS_D, label = 'Discriminator', alpha = 0.5)

    plt.plot(epoch, LOSS_G, label = 'Generator', alpha = 0.5)

    plt.title("Training Losses")

    plt.legend()

    plt.show()
EPOCH = 30

BATCH_SIZE = 20

tf.reset_default_graph()

sess = tf.InteractiveSession()

model = discoGAN()

sess.run(tf.global_variables_initializer())

train(model, sneakers, shirt, BATCH_SIZE, EPOCH)