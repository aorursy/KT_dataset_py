from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import os

import warnings

import copy

# warnings.simplefilter(action="ignore")

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt

from matplotlib import figure 

from matplotlib.backends import backend_agg

from tensorflow.examples.tutorials.mnist import input_data

import seaborn as sns

import tensorflow as tf

import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

# Dependency imports

import matplotlib

import tensorflow_probability as tfp

#matplotlib.use("Agg")



%matplotlib inline
IMAGE_SHAPE = [28, 28, 1]
data_dir = '../input/'



#mnist_onehot = input_data.read_data_sets(data_dir, one_hot=True)

mnist_conv = input_data.read_data_sets(data_dir,reshape=False ,one_hot=False)

#mnist_conv_onehot = input_data.read_data_sets(data_dir,reshape=False ,one_hot=True)



# create a set of non-shuffled test images for reference

fixed_val = np.zeros((1000,28,28,1))

for img_no in range(1000):

    

    fixed_val[img_no] = mnist_conv.validation.images[img_no].copy()



# create a fixed set of images with random noise added, for a poor man's simulation of an adversarial attach

perturbation_magnitude = 0.5

np.random.seed(42)

perturbation = np.random.random((28,28,1))

perturbed_val = np.zeros((1000, 28, 28, 1))

for img_no in range(1000):

    perturbed_val[img_no] = mnist_conv.validation.images[img_no].copy() + perturbation_magnitude * perturbation
noisy_images = np.zeros((10000,28,28,1))

for i in range(10000):

    

    noisy_images[i] = np.random.random((28,28,1))   # Test data

    

plt.imshow(noisy_images[0].reshape((28,28)), cmap='gist_gray')
learning_rate = 0.001   #initial learning rate

max_step = 50000 #number of training steps to run

batch_size = 32 #batch size

num_monte_carlo = 500 #Network draws to compute predictive probabilities.
images = tf.placeholder(tf.float32,shape=[None,28,28,1])

labels = tf.placeholder(tf.float32,shape=[None,])

hold_prob = tf.placeholder(tf.float32)

# define the model

neural_net = tf.keras.Sequential([

      tfp.layers.Convolution2DReparameterization(32, kernel_size=5,  padding="SAME", activation=tf.nn.relu),

      tf.keras.layers.MaxPooling2D(pool_size=[2, 2],  strides=[2, 2],  padding="SAME"),

      tfp.layers.Convolution2DReparameterization(64, kernel_size=5,  padding="SAME",  activation=tf.nn.relu),

      tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),

      tf.keras.layers.Flatten(),

      tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),

      tf.keras.layers.Dropout(hold_prob),

      tfp.layers.DenseFlipout(10)])

logits = neural_net(images)



# Compute the -ELBO as the loss, averaged over the batch size.

labels_distribution = tfp.distributions.Categorical(logits=logits)

neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))

kl = sum(neural_net.losses) / mnist_conv.train.num_examples

elbo_loss = neg_log_likelihood + kl

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(elbo_loss)

# Build metrics for evaluation. Predictions are formed from a single forward

# pass of the probabilistic layers. They are cheap but noisy predictions.

predictions = tf.argmax(logits, axis=1)

accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)



# Extract weight posterior statistics for layers with weight distributions

# for later visualization.

names = []

qmeans = []

qstds = []

for i, layer in enumerate(neural_net.layers):

    try:

        q = layer.kernel_posterior

    except AttributeError:

        continue

    names.append("Layer {}".format(i))

    qmeans.append(q.mean())

    qstds.append(q.stddev())
def plot_image_and_prob(image_vals, image_indices, probs):

    # utility to plot some images and the probability distributions

    import pandas as pd

    mean_probs = np.mean(probs, axis=0)

    std_probs = np.std(probs, axis=0)

    

    for image_index in image_indices:

        this_image = pd.DataFrame(probs[:,image_index,:]).melt()

        fig, ax = plt.subplots(ncols=3, figsize=(8,4))

        ax[0].imshow(image_vals[image_index][:,:,0], cmap='gist_gray');

        sns.barplot(np.arange(10), mean_probs[image_index], ax=ax[1])

        sns.boxplot('variable', 'value', data=this_image, ax=ax[2], fliersize=0)

        for patch in ax[2].artists:

            r, g, b, a = patch.get_facecolor()

            patch.set_facecolor((r, g, b, .0))

        sns.swarmplot('variable', 'value', data=this_image, ax=ax[2], alpha=0.25)

        

        ax[1].set_ylim([0,1])

        ax[1].set_ylabel('Class "Probability"')

        ax[1].set_xlabel('Class')

        ax[2].set_xlabel('Class')

        ax[1].set_ylabel('')

        ax[2].set_ylim([0,1])

        ax[0].set_title('Image #%i' % image_index)

        fig.tight_layout()

        display()

        plt.show();

        #print(mean_probs[image_index])
noisy_data = tf.data.Dataset.from_tensor_slices((noisy_images))

noisy_data = noisy_data.batch(len(noisy_images))

noisy_data_iterator = noisy_data.make_one_shot_iterator()

noisy_data_vals = noisy_data_iterator.get_next()



fixed_val_data = tf.data.Dataset.from_tensor_slices((fixed_val))

fixed_val_data = fixed_val_data.batch(len(fixed_val))

fixed_val_iterator = fixed_val_data.make_one_shot_iterator()

fixed_val_vals = fixed_val_iterator.get_next()



init_op = tf.group(tf.global_variables_initializer(),

                   tf.local_variables_initializer())



with tf.Session() as sess:

    sess.run(init_op)

    

    for step in range(max_step):

        images_b, labels_b = mnist_conv.train.next_batch(batch_size)

        images_h, labels_h = mnist_conv.validation.next_batch(mnist_conv.validation.num_examples)

            

        _ = sess.run([train_op, accuracy_update_op], feed_dict={images: images_b, labels: labels_b, hold_prob:0.5})

        if (step==0) | ((step + 1) % 500 == 0):

            loss_value, accuracy_value = sess.run([elbo_loss, accuracy], feed_dict={images: images_b, labels: labels_b, hold_prob:0.5})

            print(step + 1, loss_value, accuracy_value)

    

    # selection of intertesing mnist images

    image_vals = sess.run(fixed_val_vals)

    probs = np.asarray([sess.run((labels_distribution.probs),

                                 feed_dict={images: image_vals, hold_prob:0.5}

                                )

                        for _ in range(num_monte_carlo)])

    interesting_image_indices = [24, 29, 30, 48, 53, 54, 63, 67, 70, 80]

    plot_image_and_prob(image_vals, interesting_image_indices, probs)



    # find the worst case for noisy data and show

    image_vals = sess.run(noisy_data_vals)

    probs = np.asarray([sess.run((labels_distribution.probs),

                                 feed_dict={images: image_vals, hold_prob:0.5}

                                )

                        for _ in range(num_monte_carlo)])

    trouble_index = np.expand_dims(probs.mean(axis=0), axis=0)[0].max(axis=1).argmax()

    plot_image_and_prob(image_vals, [trouble_index], probs)

def plot_image_and_prob_compare(image0_vals, image0_probs, image1_vals, image1_probs, image_indices):

    import pandas as pd

    mean_probs0 = np.mean(image0_probs, axis=0)

    std_probs0 = np.std(image0_probs, axis=0)

    mean_probs1 = np.mean(image1_probs, axis=0)

    std_probs1 = np.std(image1_probs, axis=0)

    

    for image_index in image_indices:

        

        fig, ax = plt.subplots(ncols=6, figsize=(16,4))

        this_image0 = pd.DataFrame(image0_probs[:,image_index,:]).melt()

        ax[0].imshow(image0_vals[image_index][:,:,0], cmap='gist_gray');

        sns.barplot(np.arange(10), mean_probs0[image_index], ax=ax[1])

        sns.boxplot('variable', 'value', data=this_image0, ax=ax[2], fliersize=0)

        for patch in ax[2].artists:

            r, g, b, a = patch.get_facecolor()

            patch.set_facecolor((r, g, b, .0))

        sns.swarmplot('variable', 'value', data=this_image0, ax=ax[2], alpha=0.25)

        

        ax[1].set_ylim([0,1])

        ax[1].set_ylabel('Class "Probability"')

        ax[1].set_xlabel('Class')

        ax[2].set_xlabel('Class')

        ax[1].set_ylabel('')

        ax[2].set_ylim([0,1])

        ax[0].set_title('Image #%i' % image_index)

        

        this_image1 = pd.DataFrame(image1_probs[:,image_index,:]).melt()

        ax[3].imshow(image1_vals[image_index][:,:,0], cmap='gist_gray');

        sns.barplot(np.arange(10), mean_probs1[image_index], ax=ax[4])

        sns.boxplot('variable', 'value', data=this_image1, ax=ax[5], fliersize=0)

        for patch in ax[5].artists:

            r, g, b, a = patch.get_facecolor()

            patch.set_facecolor((r, g, b, .0))

        sns.swarmplot('variable', 'value', data=this_image1, ax=ax[5], alpha=0.25)

        

        ax[4].set_ylim([0,1])

        ax[4].set_ylabel('Class "Probability"')

        ax[4].set_xlabel('Class')

        ax[5].set_xlabel('Class')

        ax[4].set_ylabel('')

        ax[5].set_ylim([0,1])

        ax[3].set_title('Image #%i' % image_index)

        

        fig.tight_layout()

        display()

        plt.show();



fixed_val_data = tf.data.Dataset.from_tensor_slices((fixed_val))

fixed_val_data = fixed_val_data.batch(len(fixed_val))

fixed_val_iterator = fixed_val_data.make_one_shot_iterator()

fixed_val_vals = fixed_val_iterator.get_next()



perturbed_val_data = tf.data.Dataset.from_tensor_slices((perturbed_val))

perturbed_val_data = perturbed_val_data.batch(len(perturbed_val))

perturbed_val_iterator = perturbed_val_data.make_one_shot_iterator()

perturbed_val_vals = perturbed_val_iterator.get_next()



init_op = tf.group(tf.global_variables_initializer(),

                   tf.local_variables_initializer())



with tf.Session() as sess:

    sess.run(init_op)

    

    for step in range(1000):#max_step):

        images_b, labels_b = mnist_conv.train.next_batch(batch_size)

        images_h, labels_h = mnist_conv.validation.next_batch(mnist_conv.validation.num_examples)

            

        _ = sess.run([train_op, accuracy_update_op], feed_dict={images: images_b, labels: labels_b, hold_prob:0.5})

        if (step==0) | ((step + 1) % 500 == 0):

            loss_value, accuracy_value = sess.run([elbo_loss, accuracy], feed_dict={images: images_b, labels: labels_b, hold_prob:0.5})

            print(step + 1, loss_value, accuracy_value)

            

    original_vals = sess.run(fixed_val_vals)

    perturbed_vals = sess.run(perturbed_val_vals)

    

    original_probs = np.asarray([sess.run((labels_distribution.probs),

                                 feed_dict={images: original_vals, hold_prob:0.5}

                                )

                        for _ in range(num_monte_carlo)])

    perturbed_probs = np.asarray([sess.run((labels_distribution.probs),

                                 feed_dict={images: perturbed_vals, hold_prob:0.5}

                                )

                        for _ in range(num_monte_carlo)])

    

    # find a pathological case where adding noise drastically changes the classification

    sensitive_index = ((np.expand_dims(original_probs.mean(axis=0), axis=0)[0].max(axis=1) < 0.4)*999. + # big number if no probability above threshold

                       (np.expand_dims(perturbed_probs.mean(axis=0), axis=0)[0].max(axis=1) < 0.4)*999. + # big number if no probability above threshold

                       np.abs(np.expand_dims(original_probs.mean(axis=0), axis=0)[0] * np.expand_dims(perturbed_probs.mean(axis=0), axis=0)[0]).sum(axis=1)).argmin()

    

    plot_image_and_prob_compare(original_vals, original_probs, perturbed_vals, perturbed_probs, [sensitive_index])
