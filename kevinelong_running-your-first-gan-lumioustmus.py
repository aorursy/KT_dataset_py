%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, LeakyReLU
from keras.utils.np_utils import to_categorical   

# Main TFGAN library.
tfgan = tf.contrib.gan

tf.set_random_seed(0)
tf.reset_default_graph()
#from learntools.gans.generators import basic_generator
#from learntools.gans.discriminators import basic_discriminator
#from learntools.gans.gan_utils import visualize_training_generator, dataset_to_stream

import sys
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils/gans/')
from generators import basic_generator
from discriminators import basic_discriminator
from gan_utils import visualize_training_generator, dataset_to_stream
train_fname = '../input/digit-recognizer/train.csv'
# Size of each digit
img_rows, img_cols = 28, 28
# Target has 10 values corresponding to 10 numbers (0, 1, 2 ... 9)
num_classes = 10
# Choice of batch size is not critical
batch_size = 40

raw = pd.read_csv(train_fname)
num_images = raw.shape[0]
x_as_array = raw.values[:,1:]
# Reshape from 1 vector into an image. Last dimension shows it is greyscale, which is 1 channel
x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
# Optimization with default params is better when vals scaled to [-1, 1]
image_array = ((x_shaped_array - 128)/ 128).astype(np.float32)
# set up target
labels_array = to_categorical(raw.values[:,0], num_classes=10)

# following 2 lines create the iterator/stream of tensors consumed in model training
my_dataset = tf.data.Dataset.from_tensor_slices((image_array))
batched_dataset = dataset_to_stream(my_dataset, batch_size)
noise_dims = 64
gan_model = tfgan.gan_model(
    basic_generator,
    basic_discriminator,
    real_data=batched_dataset,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))
# Example of classical loss function.
#vanilla_gan_loss = tfgan.gan_loss(
#    gan_model,
#    generator_loss_fn=tfgan.losses.minimax_generator_loss,
#    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)

# Wasserstein loss (https://arxiv.org/abs/1701.07875) with the 
# gradient penalty from the improved Wasserstein loss paper 
# (https://arxiv.org/abs/1704.00028).
improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)
generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)
images_to_eval = 20

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(tf.random_normal([images_to_eval, noise_dims]))

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(eval_images[:images_to_eval,...], num_cols=10)
g_d_updates_per_step = tfgan.GANTrainSteps(1,2)  # do 1 gen step, then 2 disc steps.  
train_step_fn = tfgan.get_sequential_train_steps(g_d_updates_per_step)

global_step = tf.train.get_or_create_global_step()

n_batches = 1501
with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in range(n_batches):
        train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % 100 == 0:
            digits_np = sess.run([generated_data_to_visualize])
            visualize_training_generator(i, start_time, digits_np)