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

tf.reset_default_graph()
import sys
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils/gans/')
from generators import conditional_generator
from discriminators import conditional_discriminator
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
# Similar to last example, but this one includes labels
my_dataset = tf.data.Dataset.from_tensor_slices((image_array, labels_array))
real_images, one_hot_labels = dataset_to_stream(my_dataset, batch_size)
noise_dims = 64
conditional_gan_model = tfgan.gan_model(
    generator_fn=conditional_generator,
    discriminator_fn=conditional_discriminator,
    real_data=real_images,
    generator_inputs=(tf.random_normal([batch_size, noise_dims]), 
                      one_hot_labels))

generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_loss = tfgan.gan_loss(conditional_gan_model, gradient_penalty_weight=1.0)
gan_train_ops = tfgan.gan_train_ops(
    conditional_gan_model,
    gan_loss,
    generator_optimizer,
    discriminator_optimizer)


# Set up class-conditional visualization. We feed class labels to the generator
# so that the the first column is `0`, the second column is `1`, etc.
images_to_eval = 20
assert images_to_eval % 10 == 0

random_noise = tf.random_normal([images_to_eval, 64])
one_hot_labels = tf.one_hot(
    [i for _ in range(images_to_eval // 10) for i in range(10)], depth=10) 
with tf.variable_scope('Generator', reuse=True):
    eval_images = conditional_gan_model.generator_fn((random_noise, one_hot_labels))
reshaped_eval_imgs = tfgan.eval.image_reshaper(
    eval_images[:images_to_eval, ...], num_cols=10)
g_d_updates_per_step = tfgan.GANTrainSteps(1,2)  # do 1 gen step, then 2 disc steps.  
train_step_fn = tfgan.get_sequential_train_steps(g_d_updates_per_step)
global_step = tf.train.get_or_create_global_step()

with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in range(1501):
        loss, done_training = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % 100 == 0:
            digits_np = sess.run([reshaped_eval_imgs])
            visualize_training_generator(i, start_time, digits_np)