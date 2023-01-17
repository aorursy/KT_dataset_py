import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, Reshape, Conv2DTranspose, LeakyReLU, concatenate

# Main TFGAN library
tfgan = tf.contrib.gan

tf.reset_default_graph()
#from learntools.gans.generators import encoder_decoder_generator
#from learntools.gans.discriminators import basic_discriminator
#from learntools.gans.gan_utils import visualize_training_generator, dataset_to_stream, parse_img_dir

import sys
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils/gans/')
from generators import encoder_decoder_generator
from discriminators import basic_discriminator
from gan_utils import visualize_training_generator, dataset_to_stream, parse_img_dir
# Start with a dataset of directory names.
output_height = 64
output_width = 64
batch_size = 32
max_epochs = 50
imgs_to_visualize=6

train_dir = '../input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/train'
left_img_provider, right_img_provider = parse_img_dir(train_dir, output_height, output_width, batch_size, max_epochs)

# Validation data
val_dir = '../input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/val'
val_left_img_provider, val_right_img_provider = parse_img_dir(val_dir, output_height, output_width, batch_size, max_epochs)
# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=encoder_decoder_generator,
    discriminator_fn=basic_discriminator,
    real_data=left_img_provider,
    generator_inputs=right_img_provider)

# Build the GAN loss and standard pixel loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    #generator_loss_fn=tfgan.losses.least_squares_generator_loss,
    #discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss, gradient_penalty_weight=1.0
    )

generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer,
    discriminator_optimizer)    
# l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# Modify the loss tuple to include the pixel loss.
# gan_loss = tfgan.losses.combine_adversarial_loss(gan_loss, gan_model, l1_pixel_loss, gradient_ratio=1)
# Set up visualization

with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(right_img_provider)

reshaped_cond_val_images = tfgan.eval.image_reshaper(val_right_img_provider[:imgs_to_visualize, ...], num_cols=1)
reshaped_gen_imgs = tfgan.eval.image_reshaper(eval_images[:imgs_to_visualize, ...], num_cols=1)
reshaped_goal_val_images = tfgan.eval.image_reshaper(val_left_img_provider[:imgs_to_visualize, ...], num_cols=1)
g_d_updates_per_step = tfgan.GANTrainSteps(1,2)  # do 1 gen step, then 2 disc steps.  
train_step_fn = tfgan.get_sequential_train_steps(g_d_updates_per_step) # default is 1:1 ratio

global_step = tf.train.get_or_create_global_step()

num_steps = 3001
with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in range(num_steps):
        loss, done_training = train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % 100 == 0:
            plottables = sess.run([reshaped_cond_val_images, reshaped_gen_imgs, reshaped_goal_val_images])
            visualize_training_generator(i, start_time, plottables, undo_normalization=True)