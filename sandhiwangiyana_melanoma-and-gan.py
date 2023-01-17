import glob

import os

import time



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

print(tf.__version__)



from tensorflow.keras import layers

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from IPython import display
# asign some paths

train_csv_path = '../input/siim-isic-melanoma-classification/train.csv'

test_csv_path = '../input/siim-isic-melanoma-classification/test.csv'

image_path = '../input/siim-isic-melanoma-classification/jpeg/train/'



# read the csv data using pandas

train_df = pd.read_csv(train_csv_path)

test_df = pd.read_csv(test_csv_path)



print("unique values in column 'target': {}".format(list(train_df['target'].unique())))

target_dis = list(train_df['target'].value_counts())

benign_per = target_dis[0]/sum(target_dis)

print("target count distribution: {}".format(target_dis))

print("benign percentage: {:.2f}% vs malignant: {:.2f}%".format(benign_per*100, (1-benign_per)*100))
# detect and initialize TPU (ignore if using GPU)

# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#     print('Device:', tpu.master())

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     # set distribution strategy

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# except:

#     strategy = tf.distribute.get_strategy()

# print('Number of replicas:', strategy.num_replicas_in_sync)



# # Use these params if using TPU

# IMAGE_SIZE = [128, 128]  # used for reshaping

# AUTOTUNE = tf.data.experimental.AUTOTUNE

# GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-128x128')  # store dataset to gcs buckets for the TPU to access in cloud

# BATCH_SIZE = 16 * strategy.num_replicas_in_sync
path_tfrec = '../input/melanoma-128x128/'

path_jpg = '../input/jpeg-melanoma-128x128/train/'

IMAGE_SIZE = [128, 128]



malignant = train_df[train_df["target"] == 1]  # list of malignant images



def preprocess_X():  # load the images into memory

    X = []

    for img in malignant.image_name.values:

        img_name = path_jpg + img + '.jpg'

        i = tf.keras.preprocessing.image.load_img(img_name) #color_mode='grayscale')

        i = tf.keras.preprocessing.image.img_to_array(i)

        i = preprocess_input(i)  # preprocessing fits the pixel value from -127.5 to 127.5

        X.append(i)

    return np.array(X)  # convert to numpy array
X = preprocess_X()

X.shape
def display_img(arr):

    i = tf.keras.preprocessing.image.array_to_img(arr)

    plt.imshow(i, cmap='gray')



plt.figure(figsize=(7,7))

for i in range(9):

    plt.subplot(3,3, i+1)

    display_img(X[i])  
BUFFER_SIZE = 584

BATCH_SIZE = 32  # from 128

EPOCHS = 50  # from 50

noise_dim = 200  # from 100

num_examples_to_generate = 9



# We will reuse this seed overtime (so it's easier to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim])
def augmentation_pipeline(image):

    image = tf.image.random_flip_left_right(image)

#     image = tf.image.resize(image, IMAGE_RESIZE)

    return image
# Simple dataset processing with batch and shuffle

def get_dataset():

    ds = tf.data.Dataset.from_tensor_slices(X)

#     ds = ds.map(augmentation_pipeline)

    ds = ds.shuffle(BUFFER_SIZE)

    ds = ds.batch(BATCH_SIZE)

    return ds

    

train_dataset = get_dataset()

# inspect a batch

n_batch = 0

for i in train_dataset:

    n_batch += 1

print(f"num of batch: {n_batch}, shape of each batch: {i.shape}")
def make_generator_model():

    model = tf.keras.Sequential()   # dense unit is configured to match soon tobe reshaped layer

    model.add(layers.Dense(32*32*256, use_bias=False, input_shape=(noise_dim,)))  # starts with 1D array, input is noise array of 100

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    # 32x32 bcz there's 2 conv2D. 128/2/2=32

    model.add(layers.Reshape((32, 32, 256)))



    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))



    return model



# create the generator

generator = make_generator_model()

generator.summary()
noise = tf.random.normal([1, noise_dim])  # outputs random values from normal dist. to a certain array shape

generated_image = generator(noise, training=False)  # interesting, doesn't need .fit .predict or anything



plt.imshow(generated_image[0, :, :, :]*255)#, cmap='gray')
def make_discriminator_model():

    model = tf.keras.Sequential()   # basic binary classification model

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',

                                     input_shape=[128, 128, 3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model



# create D

discriminator = make_discriminator_model()

print(discriminator.summary())
decision = discriminator(generated_image)

print(decision)
# This method returns a helper function to compute cross entropy loss (prob between 0 and 1)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    # ones_like creates array of ones with similar shape as the input array

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)  # but here they use the same Adam anyway

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
# Notice the use of `tf.function`

# This annotation causes the function to be "compiled".

@tf.function

def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, noise_dim])



    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=True)



        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)



    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def train(dataset, epochs):

    for epoch in range(epochs):

        start = time.time()



        for image_batch in dataset:

            train_step(image_batch)



        # Produce images for the GIF as we go

        display.clear_output(wait=True)

        generate_and_save_images(generator,

                                 epoch + 1,

                                 seed)



        # Save the model every 15 epochs

        if (epoch + 1) % 15 == 0:

            checkpoint.save(file_prefix = checkpoint_prefix)



        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



        # Generate after the final epoch

        display.clear_output(wait=True)

        generate_and_save_images(generator, epochs, seed)
def generate_and_save_images(model, epoch, test_input):

    # Notice `training` is set to False.

    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)  # same as num_examples_to_generate

    fig = plt.figure(figsize=(12,12))



    for i in range(predictions.shape[0]):

        plt.subplot(3, 3, i+1)

        plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5, cmap='gray')

        plt.axis('off')



    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
train(train_dataset, EPOCHS)
benign = train_df[train_df["target"] == 0]

malignant = train_df[train_df["target"] == 1]



def show_img(target, n=16):

    img_name = target.image_name.values

    ex_img = np.random.choice(img_name, n)  # grab n number of images

    plt.figure(figsize=(15,15))

    for i in range(n):

        plt.subplot(4, 4, i + 1)

        img = plt.imread(image_path + ex_img[i]+'.jpg')

        plt.imshow(img, cmap='gray')

        plt.axis('off')

    plt.tight_layout()
show_img(benign)
show_img(malignant)