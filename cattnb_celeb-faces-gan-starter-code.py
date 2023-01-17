import warnings # We'll use this to suppress warnings caused by TensorFlow

warnings.simplefilter(action='ignore', category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # generating plot



import tensorflow as tf # modeling/training

tf.enable_eager_execution() # Must execute this at the beginning of the code

                            # See https://www.tensorflow.org/guide/eager for details

import time # Used for epoch timing



import imageio # GIF generation

import glob # GIF generation

import PIL # GIF generation





import os

data_dir = '/kaggle/input/celeba-dataset/'

os.listdir(data_dir)
list_eval_partition = pd.read_csv(data_dir + 'list_eval_partition.csv')

names_df = list_eval_partition['image_id']

names_df.head()
img_names = names_df.sample(n=16).values

shapes = []

plt.figure(figsize=(10,10))

for i, name in enumerate(img_names):

    plt.subplot(4, 4, i + 1)

    img = plt.imread(data_dir + 'img_align_celeba/img_align_celeba/' + name)

    shapes.append(img.shape)

    plt.imshow(img)

    plt.title(name)

    plt.axis('off')

_=plt.suptitle('')
def load_and_preprocess_image(name):

    image = tf.io.read_file(data_dir + 'img_align_celeba/img_align_celeba/' + name)

    image = tf.image.decode_jpeg(image)

    image = tf.image.resize_images(image, (216, 176))

    image = (image - 127.5) / 127.5

    return image
BATCH_SIZE = 256

name_ds = tf.data.Dataset.from_tensor_slices(names_df.values)

image_ds = name_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

image_ds = image_ds.shuffle(buffer_size=2000).batch(BATCH_SIZE)



num_batches = int(np.ceil(len(names_df) / BATCH_SIZE))

print('There are {} batches'.format(num_batches))
from tensorflow.keras import layers

def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(12*11*128, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((12, 11, 128)))

    assert model.output_shape == (None, 12, 11, 128)



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(3, 4), padding='same', use_bias=False))

    assert model.output_shape == (None, 36, 44, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(3, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 108, 88, 32)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 216, 176, 3)



    return model
generator = make_generator_model()

noise_image = tf.random.normal([1,100,])

generated_image = generator(noise_image, training=False)

plt.imshow((generated_image[0]*127.5 +127.5) / 255.)

_=plt.axis('off')

generator.summary()
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[216, 176, 3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())

    model.add(layers.Dense(1))



    return model
discriminator = make_discriminator_model()

discriminator.summary()

print(discriminator(generated_image))
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):

    real_loss = loss_obj(tf.ones_like(real_output), real_output)

    fake_loss = loss_obj(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss



def generator_loss(fake_output):

    return loss_obj(tf.ones_like(fake_output), fake_output)
gen_optimizer = tf.keras.optimizers.Adam(1e-4)

disc_optimizer = tf.keras.optimizers.Adam(1e-4)
@tf.function

def train_step(images):

    noise = tf.random.normal([BATCH_SIZE, 100])

    

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        

        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=True)

        

        disc_loss = discriminator_loss(real_output, fake_output)

        gen_loss = generator_loss(fake_output)

        

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)

    

    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    

    return gen_loss, disc_loss
def show_and_generate_images(model, epoch, test_output):

    predictions = model(test_output, training=False)

    

    plt.figure(figsize=(10,10))

    for i in range(len(test_output)):

        plt.subplot(4,4,i+1)

        plt.imshow((predictions[i] * 127.5 +127.5) / 255.)

        plt.axis('off')

        

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    _=plt.show()
def train(dataset, epochs):

    print('Begining to train...')

    

    history = pd.DataFrame(['gen_loss', 'disc_loss'])

    for epoch in range(epochs):

        start = time.time()

        epoch_gen_loss = tf.keras.metrics.Mean()

        epoch_disc_loss = tf.keras.metrics.Mean()

        for i, images in enumerate(dataset):

            gen_loss, disc_loss = train_step(images)

            epoch_gen_loss.update_state(gen_loss)

            epoch_disc_loss.update_state(disc_loss)



        show_and_generate_images(generator, epoch + 1, seed)

        stats = 'Epoch {0} took {1} seconds. Gen_loss: {2:0.3f}, Disc_loss: {3:0.3f}'

        print(stats.format(epoch + 1, int(time.time() - start), 

                           epoch_gen_loss.result().numpy(), 

                           epoch_disc_loss.result().numpy()))

        history = history.append({'gen_loss': epoch_gen_loss.result().numpy(), 

                                  'disc_loss': epoch_disc_loss.result().numpy()}, 

                                  ignore_index=True)

        

    return history
EPOCHS = 32

seed = tf.random.normal([16, 100])

history = train(image_ds, EPOCHS)

history.index = history.index + 1
ax = plt.axes(xlabel='epoch', ylabel='loss')

history.plot(ax=ax, figsize=(10,7))

_=plt.title('Loss History')
anim_file = 'dcgan.gif'



with imageio.get_writer(anim_file, mode='I') as writer:

  filenames = glob.glob('image*.png')

  filenames = sorted(filenames)

  last = -1

  for i,filename in enumerate(filenames):

    frame = 2*(i**0.5)

    if round(frame) > round(last):

      last = frame

    else:

      continue

    image = imageio.imread(filename)

    writer.append_data(image)

  image = imageio.imread(filename)

  writer.append_data(image)

  IPython.display.Image(filename=anim_file)