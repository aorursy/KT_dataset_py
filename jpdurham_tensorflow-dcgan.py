from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt, zipfile

import numpy as np

import glob

import imageio

import xml

import xml.etree.ElementTree as ET

import time

import PIL

import tensorflow as tf

tf.enable_eager_execution()

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

from tensorflow.keras.preprocessing.image import NumpyArrayIterator, ImageDataGenerator

import IPython

from IPython import display



import os

ROOT = '../input/'

dirs = os.listdir(ROOT)

print(dirs)

IMAGES_PATH = ROOT + 'all-dogs/all-dogs/'

ANNOTATIONS_PATH = ROOT + 'annotation/Annotation/'



IMGS = os.listdir(IMAGES_PATH)

BREEDS = os.listdir(ANNOTATIONS_PATH)



BATCH_SIZE = 256
# Look at annotaions (xml) of the first image

ex = IMGS[0].split('.')[0] # format of filenames is `000000000_0000.jpg`

dom = xml.dom.minidom.parse(glob.glob(ANNOTATIONS_PATH + ex[0].split('_')[0] + '*/' + ex)[0])

pretty_xml_as_string = dom.toprettyxml()

print(pretty_xml_as_string)
BREED_COUND = len(BREEDS)

IMG_SIZE = 64

CHANNELS = 3



cropped_images = []

cropped_image_labels = []

for breed in BREEDS:

    breed_path = os.path.join(ANNOTATIONS_PATH, breed)

    for record in os.listdir(breed_path):

        tree = ET.parse(os.path.join(breed_path, record))

        try:

            orig_img_path = os.path.join(IMAGES_PATH, record + '.jpg')

            orig_img = PIL.Image.open(orig_img_path)

        except:

            print('Image {} was not found'.format(record))

            continue

        root = tree.getroot()

        for obj in root.findall('object'):

            name = obj.find('name').text

            pose = obj.find('pose').text

            bndbox = obj.find('bndbox')

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

            

            """

            Using the minimum distance will prevent image distoration

            since it keeps the image square

            """

            if False:

                min_dist = np.min((xmax - xmin, ymax - ymin))

                crp_img = orig_img.crop((xmin, ymin, xmin + min_dist, ymin + min_dist))

            crp_img = orig_img.crop((xmin, ymin, xmax, ymax))

            crp_img = crp_img.resize((IMG_SIZE, IMG_SIZE), PIL.Image.ANTIALIAS)

            cropped_images.append(np.asarray(crp_img))

            cropped_image_labels.append(name)

            

IMG_COUNT = len(cropped_images)
# Build training image generator

training_image_gen = ImageDataGenerator(

    horizontal_flip=True

)

training_examples = (np.asarray(cropped_images)-127.5)/127.5

training_labels = np.asarray(cropped_image_labels)

training_dataset = NumpyArrayIterator(training_examples,

                                      training_labels,

                                      shuffle=True,

                                      image_data_generator=training_image_gen,

                                      batch_size=BATCH_SIZE)
# Take a look at the images

plt.figure(figsize=(15,15))

i = 0

w = 4

h = 4

# for val in [training_dataset[i] for i in range(w*h)]:

#     print(val[0][0], val[1][0])

for image, label in [(val[0][0], val[1][0]) for val in [training_dataset[i] for i in range(w*h)]]:

    plt.subplot(h,w,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    image = (image*127.5+127.5).astype(int)

    plt.imshow(image)

    plt.xlabel(label, fontsize=12)

    i += 1

plt.show()
# For overriding default kernel initializer, glorot uniform

KERNEL_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.25)



# Generator function

def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Reshape((8, 8, 512)))

    assert model.output_shape == (None, 8, 8, 512) # Note: None here is the batch size

    

    model.add(layers.Conv2DTranspose(256, (5, 5), 

                                     strides=(2, 2),

                                     padding='same',

                                     use_bias=False,

                                     kernel_initializer=KERNEL_INIT))

    assert model.output_shape == (None, 16, 16, 256)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(128, (5, 5),

                                     strides=(2, 2),

                                     padding='same',

                                     use_bias=False,

                                     kernel_initializer=KERNEL_INIT))

    assert model.output_shape == (None, 32, 32, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(64, (5, 5),

                                     strides=(2, 2), 

                                     padding='same', 

                                     use_bias=False,

                                     kernel_initializer=KERNEL_INIT))

    assert model.output_shape == (None, 64, 64, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Dense(3,use_bias=False, activation='tanh'))

    assert model.output_shape == (None, IMG_SIZE, IMG_SIZE, CHANNELS)



    return model
# Create and test generator

generator = make_generator_model()



noise = tf.random.normal([1, 100])

generated_image = generator(noise, training=False)

rn_img = (generated_image[0,:,:,:].numpy()* 127.5 + 127.5).astype(int)

plt.imshow(generated_image[0,:,:,:])
# Discriminator function

def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (6, 6), strides=(2, 2), padding='same',

                                     input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.2))



    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.2))



    model.add(layers.Flatten())

    model.add(layers.Dense(1, activation='linear'))



    return model
discriminator = make_discriminator_model()

decision = discriminator(generated_image)

print (decision)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    print('         real loss: {}  fake loss: {}'.format(real_loss, fake_loss))

    total_loss = real_loss + fake_loss

    return total_loss



def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)



generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)



checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
EPOCHS = 150

STEPS_PER_EPOCH = IMG_COUNT / BATCH_SIZE

noise_dim = 100

GRID_H = 4

GRID_W = 4

num_examples_to_generate = GRID_H * GRID_W



# Reuse this seed over time to visualize progress

seed = tf.random.normal([num_examples_to_generate, noise_dim])
def train_step(images, step_of_epoch=None):

    if step_of_epoch is not None and isinstance(step_of_epoch, int):

        print('Training Step {}'.format(step_of_epoch))

        

    noise = tf.random.normal([BATCH_SIZE, noise_dim])



#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

    # overtrain generator

    for i in range(5):

        with tf.GradientTape() as gen_tape:

            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            generated_images = generator(noise, training=True)

            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        

#         generated_images = generator(noise, training=True)



    with tf.GradientTape() as disc_tape:

        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=False)



#         gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)

        

        print('gen loss: {}    disc loss: {}'.format(gen_loss, disc_loss))



#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def train(dataset, epochs):

    for epoch in range(epochs):

        print('Training epoch {}/{}'.format(epoch, epochs))

        start = time.time()

    

        i = 0

        for image_batch in dataset:

            train_step(image_batch)

            i += 1

            if i >= STEPS_PER_EPOCH:

                break



        # Produce images for a GIF as we go

        display.clear_output(wait=True)

        generate_and_save_images(generator,

                                 epoch + 1,

                                 seed)



        # Save the model every 25 epochs

        if (epoch + 1) % 25 == 0:

            checkpoint.save(file_prefix = checkpoint_prefix)



        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))



    # Generate after the final epoch

    display.clear_output(wait=True)

    generate_and_save_images(generator,

                             epochs,

                             seed)
def generate_and_save_images(model, epoch, test_input):

    # Notice `training` is set to False.

    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)



    fig = plt.figure(figsize=(10, 10))



    for i in range(predictions.shape[0]):

        plt.subplot(GRID_H, GRID_W, i+1)

        plt.imshow(((predictions[i, :, :, :]).numpy()*127.5+127.5).astype(int)) 

        plt.axis('off')



    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
train(training_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



def display_image(epoch_no):

  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



display_image(EPOCHS)
anim_file = 'dcgan-dogs.gif' 



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

    display.Image(filename=anim_file)
%%time

# Create submission file `images.zip`

z = zipfile.PyZipFile('images.zip', mode='w')



filename = 'generator_model.h5'

tf.keras.models.save_model(

    generator,

    filename,

    overwrite=True,

    include_optimizer=True,

    save_format=None

)



for k in range(10000):

    # training = False sets all layers to run in inference mode

    generated_image = generator(tf.random.normal([1, noise_dim]), training=False)

    f = str(k)+'.png'

    img = ((generated_image[0,:,:,:]).numpy()*127.5+127.5).astype(int)

    tf.keras.preprocessing.image.save_img(

        f,

        img

    )

    z.write(f); os.remove(f)

z.close()
!ls -al | grep .zip