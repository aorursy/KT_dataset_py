!pip install -q tensorflow-gpu==2.0.0-beta1
import tensorflow as tf

tf.__version__
import xml.etree.ElementTree as ET # For parsing XML

from PIL import Image # to read image

import glob

from tqdm import tqdm_notebook

import urllib

import tarfile

from imageio import imread, imsave, mimsave

import shutil

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import os

from tensorflow.keras import layers

import time
# Code slightly modified from user: cdeotte | https://www.kaggle.com/cdeotte/supervised-generative-dog-net



ROOT = '../input/'

# list of all image file names in all-dogs

IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs')

# list of all the annotation directories, each directory is a dog breed

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))



# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# iterate through each directory in annotation

for breed in breeds:

    # iterate through each file in the directory

    for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

        try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

        except: continue           

        # Element Tree library allows for parsing xml and getting specific tag values    

        tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

        # take a look at the print out of an xml previously to get what is going on

        root = tree.getroot() # <annotation>

        objects = root.findall('object') # <object>

        for o in objects:

            bndbox = o.find('bndbox') # <bndbox>

            xmin = int(bndbox.find('xmin').text) # <xmin>

            ymin = int(bndbox.find('ymin').text) # <ymin>

            xmax = int(bndbox.find('xmax').text) # <xmax>

            ymax = int(bndbox.find('ymax').text) # <ymax>

            w = np.min((xmax - xmin, ymax - ymin))

            img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

            img2 = img2.resize((64,64), Image.ANTIALIAS)

            imagesIn[idxIn,:,:,:] = np.asarray(img2)

            namesIn.append(breed)

            idxIn += 1         
# Inspect what the previous code created

print("imagesIn is a {} with {} {} by {} rgb({}) images. Shape: {}".format(type(imagesIn), imagesIn.shape[0], imagesIn.shape[1], imagesIn.shape[2], imagesIn.shape[3], imagesIn.shape))
# normalize the pixel values

imagesIn = (imagesIn[:idxIn,:,:,:]-127.5)/127.5 # Normalize the images to [-1, 1]



# this is needed because the gradient functions from TF require float32 instead of float64

imagesIn = tf.cast(imagesIn, 'float32')
# Batch and shuffle the data

BUFFER_SIZE = 60000

BATCH_SIZE = 256



train_dataset = tf.data.Dataset.from_tensor_slices(imagesIn).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(100,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Reshape((8, 8, 512)))

    assert model.output_shape == (None, 8, 8, 512)

    

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1),  padding='same', use_bias=False))

    assert model.output_shape == (None, 8, 8, 256)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2),  padding='same', use_bias=False))

    assert model.output_shape == (None, 16, 16, 128)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),  padding='same', use_bias=False))

    assert model.output_shape == (None, 32, 32, 64)

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 64, 64, 3)

    #model.add(layers.Dense(3, activation='tanh', use_bias=False))

    print("GENERATOR")

    model.summary()

    return model
generator = make_generator_model()



noise = tf.random.normal([1, 100])



generated_image = generator(noise,training=False)



plt.imshow(generated_image[0, :, :, 0])
print(generated_image.shape)

print(noise.shape)
def make_discriminator_model():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (4, 4), 

                            strides=(2, 2), 

                            padding='same', 

                            input_shape=[64, 64, 3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(128, (4, 4), 

                            strides=(2, 2), 

                            padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Flatten())

    model.add(layers.Dense(1, activation='sigmoid'))

    

    print("DISCRIMINATOR")

    model.summary()

    

    return model

    
discriminator = make_discriminator_model()

decision = discriminator(generated_image)

print (decision)
# This method returns a helper funciton to compute cross entropy loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss
def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
EPOCHS = 200

noise_dim = 100

num_examples_to_generate = 16



# we will reuse this seed overtime (so it's easier)

# to visualize progress in the animated

seed = tf.random.normal([num_examples_to_generate, noise_dim])
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
def generate_and_save_images(model, epoch, test_input):

    # Notice `training` is set to False.

    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    

    fig = plt.figure(figsize=(8,8))

    for i in range(predictions.shape[0]):

        plt.subplot(4, 4, i + 1)

        plt.imshow((predictions[i, :, :, :] + 1.)/2.)

        plt.axis('off')

    plt.savefig('image_at_epoch_{}.png'.format(epoch))

    plt.show

    
def train(dataset, epochs):

    for epoch in range(epochs):

        start = time.time()

        

        for image_batch in dataset:

            train_step(image_batch)

    # Generate after the final epoch     

    generate_and_save_images(generator,epochs,seed)    

    
%%time

train(train_dataset, EPOCHS)
from keras.preprocessing.image import image
%%time

if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

i_batch_size = 50

n_images = 10000

for i_batch in tqdm_notebook(range(0, n_images, i_batch_size)):

    noise = np.random.uniform(-1.0, 1.0, [i_batch_size, noise_dim]).astype(np.float32)

    gen_images = generator(noise, training=False)

    gen_images = gen_images * 127.5 + 127.5

    for j in range(i_batch_size):

        img = image.array_to_img(gen_images[j])

        imsave(os.path.join('../output_images',f'sample_{i_batch + j + 1}.png'), img)

        if i_batch + j + 1 == n_images:

            break

print(len(os.listdir('../output_images')))
if os.path.exists('images.zip'):

    os.remove('images.zip')

shutil.make_archive('images', 'zip', '../output_images')
