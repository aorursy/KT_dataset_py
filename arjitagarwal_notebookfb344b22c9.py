import tensorflow as tf

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 

from tensorflow.keras.layers import Flatten, BatchNormalization

from tensorflow.keras.layers import Activation, ZeroPadding2D

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import GaussianNoise

from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import initializers

from tensorflow.keras.constraints import MinMaxNorm

import numpy as np

from numpy import random



from PIL import Image

from tqdm import tqdm

import os 

import time

import matplotlib.pyplot as plt

from tensorflow.keras.mixed_precision import experimental as mixed_precision



#format time into something more readable

def hms_string(sec_elapsed):

    h = int(sec_elapsed / (60 * 60))

    m = int((sec_elapsed % (60 * 60)) / 60)

    s = sec_elapsed % 60

    return "{}:{:>02}:{:>05.2f}".format(h, m, s)



#policy = mixed_precision.Policy('mixed_float16')

#mixed_precision.set_policy(policy)
generate_res = (64,64)

PHOTO_PATH = "/kaggle/input/anime-faces/data/data"

DATA_PATH = "/kaggle/working/training_64_64.npy"#'/kaggle/input/animefacedataset/images'

MODEL_PATH = "/kaggle/working/"

SAVE_PATH = '/kaggle/working/'

SEED_SIZE = 128

BATCH_SIZE = 128

EPOCHS = 50 ## Increase this for better generated image 

img_width = 64

img_height = 64

channels = 3

n_critic = 5

learning_rate = 1e-4

beta1 = 0

beta2 = 0.9
training_data = []

for filename in tqdm(os.listdir(PHOTO_PATH)):

    path = os.path.join(PHOTO_PATH,filename)

    image = Image.open(path).resize((img_width,

            img_height),Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(training_data,(-1,img_width,

            img_height,channels))

training_data = training_data.astype(np.float32)

training_data = training_data / 127.5 - 1 #images should be normalised to [-1,1]

print(np.shape(training_data))

np.save("training_64_64.npy",training_data) #This gets saved in /kaggle/working/
def build_generator(seed):

    model = Sequential()

    model.add(Conv2DTranspose(512, input_shape = (1,1,seed), kernel_size=4, strides=1, padding='valid',use_bias=False))

    model.add(Activation("relu"))

    

    #model.add(Dropout(0.5))

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same',use_bias=False))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    

    #model.add(Dropout(0.5))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same',use_bias=False))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))



    #model.add(Dropout(0.5))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',use_bias=False))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))



    #model.add(Dropout(0.5))

    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same',use_bias=False))

    model.add(Activation("tanh", dtype = 'float32'))

    model.summary()

    return model

def build_discriminator(input_size):

    

    model = Sequential()



    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=input_size, padding="same"))

    #model.add(GaussianNoise(1))

    model.add(LeakyReLU(alpha=0.2))

    

    #model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same",use_bias=False))

    #model.add(GaussianNoise(1))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    

    #model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same",use_bias=False))

    #model.add(GaussianNoise(1))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    

    #model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same",use_bias=False))

    #model.add(GaussianNoise(1))

    #model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))



    #model.add(Dropout(0.25))

    model.add(Conv2D(1, kernel_size=4, strides=1, padding="valid",use_bias=False))

    model.add(Flatten())

    model.add(Activation('linear', dtype = 'float32'))

    model.summary()

    return model



training_data = np.load(DATA_PATH)

train_dataset = tf.data.Dataset.from_tensor_slices(training_data[:int(len(training_data)/BATCH_SIZE)*BATCH_SIZE]).shuffle(int(len(training_data)/BATCH_SIZE)*BATCH_SIZE).batch(BATCH_SIZE)

training_data = []
discriminator = build_discriminator((img_height,img_width,channels))#tf.keras.models.load_model("/kaggle/working/face_discriminator.h5")#

generator = build_generator(SEED_SIZE)#tf.keras.models.load_model("/kaggle/working/face_generator.h5")
def discriminator_loss(real_output,fake_output,penalty):

    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + penalty



def generator_loss(seed):

    fake_images = generator(seed,training=True)

    fake_output = discriminator(fake_images,training=True)

    return -tf.reduce_mean(fake_output)



generator_optimizer = Adam(learning_rate,beta1,beta2) 

#generator_optimizer = mixed_precision.LossScaleOptimizer(generator_optimizer, loss_scale='dynamic')



discriminator_optimizer = Adam(learning_rate,beta1,beta2)

#discriminator_optimizer = mixed_precision.LossScaleOptimizer(discriminator_optimizer, loss_scale='dynamic')
@tf.function

def train_disc(real_images,seed):

    with tf.GradientTape() as disc_tape:

        

        fake_images = generator(seed,training=True)

        real_output = discriminator(real_images,training=True)

        fake_output = discriminator(fake_images,training=True)

        g_penalty = gradient_penalty(real_images,fake_images)

        disc_loss = discriminator_loss(real_output,fake_output,g_penalty)

        #scaled_disc_loss = discriminator_optimizer.get_scaled_loss(disc_loss)

        

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)#(scaled_disc_loss, discriminator.trainable_variables)

        #gradients_of_discriminator = discriminator_optimizer.get_unscaled_gradients(gradients_of_discriminator)

    

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return tf.reduce_mean(real_output),g_penalty
def gradient_penalty(real_images,fake_images):

    t = np.random.uniform(size=[BATCH_SIZE, 1, 1, 1], low=0., high=1.).astype("float32")

    penalty_images = t* fake_images + (1-t)* real_images

    penalty_output = discriminator(penalty_images,training=True)

    penalty_grads = tf.gradients(penalty_output, [penalty_images])[0]

    slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(penalty_grads), axis=[1, 2, 3]))

    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)            

    gradient_penalty = gradient_penalty * 10

    

    return gradient_penalty

@tf.function

def train_gen(seed):

    with tf.GradientTape() as gen_tape:

        gen_loss = generator_loss(seed)

        

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)#(scaled_gen_loss, generator.trainable_variables)

        #gradients_of_generator = generator_optimizer.get_unscaled_gradients(gradients_of_generator) 

        

        #scaled_gen_loss = generator_optimizer.get_scaled_loss(gen_loss)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    

    return tf.reduce_mean(gen_loss)
def train(dataset, epochs):

  start = time.time()

  for epoch in range(epochs):

    epoch_start = time.time()



    gen_loss_list = []

    disc_loss_list = []

    penalty_list = []



    for i,image_batch in enumerate(dataset):

      seed = np.random.normal(size=[BATCH_SIZE,1,1, SEED_SIZE]).astype("float32")

      disc_loss, penalty = train_disc(image_batch,seed)

      disc_loss_list.append(disc_loss)

      penalty_list.append(penalty)



      

      if i % n_critic == 0:

        gen_loss = train_gen(seed)

        gen_loss_list.append(gen_loss)

       

      

    g_loss = sum(gen_loss_list) / len(gen_loss_list)

    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    penalty = sum(penalty_list) / len(penalty_list)

    

    if epoch%20 == 0:

        

        generator.save(os.path.join(SAVE_PATH,"face_generator.h5"))

        discriminator.save(os.path.join(SAVE_PATH,"face_discriminator.h5"))



    epoch_elapsed = time.time()-epoch_start

    print (f'Epoch {epoch+1}, fake output={g_loss},real output={d_loss}, penalty = {penalty}, {hms_string(epoch_elapsed)}')

        



  elapsed = time.time()-start

  print (f'Training time: {(elapsed)}')
train(train_dataset,EPOCHS)

generator.save(os.path.join(SAVE_PATH,"face_generator.h5"))

discriminator.save(os.path.join(SAVE_PATH,"face_discriminator.h5"))

print(SEED_SIZE)
noise = tf.random.normal([1,1,1, 128])

generator = tf.keras.models.load_model(os.path.join(SAVE_PATH,"face_generator.h5"))

generated_image = generator.predict(noise)

generated_image = (generated_image + 1) * 127.5

plt.imshow(np.squeeze(generated_image).astype('uint8'))