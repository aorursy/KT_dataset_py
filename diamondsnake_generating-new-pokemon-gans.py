import tensorflow as tf

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 

from tensorflow.keras.layers import Flatten, BatchNormalization

from tensorflow.keras.layers import Activation, ZeroPadding2D

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.optimizers import Adam

import numpy as np

from PIL import Image

from tqdm import tqdm

import os 

import time

import matplotlib.pyplot as plt
# Nicely formatted time string

def hms_string(sec_elapsed):

    h = int(sec_elapsed / (60 * 60))

    m = int((sec_elapsed % (60 * 60)) / 60)

    s = sec_elapsed % 60

    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
# Generation resolution - Must be square 

# Training data is also scaled to this.

# Note GENERATE_RES 4 or higher  

# will blow Google CoLab's memory and have not

# been tested extensivly.

GENERATE_RES = 1 # Generation resolution factor 

# (1=32, 2=64, 3=96, 4=128, etc.)

GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)

IMAGE_CHANNELS = 3



# Preview image 

PREVIEW_ROWS = 4

PREVIEW_COLS = 7

PREVIEW_MARGIN = 16



# Size vector to generate images from

SEED_SIZE = 100



# Configuration

DATA_PATH = '/kaggle/input/pokemon-images-and-types/images'

EPOCHS = 10

BATCH_SIZE = 32

BUFFER_SIZE = 60000



print(f"Will generate {GENERATE_SQUARE}px square images.")
print("Loading training images...")

start = time.time()



training_data = []

faces_path = os.path.join(DATA_PATH,'images')

for filename in tqdm(os.listdir(faces_path)):

    path = os.path.join(faces_path,filename)

    image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS).convert('RGBA')

    image.load() # required for png.split()

    background = Image.new("RGBA", image.size, (255, 255, 255))

    image = Image.alpha_composite(background, image).convert('RGB')

    training_data.append(np.asarray(image))

training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))

training_data = training_data.astype(np.float32)

training_data = training_data / 127.5 - 1.



elapsed = time.time()-start

print (f'Image preprocess time: {hms_string(elapsed)}')
# example training image

image
# Batch and shuffle the data

train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def build_generator(seed_size, channels):

    model = Sequential()



    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))

    model.add(Reshape((4,4,256)))



    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))



    model.add(UpSampling2D())

    model.add(Conv2D(256,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

   

    # Output resolution, additional upsampling

    model.add(UpSampling2D())

    model.add(Conv2D(128,kernel_size=3,padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))



    if GENERATE_RES>1:

        model.add(UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))

        model.add(Conv2D(128,kernel_size=3,padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))



    # Final CNN layer

    model.add(Conv2D(channels,kernel_size=3,padding="same"))

    model.add(Activation("tanh"))



    return model





def build_discriminator(image_shape):

    model = Sequential()



    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

    model.add(ZeroPadding2D(padding=((0,1),(0,1))))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))



    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))



    return model
def save_images(noise):

    image_array = np.full(( 

      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 

      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 

      255, dtype=np.uint8)



    generated_images = generator.predict(noise)



    generated_images = 0.5 * generated_images + 0.5



    image_count = 0

    for row in range(PREVIEW_ROWS):

        for col in range(PREVIEW_COLS):

            r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN

            c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN

            image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255

            image_count += 1



    im = Image.fromarray(image_array)

    return im
# test first stage generatior

generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)

noise = tf.random.normal([1, SEED_SIZE])

generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])
# test first stage discriminator

image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)

decision = discriminator(generated_image)

print (decision)
# This method returns a helper function to compute cross entropy loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss



def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1.5e-6,0.5)

discriminator_optimizer = tf.keras.optimizers.Adam(5.0e-6,0.5)
# Notice the use of `tf.function`

# This annotation causes the function to be "compiled".

@tf.function

def train_step(images):

    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])



    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(seed, training=True)



        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=True)



        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(real_output, fake_output)





        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss,disc_loss
def train(dataset, epochs):

    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))

    start = time.time()



    for epoch in range(epochs):

        epoch_start = time.time()



        gen_loss_list = []

        disc_loss_list = []



        for image_batch in dataset:

            t = train_step(image_batch)

            gen_loss_list.append(t[0])

            disc_loss_list.append(t[1])



        g_loss = sum(gen_loss_list) / len(gen_loss_list)

        d_loss = sum(disc_loss_list) / len(disc_loss_list)



        epoch_elapsed = time.time()-epoch_start

        print (f'Epoch {epoch+1}: gen_loss={g_loss}, disc_loss={d_loss}, {hms_string(epoch_elapsed)}')

        if np.floor((epoch+1)/10)==(epoch+1)/10:

            sample = save_images(fixed_seed)



    elapsed = time.time()-start

    print (f'Training time: {hms_string(elapsed)}')

    return sample

# sample of generations

sample = train(train_dataset, EPOCHS)

sample