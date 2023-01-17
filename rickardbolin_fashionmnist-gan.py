import numpy as np 

import pandas as pd 

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model, load_model

from keras.optimizers import Adam

import numpy as np

from PIL import Image

from tqdm import tqdm

import tensorflow as tf

import os

import cv2



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical



# Import data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()



# Normalize

x_train = x_train/255

x_test = x_test/255



# Reshape, keras needs the third dimension to work.

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) 



# Convert y_train to integers from 0 to the number of categories (in this case 9)

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# RUN THIS CELL IF YOU ONLY WANT A CERTAIN CLASS GENERATED



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()



# Normalize

x_train = x_train/255

x_test = x_test/255

# Reshape

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) 



x_train_class = np.zeros(shape=(10,6000,28,28,1))

for classnr in range(10):

    x_train_class[classnr] = np.zeros((6000,28,28,1))



    counter = 0

    for i in range(60000):

        if(y_train[i] == classnr):

            x_train_class[classnr, counter,:,:,:] = x_train[i]

            counter += 1



    #for i in range(10000):

    #    if(y_test[i] == 6):

    #        x_train_class6[counter] = x_test[i]

    #        counter += 1



    #x_train = x_train_class[6]

    #x_train.shape



GENERATE_RES = 1 # (1=32, 2=64, 3=96, etc.)

GENERATE_SQUARE = 28 * GENERATE_RES

IMAGE_CHANNELS = 1



# Preview image 

PREVIEW_ROWS = 4

PREVIEW_COLS = 7

PREVIEW_MARGIN = 16

SAVE_FREQ = 300



# Size vector to generate images from

SEED_SIZE = 100



# Configuration

EPOCHS = 25000

BATCH_SIZE = 32
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

   

    for i in range(GENERATE_RES):

        model.add(UpSampling2D())

        model.add(Conv2D(128,kernel_size=3,padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))



    # Final CNN layer

    model.add(Conv2D(16, (3, 3)))

    model.add(Conv2D(channels,kernel_size=3))#,padding="same"))



    model.add(Activation("tanh"))



    input = Input(shape=(seed_size,))

    generated_image = model(input)



    return Model(input,generated_image)





def build_discriminator(image_shape):

    model = Sequential()



    model.add(Conv2D(28, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))

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



    input_image = Input(shape=image_shape)



    validity = model(input_image)



    return Model(input_image, validity)

  

def save_images(cnt,noise):

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



          

    output_path = os.path.join("/kaggle/working/",'output')

    if not os.path.exists(output_path):

        os.makedirs(output_path)



    filename = os.path.join(output_path,f"train-{cnt}.png")

    im = Image.fromarray(image_array)

    im.save(filename)
def setUpAndCompile():

    image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

    discOptimizer = Adam(0.00015,0.5) # (ORIGINAL = 1.5e-4, 0.5  !)learning rate and momentum adjusted from paper

    combOptimizer = Adam(0.0001,0.5)



    discriminator = build_discriminator(image_shape)

    discriminator.compile(loss="binary_crossentropy",optimizer=discOptimizer,metrics=["accuracy"])

    generator = build_generator(SEED_SIZE,IMAGE_CHANNELS)



    random_input = Input(shape=(SEED_SIZE,))



    generated_image = generator(random_input)



    discriminator.trainable = False



    validity = discriminator(generated_image)



    combined = Model(random_input,validity)

    combined.compile(loss="binary_crossentropy",optimizer=combOptimizer,metrics=["accuracy"])

    return generator, combined, discriminator
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



y_real = np.ones((BATCH_SIZE,1))# - 0.1

y_fake = np.zeros((BATCH_SIZE,1))



#classes = 1

generator, combined, discriminator = setUpAndCompile()

#x_train = x_train_class[classes]



fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))



#generator.load_weights(os.path.join("/kaggle/working/","fashionMnist_class0_generator.h5"))

cnt = 1

for epoch in range(EPOCHS):

    idx = np.random.randint(0,x_train.shape[0],BATCH_SIZE)

    x_real = x_train[idx]



    # Generate some images

    seed = np.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))

    x_fake = generator.predict(seed)



    # Train discriminator on real and fake

    discriminator_metric_real = discriminator.train_on_batch(x_real,y_real)

    discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)



    # Train generator on Calculate losses

    generator_metric = combined.train_on_batch(seed,y_real)



    # Time for an update?

    if epoch % SAVE_FREQ == 0:

        plt.clf()

        save_images(cnt, fixed_seed)

        img=mpimg.imread("/kaggle/working/output/train-" + str(cnt) + ".png")

        #imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # <---- ONLY FOR RGB

        plt.imshow(img, cmap='gray')

        plt.show()

        cnt += 1

        generator.save("/kaggle/working/fashionMnist_class" + str(classes) + "_generator.h5")

        print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")

generator.save("/kaggle/working/fashionMnist_class" + str(classes) + "_generator.h5")


