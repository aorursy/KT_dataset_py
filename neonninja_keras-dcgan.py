import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import describe

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import img_to_array, load_img, save_img, ImageDataGenerator

import xml.etree.ElementTree as ET # for parsing XML

import glob

from tqdm import tqdm_notebook as tqdm

import os

SHAPE = (64, 64, 3)

BATCH_SIZE = 128
%%time

xmls = glob.glob("../input/annotation/Annotation/*/*")

train_data = []

for x in tqdm(xmls):

    tree = ET.parse(x)

    root = tree.getroot()

    objects = root.findall('object')

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        image_filename = "../input/all-dogs/all-dogs/" + tree.find("filename").text + ".jpg"

        try:

            image = load_img(image_filename)

            image = img_to_array(image.crop((xmin,ymin,xmax,ymax)).resize(SHAPE[:2])) / 255

            train_data.append(image)

        except:

            continue

print(len(train_data))

train_data = np.stack(train_data)
fig = plt.figure(figsize=(25, 16))

for i, im in enumerate(train_data[:10]):

    ax = fig.add_subplot(5, 10, i + 1)

    plt.imshow(im)
train_data = train_data * 2 - 1
class DCGAN():

    def __init__(self, image_shape, generator_input_dim, img_channels, debug=False):

        optimizer = Adam(0.0002, 0.5)

        

        self.img_shape = image_shape

        self.generator_input_dim = generator_input_dim

        self.channels = img_channels

        self.history = []

        self.debug = debug



        # Build models

        self._build_generator_model()

        self._build_and_compile_discriminator_model(optimizer)

        self._build_and_compile_gan(optimizer)



    def train(self, epochs, train_data, batch_size):

        

        real = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            #  Train Discriminator

            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)

            batch = train_data[batch_indexes]

            generated = self._predict_noise(batch_size)

            loss_real = self.discriminator_model.train_on_batch(batch, real)

            loss_fake = self.discriminator_model.train_on_batch(generated, fake)

            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)



            #  Train Generator

            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))

            generator_loss = self.gan.train_on_batch(noise, real)

            

            if epoch > 0 and epoch % 100 == 0 and self.debug:

                print ("---------------------------------------------------------")

                print ("******************Epoch {}***************************".format(epoch))

                print ("Discriminator loss: {}".format(discriminator_loss[0]))

                print ("Generator loss: {}".format(generator_loss))

                print ("---------------------------------------------------------")

                sample = self.gen(1)[0]

                plt.imshow(sample)

                plt.show()

                self.plot_loss()

            

            self.history.append({"D":discriminator_loss[0],"G":generator_loss})

    

    def _build_generator_model(self):

        generator_input = Input(shape=(self.generator_input_dim,))

        generator_sequence = Sequential(

                [Dense(128 * 16 * 16, activation="relu", input_dim=self.generator_input_dim),

                 Reshape((16, 16, 128)),

                 UpSampling2D(),

                 Conv2D(128, kernel_size=3, padding="same"),

                 BatchNormalization(momentum=0.8),

                 LeakyReLU(alpha=0.2),

                 UpSampling2D(),

                 Conv2D(64, kernel_size=3, padding="same"),

                 BatchNormalization(momentum=0.8),

                 LeakyReLU(alpha=0.2),

                 Conv2D(self.channels, kernel_size=3, padding="same"),

                 Activation("tanh")])

    

        generator_output_tensor = generator_sequence(generator_input)

        self.generator_model = Model(generator_input, generator_output_tensor)

        

    def _build_and_compile_discriminator_model(self, optimizer):

        discriminator_input = Input(shape=self.img_shape)

        discriminator_sequence = Sequential(

                [Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),

                 LeakyReLU(alpha=0.2),

                 Dropout(0.25),

                 Conv2D(64, kernel_size=3, strides=2, padding="same"),

                 ZeroPadding2D(padding=((0,1),(0,1))),

                 BatchNormalization(momentum=0.8),

                 LeakyReLU(alpha=0.2),

                 Dropout(0.25),

                 Conv2D(128, kernel_size=3, strides=2, padding="same"),

                 BatchNormalization(momentum=0.8),

                 LeakyReLU(alpha=0.2),

                 Dropout(0.25),

                 Conv2D(256, kernel_size=3, strides=2, padding="same"),

                 BatchNormalization(momentum=0.8),

                 LeakyReLU(alpha=0.2),

                 Dropout(0.25),

                 Flatten(),

                 Dense(1, activation='sigmoid')])

    

        discriminator_tensor = discriminator_sequence(discriminator_input)

        self.discriminator_model = Model(discriminator_input, discriminator_tensor)

        self.discriminator_model.compile(loss='binary_crossentropy',

            optimizer="Adam",

            metrics=['accuracy'])

        self.discriminator_model.trainable = False

    

    def _build_and_compile_gan(self, optimizer):

        real_input = Input(shape=(self.generator_input_dim,))

        generator_output = self.generator_model(real_input)

        discriminator_output = self.discriminator_model(generator_output)        

        

        self.gan = Model(real_input, discriminator_output)

        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    

    def _predict_noise(self, size):

        noise = np.random.normal(0, 1, (size, self.generator_input_dim))

        return self.generator_model.predict(noise)

        

    def gen(self, size):

        generated = self._predict_noise(size)

        generated = 0.5 * generated + 0.5

        return generated



    def plot_loss(self):

        hist = pd.DataFrame(self.history)

        plt.figure(figsize=(20,5))

        for colnm in hist.columns:

            plt.plot(hist[colnm],label=colnm)

        plt.legend()

        plt.ylabel("loss")

        plt.xlabel("epochs")

        plt.show()



gan = DCGAN(SHAPE, 100, SHAPE[2], debug=0)

gan.train(epochs = 20000, train_data = train_data, batch_size=BATCH_SIZE)
gan.plot_loss()
%%time

gens = gan.gen(10000)
fig = plt.figure(figsize=(25, 16))

for i in range(20):

    ax = fig.add_subplot(10, 10, i + 1)

    plt.imshow(gens[i])
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')



for i in tqdm(range(len(gens))):

    save_img("../output_images/{}.png".format(i), gens[i])
import shutil

shutil.make_archive('images', 'zip', '../output_images')