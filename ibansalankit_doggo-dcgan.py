# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
from __future__ import print_function, division



from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Sequential, Model

from keras.optimizers import Adam



import matplotlib.pyplot as plt



import sys



import numpy as np



from tqdm import tqdm_notebook as tqdm
import xml.etree.ElementTree as ET # xml parser used during pre-processing stage

import xml.dom.minidom # for printing the annotation xml nicely



# Demo: Show how the Annotation files XML is structured

dom = xml.dom.minidom.parse('../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98') # or xml.dom.minidom.parseString(xml_string)

pretty_xml_as_string = dom.toprettyxml()

#print(pretty_xml_as_string)
from PIL import Image # Python Image Library

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
img_rows = 64

img_cols = 64

channels = 3

img_shape = (img_rows, img_cols, channels)

latent_dim = 100
def build_generator():



    model = Sequential()

    model.add(Dense(32 * 8 * 8, activation = "relu", input_dim = latent_dim))

    

    model.add(Dense(64 * 8 * 8, activation = "relu"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Reshape((8, 8, 64)))

    

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size = 3, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))

    

    model.add(UpSampling2D())

    model.add(Conv2D(256, kernel_size = 3, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))



    model.add(UpSampling2D())

    model.add(Conv2D(512, kernel_size = 3, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))

    

    model.add(Conv2D(256, kernel_size = 5, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))

    

    model.add(Conv2D(128, kernel_size = 5, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))

    

    model.add(Conv2D(64, kernel_size = 3, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(Activation("relu"))

    

    model.add(Conv2D(channels, kernel_size = 3, padding = "same"))

    model.add(Activation("tanh"))

    

    return model



generator = build_generator()

generator.summary()
def build_discriminator():



    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = img_shape, padding = "same"))

    model.add(LeakyReLU(alpha = 0.2))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same"))

    model.add(ZeroPadding2D(padding = ((0, 1), (0, 1))))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(LeakyReLU(alpha = 0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(LeakyReLU(alpha = 0.2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(256, kernel_size = 3, strides = 2, padding = "same"))

    model.add(BatchNormalization(momentum = 0.8))

    model.add(LeakyReLU(alpha = 0.2))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(1, activation = 'sigmoid'))



    return model



discriminator = build_discriminator()

discriminator.summary()
optimizer = Adam(0.0002, 0.5)



# Build and compile the discriminator

discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy',

                      optimizer=optimizer,

                      metrics=['accuracy'])



# Build the generator

generator = build_generator()



# The generator takes noise as input and generates imgs

z = Input(shape=(latent_dim,))

img = generator(z)



# For the combined model we will only train the generator

discriminator.trainable = False



# The discriminator takes generated images as input and determines validity

valid = discriminator(img)



# The combined model  (stacked generator and discriminator)

# Trains the generator to fool the discriminator

combined = Model(z, valid)

combined.compile(loss='binary_crossentropy', optimizer=optimizer)
def save_imgs(epoch):

    r = 1

    c = 5

    noise = np.random.normal(0.0, 1.0, (batch_size, latent_dim))

    gen_imgs = generator.predict(noise)

    

    # Rescale images 0 - 1

    gen_imgs = 0.5 * gen_imgs + 0.5

    

    cnt = 0

    for k in range(r):

        plt.figure(figsize=(15,3))

        for j in range(c):

            plt.subplot(1,5,j+1)

            plt.axis('off')

            plt.imshow(gen_imgs[cnt, :,:,:])

            cnt +=1

        #plt.savefig('gen_images_{}.png'.format(epoch))

        plt.show()

        plt.close()
epochs = 100000

batch_size=32

save_interval=1000

ComputeLB = True



# Load the dataset

X_train = imagesIn



# Rescale -1 to 1

X_train = X_train / 127.5 - 1.



# Adversarial ground truths

valid = np.ones((batch_size, 1))

fake = np.zeros((batch_size, 1))



g_loss_list = []

d_loss_list = []



for epoch in tqdm(range(epochs)):

    

    # ---------------------

    #  Train Discriminator

    # ---------------------



    # Select a random half of images

    idx = np.random.randint(0, X_train.shape[0], batch_size)

    imgs = X_train[idx]



    # Sample noise and generate a batch of new images

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_imgs = generator.predict(noise)



    # Train the discriminator (real classified as ones and generated as zeros)

    d_loss_real = discriminator.train_on_batch(imgs, valid)

    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



    # ---------------------

    #  Train Generator

    # ---------------------



    # Train the generator (wants discriminator to mistake images as real)

    g_loss = combined.train_on_batch(noise, valid)



    # Plot the progress

    g_loss_list.append(g_loss)

    d_loss_list.append(d_loss[0])

    # If at save interval => save generated image samples

    if epoch % save_interval == 0:

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        save_imgs(epoch)
save_imgs(epochs)
def plot_loss (gen_loss_list, dis_loss_list):

    plt.figure(figsize=(10,5))

    plt.title("Generator and Discriminator Loss")

    plt.plot(gen_loss_list,label="G")

    plt.plot(dis_loss_list,label="D")

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.show()

    

plot_loss(g_loss_list,d_loss_list)
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

im_batch_size = 50

n_images=10000

for i_batch in range(0, n_images, im_batch_size):

    noise= np.random.normal(0,1, [im_batch_size, 100])

    gen_images = generator.predict(noise)

    images = gen_images[:im_batch_size,:,:,:]

    for i_image in range(len(images)):

        img = gen_images[i_image,:,:,:]*127.5+127.5

        img = Image.fromarray(img.astype('uint8'))

        img.save(os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))



import shutil

shutil.make_archive('images', 'zip', '../output_images')