# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2

from sklearn.model_selection import train_test_split

from keras import layers

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout

from keras.layers import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import Conv2DTranspose, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras.utils.generic_utils import Progbar

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# path to images

path = '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/'



# animal categories

categories = ['dogs', 'panda', 'cats']

num_classes = len(categories)
# let's display some of the pictures



for category in categories:

    fig, _ = plt.subplots(3,4)

    fig.suptitle(category)

    for k, v in enumerate(os.listdir(path+category)[:12]):

        img = plt.imread(path+category+'/'+v)

        plt.subplot(3, 4, k+1)

        plt.axis('off')

        plt.imshow(img)

    plt.show()
shape0 = []

shape1 = []



for category in categories:

    for files in os.listdir(path+category):

        shape0.append(plt.imread(path+category+'/'+ files).shape[0])

        shape1.append(plt.imread(path+category+'/'+ files).shape[1])

    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))

    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))

    shape0 = []

    shape1 = []
# initialize the data and labels

data = []

labels = []

imagePaths = []

HEIGHT = 32

WIDTH = 54

N_CHANNELS = 3



# grab the image paths and randomly shuffle them

for k, category in enumerate(categories):

    for f in os.listdir(path+category):

        imagePaths.append([path+category+'/'+f, k]) # k=0 for dogs, k=1 for pandas ans k=2 for cats



import random

random.shuffle(imagePaths)

print(imagePaths[:10])



# loop over the input images

for imagePath in imagePaths:

    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring

    # aspect ratio) and store the image in the data list

    image = cv2.imread(imagePath[0])

    image = cv2.resize(image, (WIDTH, HEIGHT))  

    data.append(image)

    # extract the class label from the image path and update the

    # labels list

    label = imagePath[1]

    labels.append(label)
# Let's check everything is ok

plt.subplots(3,4)

for i in range(12):

    plt.subplot(3,4, i+1)

    plt.imshow(data[i])

    plt.axis('off')

    plt.title(categories[labels[i]])

plt.show()
def build_generator(latent_size=100):

    # we will map a pair of (z, L), where z is a latent vector and L is a

    # label drawn from P_c, to image space (..., 54, 32, 3)

    cnn = Sequential()



    cnn.add(Dense(3*54*32, input_dim=latent_size, activation='relu'))

    cnn.add(Reshape((4, 3, 432)))



    # upsample to (8, 6, ...)

    cnn.add(Conv2DTranspose(192, 2, strides=2, padding='valid',

                        activation='relu',

                        kernel_initializer='glorot_normal'))

    cnn.add(BatchNormalization())



    # upsample to (16, 18, ...)

    cnn.add(Conv2DTranspose(96, 5, strides=(2,3), padding='same',

                        activation='relu',

                        kernel_initializer='glorot_normal'))

    cnn.add(BatchNormalization())



    # upsample to (32, 54, ...)

    cnn.add(Conv2DTranspose(3, 5, strides=(2,3), padding='same',

                        activation='tanh',

                        kernel_initializer='glorot_normal'))





    # this is the z space commonly referred to in GAN papers

    latent = Input(shape=(latent_size, ))



    # this will be our label

    image_class = Input(shape=(1,), dtype='int32')



    cls = Embedding(num_classes, latent_size,

                    embeddings_initializer='glorot_normal')(image_class)



    # hadamard product between z-space and a class conditional embedding

    h = layers.multiply([latent, cls])



    fake_image = cnn(h)



    return Model([latent, image_class], fake_image)





def build_discriminator():

    # build a relatively standard conv net, with LeakyReLUs as suggested in

    # the reference paper

    cnn = Sequential()



    cnn.add(Conv2D(32, 3, padding='same', strides=2,

                   input_shape=(32, 54, 3)))

    cnn.add(LeakyReLU(0.2))

    cnn.add(Dropout(0.3))



    cnn.add(Conv2D(64, 3, padding='same', strides=1))

    cnn.add(LeakyReLU(0.2))

    cnn.add(Dropout(0.3))



    cnn.add(Conv2D(128, 3, padding='same', strides=2))

    cnn.add(LeakyReLU(0.2))

    cnn.add(Dropout(0.3))



    cnn.add(Conv2D(256, 3, padding='same', strides=1))

    cnn.add(LeakyReLU(0.2))

    cnn.add(Dropout(0.3))



    cnn.add(Flatten())



    image = Input(shape=(32, 54, 3))



    features = cnn(image)



    # first output (name=generation) is whether or not the discriminator

    # thinks the image that is being shown is fake, and the second output

    # (name=auxiliary) is the class that the discriminator thinks the image

    # belongs to.

    fake = Dense(1, activation='sigmoid', name='generation')(features)

    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)



    return Model(image, [fake, aux])
epochs = 200

batch_size = 100

latent_size = 100



# Adam parameters suggested in https://arxiv.org/abs/1511.06434

adam_lr = 0.0002

adam_beta_1 = 0.5



# build the discriminator

print('Discriminator model:')

discriminator = build_discriminator()

discriminator.compile(optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

discriminator.summary()



# build the generator

generator = build_generator(latent_size)



latent = Input(shape=(latent_size, ))

image_class = Input(shape=(1,), dtype='int32')



# get a fake image

fake = generator([latent, image_class])



# we only want to be able to train generation for the combined model

discriminator.trainable = False

fake, aux = discriminator(fake)

combined = Model([latent, image_class], [fake, aux])



print('Combined model:')

combined.compile(optimizer=Adam(learning_rate=adam_lr, beta_1=adam_beta_1),loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

combined.summary()



x_train = (np.array(data).astype(np.float32)-127.5) / 127.5

y_train = np.array(labels)



#(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2)



#x_train = (x_train.astype(np.float32) - 127.5) / 127.5





num_train = x_train.shape[0]



print(x_train.shape, y_train.shape)



for epoch in range(1, epochs + 1):

    print('Epoch {}/{}'.format(epoch, epochs))



    num_batches = int(np.ceil(x_train.shape[0] / float(batch_size)))

    progress_bar = Progbar(target=num_batches)



    epoch_gen_loss = []

    epoch_disc_loss = []



    for index in range(num_batches):

        # get a batch of real images

        image_batch = x_train[index * batch_size:(index + 1) * batch_size]

        label_batch = y_train[index * batch_size:(index + 1) * batch_size]



        # generate a new batch of noise

        noise = np.random.uniform(-1, 1, (len(image_batch), latent_size)) # numpy array with shape (len(image_batch), latent_size)



        # sample some labels from p_c

        sampled_labels = np.random.randint(0, num_classes, len(image_batch)) # numpy array of int in [[0, num_classes-1]] with shape len(image_batch)



        # generate a batch of fake images, using the generated labels as a

        # conditioner. We reshape the sampled labels to be

        # (len(image_batch), 1) so that we can feed them into the embedding

        # layer as a length one sequence

        generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0) 

                    # sampled_labels.reshape((-1, 1)) -> sample_labels has shape (len(image_batch), 1)

                    # generated_images has shape (-1, 32, 55, 3)

        x = np.concatenate((image_batch, generated_images))



        # use one-sided soft real/fake labels

        # Salimans et al., 2016

        # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)

        soft_zero, soft_one = 0, 0.95

        y = np.array([soft_one] * len(image_batch) + [soft_zero] * len(image_batch))

        aux_y = np.concatenate((label_batch, sampled_labels), axis=0)



        # we don't want the discriminator to also maximize the classification

        # accuracy of the auxiliary classifier on generated images, so we

        # don't train discriminator to produce class labels for generated

        # images (see https://openreview.net/forum?id=rJXTf9Bxg).

        # To preserve sum of sample weights for the auxiliary classifier,

        # we assign sample weight of 2 to the real images.

        disc_sample_weight = [np.ones(2 * len(image_batch)), np.concatenate((np.ones(len(image_batch)) * 2, np.zeros(len(image_batch))))]



        # see if the discriminator can figure itself out...

        epoch_disc_loss.append(discriminator.train_on_batch(x, [y, aux_y], sample_weight=disc_sample_weight))



        # make new noise. we generate 2 * batch size here such that we have

        # the generator optimize over an identical number of images as the

        # discriminator

        noise = np.random.uniform(-1, 1, (2 * len(image_batch), latent_size))

        sampled_labels = np.random.randint(0, num_classes, 2 * len(image_batch))



        # we want to train the generator to trick the discriminator

        # For the generator, we want all the {fake, not-fake} labels to say

        # not-fake

        trick = np.ones(2 * len(image_batch)) * soft_one



        epoch_gen_loss.append(combined.train_on_batch([noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        

        progress_bar.update(index + 1)

    

    # generate some digits to display pictures every 20 epochs and the last one

    if epoch%20 == 1 or epoch == epochs:

        num_cols = 4 # num_cols : number of pictures per classes

        noise = np.tile(np.random.uniform(-1, 1, (num_cols, latent_size)), (num_classes, 1)) 

                            # generates num_classes arrays of shape (num_rows, latent_size)

                            # along the horizontal axis and 1 row along yhe other axis.

        sampled_labels = np.array([[i] * num_cols for i in range(num_classes)]).reshape(-1, 1)

        

        # get a batch to display

        generated_images = generator.predict([noise, sampled_labels], verbose=0)

        generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)

        

        # display images

        fig, _ = plt.subplots(num_classes, num_cols)

        for k in range(num_classes * num_cols):

            plt.subplot(num_classes, num_cols, k+1)

            plt.axis('off')

            plt.imshow(generated_images[k])

            plt.title(categories[sampled_labels[k][0]])

        plt.show()