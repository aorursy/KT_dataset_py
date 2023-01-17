from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import os, math, cv2, glob, random, time

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

from IPython import display

from keras.optimizers import SGD, Adam

from keras.callbacks import LearningRateScheduler

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate, BatchNormalization, Conv2DTranspose, LeakyReLU, Dropout





train_dataset_path = '../input/10-monkey-species/training/training'

validation_dataset_path  = '../input/10-monkey-species/validation/validation'



# width 550 height 367

IMAGE_HEIGHT      = 64

IMAGE_WIDTH       = 64

IMAGE_SIZE        = (IMAGE_HEIGHT, IMAGE_WIDTH)



CATEGORIES        = os.listdir(train_dataset_path)

# len all images 1370

ALL_DATA_LENGTH   = 1370

INPUT_DATA_LENGTH = 100

INPUT_DENSE       = IMAGE_HEIGHT * IMAGE_WIDTH * 3

BATCH_SIZE        = 32 

tf.__version__
x = np.random.choice(np.arange(INPUT_DATA_LENGTH), INPUT_DATA_LENGTH)



def get_image_to_path():

    idxIn = 0; namesIn = [];

    imagesIn = np.zeros((ALL_DATA_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    for path_im in [train_dataset_path, validation_dataset_path]:

        for category in CATEGORIES:

            path = f'{path_im}/{category}/'

            class_id = CATEGORIES.index(category)

            for image in os.listdir(path):

                img = Image.open(os.path.join(path, image))

                w = img.size[0]

                h = img.size[1]

                sz = np.min((w,h))

                a=0; b=0

                if w<h: b = (h-sz)//2

                else: a = (w-sz)//2

                img = img.crop((0+a, 0+b, sz+a, sz+b))  

                img = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.ANTIALIAS)   

                imagesIn[idxIn,:,:,:] = np.asarray(img)

                namesIn.append(os.path.join(path, image))

                idxIn += 1

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    namesIn = np.array(namesIn)[idx]

    return imagesIn, namesIn

    

data_images, data_names = get_image_to_path()  
# BUILD DISCRIMINATIVE NETWORK

monkey = Input((INPUT_DENSE,))

monkeyName = Input((INPUT_DATA_LENGTH,))

x = Dense(INPUT_DENSE, activation='sigmoid')(monkeyName) 

x = Reshape((2,INPUT_DENSE,1))(concatenate([monkey,x]))

x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)

discriminated = Flatten()(x)



# COMPILE

discriminator = Model([monkey,monkeyName], discriminated)

discriminator.get_layer('conv').trainable = False

discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')



# DISPLAY ARCHITECTURE

discriminator.summary()
# TRAINING DATA

train_y = (data_images[:INPUT_DATA_LENGTH,:,:,:]/255.).reshape((-1,INPUT_DENSE))

train_X = np.zeros((INPUT_DATA_LENGTH,INPUT_DATA_LENGTH))

for i in range(INPUT_DATA_LENGTH): train_X[i,i] = 1

zeros = np.zeros((INPUT_DATA_LENGTH,INPUT_DENSE))



# TRAIN NETWORK

lr = 0.5

for k in range(5):

    annealer = LearningRateScheduler(lambda x: lr)

    h = discriminator.fit([zeros,train_X],

                          train_y,

                          epochs=5,

                          batch_size=BATCH_SIZE,

                          callbacks=[annealer],

                          verbose=0)

    print('Epoch',(k+1)*10,'/30 - loss =',h.history['loss'][-1] )

    if h.history['loss'][-1]<0.533: lr = 0.1
del train_X, train_y, data_images
print('Discriminator Recalls from Memory Monkey')    

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        xx = np.zeros((INPUT_DATA_LENGTH))

        xx[np.random.randint(INPUT_DATA_LENGTH)] = 1

        plt.subplot(1,5,j+1)

        img = discriminator.predict([zeros[0,:].reshape((-1,INPUT_DENSE)),xx.reshape((-1,INPUT_DATA_LENGTH))]).reshape((-1,64,64,3))

        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))

        plt.axis('off')

        plt.imshow(img)

    plt.show()
# BUILD GENERATOR NETWORK

BadMemory = True



if BadMemory:

    seed = Input((INPUT_DATA_LENGTH,))

    x = Dense(int(INPUT_DENSE/6))(seed)

    x = LeakyReLU()(x)



    x = Reshape((8,8,32))(x)



    x = Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = Conv2DTranspose(64, (2, 2), padding='same', strides=(2, 2))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = Conv2DTranspose(32, (2, 2), padding='same', strides=(2, 2))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = Conv2DTranspose(3, (1, 2), padding='same')(x)

    x = LeakyReLU()(x)

    generated = Flatten()(x)

else:

    seed = Input((INPUT_DATA_LENGTH,))

    x = Dense(INPUT_DENSE)(seed)

    generated = LeakyReLU()(x)



# COMPILE

generator = Model(seed, [generated,Reshape((INPUT_DATA_LENGTH,))(seed)])



# DISPLAY ARCHITECTURE

generator.summary()
# BUILD GENERATIVE ADVERSARIAL NETWORK

discriminator.trainable=False    

gan_input = Input(shape=(INPUT_DATA_LENGTH,))

x = generator(gan_input)

gan_output = discriminator(x)



# COMPILE GAN

gan = Model(gan_input, gan_output)

gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])

gan.compile(optimizer=Adam(2), loss='mean_squared_error')



# DISPLAY ARCHITECTURE 

gan.summary()
# TRAINING DATA

train = np.zeros((INPUT_DATA_LENGTH,INPUT_DATA_LENGTH))

for i in range(INPUT_DATA_LENGTH): train[i,i] = 1

zeros = np.zeros((INPUT_DATA_LENGTH,INPUT_DENSE))



# TRAIN NETWORKS

ep = 1; it = 15

if BadMemory: lr = 0.01

else: lr = 5.

    

for k in range(it):  



    # BEGIN DISCRIMINATOR COACHES GENERATOR

    annealer = LearningRateScheduler(lambda x: lr)

    h = gan.fit(train, zeros, epochs = ep, batch_size=BATCH_SIZE, callbacks=[annealer], verbose=0)



    # DISPLAY GENERATOR LEARNING PROGRESS 

    print('Epoch',(k+1),'/'+str(it)+' - loss =',h.history['loss'][-1] )

    plt.figure(figsize=(15,3))

    for j in range(5):

        xx = np.zeros((INPUT_DATA_LENGTH))

        xx[np.random.randint(INPUT_DATA_LENGTH)] = 1

        plt.subplot(1,5,j+1)

        img = generator.predict(xx.reshape((-1,INPUT_DATA_LENGTH)))[0].reshape((-1,64,64,3))

        img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))

        plt.axis('off')

        plt.imshow(img)

    plt.show()  

            

    # ADJUST LEARNING RATES

    if BadMemory:

        ep *= 2

        if ep>=32: lr = 0.001

        if ep>256: ep = 256

    else:

        if h.history['loss'][-1] < 25: lr = 1.

        if h.history['loss'][-1] < 1.5: lr = 0.5
class MonkeyGenerator:

    index = 0   

    def getMonkey(self,seed):

        xx = np.zeros((INPUT_DATA_LENGTH))

        xx[self.index] = 0.70

        xx[np.random.randint(INPUT_DATA_LENGTH)] = 0.30

        img = generator.predict(xx.reshape((-1,INPUT_DATA_LENGTH)))[0].reshape((64,64,3))

        self.index = (self.index+1)%1000

        return Image.fromarray( img.astype('uint8') )

    

# DISPLAY EXAMPLE DOGS

d = MonkeyGenerator()

for k in range(3):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = d.getMonkey(np.random.normal(0,1,100))

        plt.axis('off')

        plt.imshow(img)

    plt.show() 