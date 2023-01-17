# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import *

import numpy as np

from matplotlib import pyplot as plt

#from tensorflow.keras.datasets import mnist,cifar10

import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

from sklearn.datasets import olivetti_faces
data=olivetti_faces.fetch_olivetti_faces()
faceImages=data['images']

faceImages=faceImages.reshape((faceImages.shape[0],faceImages.shape[1],faceImages.shape[2],1))

faceImages=faceImages/255.

faceImages.shape
plt.imshow(faceImages[np.random.randint(len(faceImages)-1)].reshape(64,64),cmap='gray')
adam=Adam(learning_rate=0.0002,beta_1=0.5)
generator = Sequential()

generator.add(Dense(512*4*4, input_dim=100))

generator.add(Reshape((4,4,512)))

generator.add(Conv2DTranspose(512,2,strides=2,padding='same',activation='tanh'))

#generator.add(LeakyReLU(0.2))

generator.add(Conv2DTranspose(256,2,strides=2,padding='same',activation='tanh'))

generator.add(LeakyReLU(0.2))

generator.add(Conv2DTranspose(128,2,strides=2,padding='same',activation='tanh'))

generator.add(LeakyReLU(0.2))

generator.add(Conv2DTranspose(1,2,strides=2,padding='same',activation='sigmoid'))

generator.compile(loss='binary_crossentropy', optimizer=adam)

generator.summary()
discriminator = Sequential()

discriminator.add(Convolution2D(32,5, input_shape=(64,64,1)))

discriminator.add(LeakyReLU(0.2))

discriminator.add(MaxPool2D(2))

discriminator.add(Convolution2D(64,3))

discriminator.add(LeakyReLU(0.2))

discriminator.add(MaxPool2D(2))

discriminator.add(BatchNormalization())

discriminator.add(Convolution2D(128,3))

discriminator.add(LeakyReLU(0.2))

discriminator.add(MaxPool2D(2))

discriminator.add(BatchNormalization())

discriminator.add(Flatten())

discriminator.add(Dense(1024))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(rate=0.20))

discriminator.add(BatchNormalization())

discriminator.add(Dense(512))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(rate=0.20))

discriminator.add(BatchNormalization())

discriminator.add(Dense(1,activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator.summary()
discriminator.trainable=False

inLayer=Input(shape=(100,))

hidLayer=generator(inLayer)

outLayer=discriminator(hidLayer)

gan=Model(inputs=inLayer,outputs=outLayer)

gan.compile(loss='binary_crossentropy',optimizer=adam,metrics=['acc'])

gan.summary()
def generateRandomImage():

    noise=np.random.normal(0,1,[1,100])

    fakeImg=generator.predict(noise)

    return fakeImg
batch_size=1

noise=np.random.normal(-1,1,[batch_size,100])

yGen=np.ones(batch_size)

#gan.train_on_batch(noise,yGen)

img=generator.predict(noise)

plt.imshow(img.reshape(64,64))
def train(epochs=10,batch_size=10):

    imgPerBatch=faceImages.shape[0]//batch_size

    dloss=0.0

    for e in range(1,epochs+1):

        print("Running epoch:",e)

        for i in range(imgPerBatch):

            noise=np.random.normal(0,1,[batch_size,100])

            fakeImages=generator.predict(noise)

            realImages=faceImages[np.random.randint(faceImages.shape[0],size=batch_size)]

            Xd=np.concatenate([realImages,fakeImages])



            yDis=np.zeros(batch_size*2)

            yDis[:batch_size]=0.9

            discriminator.trainable=True

            dloss=discriminator.train_on_batch(Xd,yDis)



            noise=np.random.normal(0,1,[batch_size,100])

            yGen=np.ones(batch_size)

            discriminator.trainable=False

            gloss=gan.train_on_batch(noise,yGen)

        print("\tdLoss after epoch:",e,'->',dloss)
def plotGeneratedImages(examples=100, dim=(10, 10), figsize=(20, 20)):

    noise = np.random.normal(0, 1, size=[examples, 100])

    generatedImages = generator.predict(noise)

    generatedImages = generatedImages.reshape(examples, 64, 64)



    plt.figure(figsize=figsize)

    for i in range(generatedImages.shape[0]):

        plt.subplot(dim[0], dim[1], i+1)

        plt.imshow(generatedImages[i], interpolation='nearest')#, cmap='gray_r')

        plt.axis('off')

    plt.tight_layout()
train(epochs=20000,batch_size=400)
plotGeneratedImages()