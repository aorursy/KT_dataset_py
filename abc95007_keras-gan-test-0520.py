# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_Data = pd.read_csv("../input/test.csv")

X_Data.head(10)
import os

import numpy as np

from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

#%matplotlib inline
class GAN(object):

    def __init__(self, width=28, height=28, channels=1):

        self.width = width

        self.height = height

        self.channels = channels

        self.shape = (self.width , self.height , self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()

        self.G.compile(loss="binary_crossentropy" , optimizer = self.optimizer)

        self.D = self.__discriminator()

        self.D.compile(loss="binary_crossentropy" , optimizer = self.optimizer, metrics=["accuracy"])

        

        self.stack_generator_discriminator = self.__stack_generator_discriminator()

        self.stack_generator_discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer)

        self.lossSummary = []

        

    def __generator(self):

        model = Sequential()

        model.add(Dense(256, input_shape=(100,)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense((self.width * self.height * self.channels), activation="tanh"))

        model.add(Reshape((self.width, self.height, self.channels)))

        model.summary()

        return model

    

    def __discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape = self.shape))

        model.add(Dense(512))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        return model



    def __stack_generator_discriminator(self):

        self.D.Trainabel = False;

        model = Sequential()

        model.add(self.G)

        model.add(self.D)

        return model

    

    def train(self, X_train, epochs = 1000, batch=32, save_interal=100):

        for cnt in range(epochs):

            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))

            legit_images = X_train[random_index:random_index + np.int64(batch/2)].reshape(np.int64(batch/2)\

                            , self.width, self.height, self.channels) 

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))

            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))

            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))



            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)



            noise = np.random.normal(0, 1, (batch,100))

            y_mislabled = np.ones((batch,1))

            g_loss = self.stack_generator_discriminator.train_on_batch(noise, y_mislabled)

            self.lossSummary.append((cnt, d_loss[0], d_loss[1], g_loss))

            print ('epoch: %d, [Discriminator :: d_loss: %f :: accuracy %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], d_loss[1]*100, g_loss))

            
if __name__ == '__main__':    

    # Rescale -1 to 1

    X_train = (X_Data.astype(np.float32) - 127.5) / 127.5

    X_train = np.array(X_train)[:,:,np.newaxis]

    X_train.shape



    gan = GAN()

    gan.train(X_train)

            
noise = np.random.normal(0,1,(20,100))

img = gan.G.predict(noise)



for i in range(20):

    plt.subplot(4,5,i+1)

    plt.imshow(img[i].reshape((28,28)))

plt.show()