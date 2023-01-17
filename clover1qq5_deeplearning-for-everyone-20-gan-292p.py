# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout

from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D

from tensorflow.keras.models import Sequential, Model



import matplotlib.pyplot as plt
generator = Sequential()

generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))

generator.add(BatchNormalization())

generator.add(Reshape((7,7,128)))
generator.add(UpSampling2D())

generator.add(Conv2D(64, kernel_size=5, padding='same'))

generator.add(BatchNormalization())

generator.add(Activation(LeakyReLU(0.2)))

generator.add(UpSampling2D())

generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
discriminator = Sequential()

discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'))

discriminator.add(Activation(LeakyReLU(0.2)))

discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))

discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.trainable= False
ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))

gan = Model(ginput, dis_output)

gan.compile(loss='binary_crossentropy', optimizer='adam')

gan.summary()
def gan_train(epoch, batch_size, saving_interval):

  (X_train, _), (_, _) = mnist.load_data() 

  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

  X_train = (X_train - 127.5) / 127.5  



  true = np.ones((batch_size, 1))

  fake = np.zeros((batch_size, 1))



  for i in range(epoch):

    

          idx = np.random.randint(0, X_train.shape[0], batch_size)

          imgs = X_train[idx]

          d_loss_real = discriminator.train_on_batch(imgs, true)



        

          noise = np.random.normal(0, 1, (batch_size, 100))

          gen_imgs = generator.predict(noise)

          d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)



          

          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

          g_loss = gan.train_on_batch(noise, true)



          print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)



          if i % saving_interval == 0:

         

              noise = np.random.normal(0, 1, (25, 100))

              gen_imgs = generator.predict(noise)



       

              gen_imgs = 0.5 * gen_imgs + 0.5



              fig, axs = plt.subplots(5, 5)

              count = 0

              for j in range(5):

                  for k in range(5):

                      axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')

                      axs[j, k].axis('off')

                      count += 1

              fig.savefig("gan_mnist_%d.png" % i) #fig.savefig(파일명), 책은 error뜸. "gan_images/gan_mnist_%d.png"를 "gan_mnist_%d.png"로만 수정.



gan_train(4001, 32, 200)  