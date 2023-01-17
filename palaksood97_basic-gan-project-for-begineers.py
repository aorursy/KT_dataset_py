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
import numpy as np

import os



import keras

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten

from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D # new! 

from keras.optimizers import RMSprop

 

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline
input_images = "/kaggle/input/trees-dataset/full_numpy_bitmap_tree.npy"
data = np.load(input_images)
data.shape
data[4242]
data = data/255

data = np.reshape(data,(data.shape[0],28,28,1)) 

img_w,img_h = data.shape[1:3]

data.shape
plt.imshow(data[4242,:,:,0], cmap='Greys')
def discriminator_builder(depth=64,p=0.4):



    inputs = Input((img_w,img_h,1))

    

    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(inputs)

    conv1 = Dropout(p)(conv1)

    

    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)

    conv2 = Dropout(p)(conv2)

    

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)

    conv3 = Dropout(p)(conv3)

    

    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)

    conv4 = Flatten()(Dropout(p)(conv4))

    

    output = Dense(1, activation='sigmoid')(conv4)



    model = Model(inputs=inputs, outputs=output)

    model.summary()

    

    return model
discriminator = discriminator_builder()
discriminator.compile(loss='binary_crossentropy', 

                      optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0), 

                      metrics=['accuracy'])
def generator_builder(z_dim=100,depth=64,p=0.4):

    

    inputs = Input((z_dim,))

    

    dense1 = Dense(7*7*64)(inputs)

    dense1 = BatchNormalization(momentum=0.9)(dense1) # default momentum for moving average is 0.99

    dense1 = Activation(activation='relu')(dense1)

    dense1 = Reshape((7,7,64))(dense1)

    dense1 = Dropout(p)(dense1)

    

    conv1 = UpSampling2D()(dense1)

    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)

    conv1 = BatchNormalization(momentum=0.9)(conv1)

    conv1 = Activation(activation='relu')(conv1)

    

    conv2 = UpSampling2D()(conv1)

    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)

    conv2 = BatchNormalization(momentum=0.9)(conv2)

    conv2 = Activation(activation='relu')(conv2)

    

    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)

    conv3 = BatchNormalization(momentum=0.9)(conv3)

    conv3 = Activation(activation='relu')(conv3)



    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

  

    model = Model(inputs=inputs, outputs=output)

    model.summary()

    

    return model
generator = generator_builder()
def adversarial_builder(z_dim=100):

    model = Sequential()

    model.add(generator)

    model.add(discriminator)

    model.compile(loss='binary_crossentropy', 

                  optimizer=RMSprop(lr=0.0004, decay=3e-8, clipvalue=1.0), 

                  metrics=['accuracy'])

    model.summary()

    return model
adversarial_model = adversarial_builder()
def make_trainable(net, val):

    net.trainable = val

    for l in net.layers:

        l.trainable = val
def train(epochs=1000,batch=128):

    

    d_metrics = []

    a_metrics = []

    

    running_d_loss = 0

    running_d_acc = 0

    running_a_loss = 0

    running_a_acc = 0

    

    for i in range(epochs):

        

        if i%100 == 0:

            print(i)

        

        real_imgs = np.reshape(data[np.random.choice(data.shape[0],batch,replace=False)],(batch,28,28,1))

        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))



        x = np.concatenate((real_imgs,fake_imgs))

        y = np.ones([2*batch,1])

        y[batch:,:] = 0

        

        make_trainable(discriminator, True)

        

        d_metrics.append(discriminator.train_on_batch(x,y))

        running_d_loss += d_metrics[-1][0]

        running_d_acc += d_metrics[-1][1]

        

        make_trainable(discriminator, False)

        

        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])

        y = np.ones([batch,1])



        a_metrics.append(adversarial_model.train_on_batch(noise,y)) 

        running_a_loss += a_metrics[-1][0]

        running_a_acc += a_metrics[-1][1]

        

        if (i+1)%100 == 0:



            print('Epoch #{}'.format(i+1))

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)

            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)

            print(log_mesg)



            noise = np.random.uniform(-1.0, 1.0, size=[16, 100])

            gen_imgs = generator.predict(noise)



            plt.figure(figsize=(5,5))



            for k in range(gen_imgs.shape[0]):

                plt.subplot(4, 4, k+1)

                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')

                plt.axis('off')

                

            plt.tight_layout()

            plt.show()

    

    return a_metrics, d_metrics
a_metrics_complete, d_metrics_complete = train(epochs=10000)