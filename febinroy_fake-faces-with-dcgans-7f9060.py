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
from PIL import Image

import matplotlib.pyplot as plt

import os

from keras import preprocessing

from keras.models import Sequential

from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

import tensorflow as tf

import os

cwd = os.getcwd()

os.chdir(cwd)

print(os.listdir("../input"))
path_celeb = []

train_path_celeb = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/"

for path in os.listdir(train_path_celeb):

    if '.jpg' in path:

        path_celeb.append(os.path.join(train_path_celeb, path))
new_path=path_celeb[0:50000]

len(new_path)
crop = (30, 55, 150, 175) #croping size for the image so that only the face at centre is obtained

images = [np.array((Image.open(path).crop(crop)).resize((64,64))) for path in new_path]



for i in range(len(images)):

    images[i] = ((images[i] - images[i].min())/(255 - images[i].min()))

    #images[i] = images[i]*2-1  #uncomment this if activation is tanh for generator last layer

    

images = np.array(images) 
train_data=images
print(path_celeb)
len(path_celeb)
print(train_data.shape)
plt.figure(figsize=(10,10))

fig,ax=plt.subplots(2,5)

fig.suptitle("Real Images")

idx=800



for i in range(2):

    for j in range(5):

            ax[i,j].imshow(train_data[idx].reshape(64,64,3))

            #ax[i,j].set_title("Real Image")

            

            idx+=600

            

plt.tight_layout()

plt.show()
X_train = train_data
noise_shape = 100
generator=Sequential()

generator.add(Dense(4*4*512,input_shape=[noise_shape]))

generator.add(Reshape([4,4,512]))

generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization())

generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization())

generator.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization())

generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",

                                 activation='sigmoid'))
generator.summary()
discriminator=Sequential()

discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[64,64, 3]))

discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())

discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())

discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Flatten())

discriminator.add(Dropout(0.5))

discriminator.add(Dense(1,activation='sigmoid'))
discriminator.summary()
GAN =Sequential([generator,discriminator])
discriminator.compile(optimizer='adam',loss='binary_crossentropy')

discriminator.trainable = False
GAN.compile(optimizer='adam',loss='binary_crossentropy')
GAN.layers
GAN.summary()
epochs = 100  #set epoch according to your training dataset size,i had chosen 50k images hence epochs are high as 300...

batch_size = 128
D_loss=[] #list to collect loss for the discriminator model

G_loss=[] #list to collect loss for generator model
with tf.device('/gpu:0'):

 for epoch in range(epochs):

    print(f"Currently on Epoch {epoch+1}")

    

    # For every batch in the dataset

    for i in range(X_train.shape[0]//batch_size):

        

        if (i)%100 == 0:

            print(f"\tCurrently on batch number {i} of {len(X_train)//batch_size}")

            

        noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])

        

        gen_image = generator.predict_on_batch(noise)

        

        train_dataset = X_train[i*batch_size:(i+1)*batch_size]

        #train on real image

        train_label=np.ones(shape=(batch_size,1))

        discriminator.trainable = True

        d_loss1 = discriminator.train_on_batch(train_dataset,train_label)

        

        #train on fake image

        train_label=np.zeros(shape=(batch_size,1))

        d_loss2 = discriminator.train_on_batch(gen_image,train_label)

        

        

        noise=np.random.uniform(-1,1,size=[batch_size,noise_shape])

        train_label=np.ones(shape=(batch_size,1))

        discriminator.trainable = False

        #train the generator

        g_loss = GAN.train_on_batch(noise, train_label)

        D_loss.append(d_loss1+d_loss2)

        G_loss.append(g_loss)

        

         

    if epoch % 5 == 0:

        samples = 10

        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples,100)))



        for k in range(samples):

            plt.subplot(2, 5, k+1)

            plt.imshow(x_fake[k].reshape(64,64,3))

            plt.xticks([])

            plt.yticks([])



        

        plt.tight_layout()

        plt.show()

    print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (epoch+1, d_loss1, d_loss2, g_loss))        

print('Training is complete')
noise=np.random.uniform(-1,1,size=[500,noise_shape])
im=generator.predict(noise)
for i in range(5):

 plt.figure(figsize=(7,7))   

 for k in range(20):

            noise=np.random.uniform(-1,1,size=[100,noise_shape])

            im=generator.predict(noise) 

            plt.subplot(5, 4, k+1)

            plt.imshow(im[k].reshape(64,64,3))

            plt.xticks([])

            plt.yticks([])

 

 plt.tight_layout()

 plt.show()
plt.figure(figsize=(10,10))

plt.plot(G_loss,color='red',label='Generator_loss')

plt.plot(D_loss,color='blue',label='Discriminator_loss')

plt.legend()

plt.xlabel('total batches')

plt.ylabel('loss')

plt.title('Model loss per batch')

plt.show()
import pickle

Pkl_Filename = "DCGAN.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(GAN, file)