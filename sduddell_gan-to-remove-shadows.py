# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# feed shadow images to generator , feed shadowless images to discriminator

gan_fake_data = []
des_real_data = []

norm_img = np.zeros((128,128))

fake_data_dir = "/kaggle/input/shadow-images/ISTD_Dataset/train/train_A"

for filenames in sorted(os.listdir(fake_data_dir)):
    shadow_image = cv2.imread(os.path.join(fake_data_dir, filenames))
    shadow_image = cv2.resize(shadow_image,(128,128))
    #norm_img = cv2.normalize(shadow_image,  norm_img, 0, 255, cv2.NORM_MINMAX)
    gan_fake_data.append(shadow_image)

plt.subplot(1,2,1)    
plt.imshow(shadow_image)

plt.subplot(1,2,2)
plt.imshow(norm_img)

real_data_dir = "/kaggle/input/shadow-images/ISTD_Dataset/train/train_C"

for filenames in sorted(os.listdir(real_data_dir)):
    shadow_free_image = cv2.imread(os.path.join(real_data_dir, filenames))
    shadow_free_image = cv2.resize(shadow_free_image,(124,124))
    #norm_img = cv2.normalize(shadow_image,  norm_img, 0, 255, cv2.NORM_MINMAX)
    des_real_data.append(shadow_free_image)
    
gan_fake_data = np.array(gan_fake_data)
des_real_data = np.array(des_real_data)

print(gan_fake_data[0][1:10,1,1])
print(des_real_data.shape)
print(des_real_data[0][1:10,1,1])
#gan_fake_data = gan_fake_data.reshape(gan_fake_data.shape[0],30000)
#des_real_data = des_real_data.reshape(des_real_data.shape[0],30000)

print(gan_fake_data.shape)

def adam_optimizer():
    return Adam(lr=0.000002, beta_1=0.5)

shadow_mask_dir = "/kaggle/input/shadow-images/ISTD_Dataset/train/train_B"
mask_data = []

for filenames in sorted(os.listdir(shadow_mask_dir)):
    mask_dirs = cv2.imread(os.path.join(shadow_mask_dir,filenames))
    mask_dirs = cv2.resize(mask_dirs,(100,100))
    mask_data.append(mask_dirs)


def create_generator():
    #encoder
    generator= keras.Sequential()
    generator.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same',input_shape=(128,128,3)))
    generator.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Dropout(0.2))
    generator.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))

    generator.add(keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Dropout(0.2))
    generator.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    generator.add(keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Dropout(0.2))
    #generator.add(keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(1,1)))

    generator.add(keras.layers.Conv2D(1,kernel_size=(3,3),activation='relu', padding='same'))

    generator.add(keras.layers.Conv2DTranspose(256,kernel_size=(3,3),activation='relu', padding='same'))
    #generator.add(keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu', padding='same'))

    generator.add(keras.layers.UpSampling2D(input_shape = (62,62,256)))
    generator.add(keras.layers.Conv2DTranspose(128,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu', padding='same'))

    generator.add(keras.layers.Conv2DTranspose(64,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding='same'))
    generator.add(keras.layers.Conv2D(3,kernel_size=(3,3),activation='relu', padding='same'))

    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

g=create_generator()
g.summary()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_fake_data_dir = "/kaggle/input/shadow-images/ISTD_Dataset/test/test_A"
test_real_data_dir = "/kaggle/input/shadow-images/ISTD_Dataset/test/test_B"

test_gan_fake_data = []
test_des_real_data = []

for filenames in sorted(os.listdir(test_fake_data_dir)):
    shadow_image = cv2.imread(os.path.join(test_fake_data_dir, filenames))
    shadow_image = cv2.resize(shadow_image,(128,128))
    #norm_img = cv2.normalize(shadow_image,  norm_img, 0, 255, cv2.NORM_MINMAX)
    test_gan_fake_data.append(shadow_image)


for filenames in sorted(os.listdir(test_real_data_dir)):
    shadow_free_image = cv2.imread(os.path.join(test_real_data_dir, filenames))
    shadow_free_image = cv2.resize(shadow_free_image,(124,124))
    #norm_img = cv2.normalize(shadow_image,  norm_img, 0, 255, cv2.NORM_MINMAX)
    test_des_real_data.append(shadow_free_image)
    
test_gan_fake_data = np.array(test_gan_fake_data)
test_des_real_data = np.array(test_des_real_data)

#training the model
g.fit(gan_fake_data,des_real_data,epochs=400,batch_size = 80, shuffle=True,
      validation_data=(test_gan_fake_data, test_des_real_data))
plt.subplot(2,2,1)
plt.imshow(test_gan_fake_data[0])


print(test_gan_fake_data[0].shape)
test_output_image = g.predict_classes(test_gan_fake_data)
print(test_output_image.shape)

#plt.subplot(2,2,2)
#plt.imshow(test_output_image[0])

#plt.subplot(2,2,3)
#plt.imshow(gan_fake_data[0])


print(test_gan_fake_data.shape)
test_output_image = g.predict_classes(gan_fake_data)

print("output shape is " ,test_output_image.shape)
print("first image data is ", test_output_image[0].shape)
#plt.subplot(2,2,4)
#plt.imshow(test_output_image[0])

import cv2
rgb_img = cv2.cvtColor(test_output_image[0], cv2.COLOR_GRAY2RGB)
plt.subplot(2,2,2)
plt.imshow(rgb_img)

"""
def create_discriminator():
    discriminator=Sequential()
    
    discriminator.add(Dense(units=2048,input_dim=30000))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=1024))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=1024))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])
    return discriminator
d =create_discriminator()
d.summary()
"""
print(Input(shape=(10)))

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(30000,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_gan(d,g)
gan.summary()
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    rand_num= np.random.randint(low=0, high = gan_fake_data.shape[0],size = examples)
    generated_images = generator.predict(gan_fake_data[rand_num])
    generated_images = generated_images.reshape(100,100,100,3)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)
def training(epochs=1, batch_size=500):
    
    #Loading the data
    batch_count = gan_fake_data.shape[0] / batch_size
    
    # Creating GAN
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            fake_images= gan_fake_data[np.random.randint(low=0,high=gan_fake_data.shape[0],size= batch_size)]
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(fake_images)

            # Get a random set of  real images
            real_images =des_real_data[np.random.randint(low=0,high=des_real_data.shape[0],size=batch_size)]

            #Construct different batches of  real and fake data 
            X= np.concatenate([real_images, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            fake_images_noise= gan_fake_data[np.random.randint(low=0,high=gan_fake_data.shape[0],size= batch_size)]
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(fake_images_noise, y_gen)
            
        if e == 1 or e % 20 == 0:
           
            plot_generated_images(e, generator)

training(400,128)