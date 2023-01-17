import numpy as np 
import pandas as pd 

from keras.layers import Dense, Dropout, Input, ReLU, UpSampling2D, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import glob

from PIL import Image
import matplotlib.pyplot as plt
import os
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
import tensorflow as tf

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path_arts = []
train_path_arts = "/kaggle/input/abstract-art-gallery/Abstract_gallery/"
for path in os.listdir(train_path_arts):
    if '.jpg' in path:
        path_arts.append(os.path.join(train_path_arts, path))
    
new_path=path_arts
images = [np.array((Image.open(path)).resize((128,128))) for path in new_path] 

for i in range(len(images)):
    images[i] = ((images[i] - images[i].min())/(255 - images[i].min()))
    #images[i] = images[i]*2-1  #uncomment this if activation is tanh for generator last layer
    
images = np.array(images) 
train_data=images
len(train_data)
plt.figure(figsize=(10,10))
fig,ax=plt.subplots(2,5)
fig.suptitle("Real Images")
idx=0

for i in range(2):
    for j in range(5):
            ax[i,j].imshow(train_data[idx].reshape(128,128,3))
            #ax[i,j].set_title("Real Image")
            
            idx+=200
            
plt.tight_layout()
plt.show()
X_train = train_data 
noise_shape = 100
generator = Sequential()

generator.add(Dense(4*4*512,input_shape=[noise_shape])) 
generator.add(Reshape([4,4,512]))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D()) 
generator.add(Conv2D(1024, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D()) 
generator.add(Conv2D(512, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D())
generator.add(Conv2D(256, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D()) 
generator.add(Conv2D(128, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(UpSampling2D()) 
generator.add(Conv2D(64, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(Conv2D(32, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))

generator.add(Conv2D(3, kernel_size=3, padding="same"))
generator.add(Activation("tanh"))
generator.summary()
discriminator=Sequential()
discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[128,128, 3]))
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
GAN.summary()
D_loss=[] 
G_loss=[]
import time

epochs = 400
batch_size = 64
current_time = time.time()

with tf.device('/gpu:0'):
 for e in range(epochs):
    start_time = time.time()
    for _ in range(batch_size):
        # I reccomend you to look "Training Diagram" (at the top) 
        noise = np.random.normal(0,1, [batch_size,100])
        
        generated_images = generator.predict(noise)
       
        image_batch = X_train[np.random.randint(low = 0, high = X_train.shape[0], size = batch_size)] #get samples from real data
        
        x = np.concatenate([image_batch, generated_images])
        
        y_dis = np.zeros(batch_size*2) 
        y_dis[:batch_size] = 1 # we labeled real images as 1 and generated images as 0
        
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x,y_dis) # we are training discriminator (train_on_batch)
        
        noise = np.random.normal(0,1,[batch_size,100])
        
        y_gen = np.ones(batch_size) # our generator says "these images are real"
        
        discriminator.trainable = False
        
        g_loss = GAN.train_on_batch(noise, y_gen) #train_on_batch
        
        D_loss.append(d_loss)
        G_loss.append(g_loss)
        
    if (e%20 == 0) or (e == epochs-1) :
        print("epochs: ",e)
    if e == epochs-1:
        print("Time since start: {}".format(np.round(start_time - current_time)))
        print("Training Complete.")
    
    # printing results
    if e%100 == 0:
        print("Time since start: {}".format(np.round(start_time - current_time)))
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128,128, 3)

        fig, axs = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
        axs[0,0].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[0,0].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128,128, 3)
        axs[0,1].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[0,1].axis("off")
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128,128, 3)
        axs[0,2].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[0,2].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128, 128,3)
        axs[1,0].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[1,0].axis("off")

        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128,128,3)
        axs[1,1].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[1,1].axis("off")
        
        noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
        generated_images = generator.predict(noise)
        generated_images = generated_images #.reshape(128,128,3)
        axs[1,2].imshow(generated_images[66].reshape(128,128,3), interpolation = "nearest")
        axs[1,2].axis("off")
        
        fig.suptitle("{} Epochs Result".format(str(e)),fontsize=15)
        plt.show()   
from keras import models
generator.save("generator_abstract_art.h5") # you can donwload this model from output part of this kernel.
discriminator.save("discriminator_abstract_art.h5") # you can donwload this model from output part of this kernel.
fig = plt.figure(figsize = (12,6))
plt.plot(noise)
plt.title("Noise")
plt.show()
index = 0
D_loss2 = []
G_loss2 = []

for i in range(1,epochs+1):
    D_loss2.append(D_loss[index])
    G_loss2.append(G_loss[index])
    index += batch_size

fig, axs = plt.subplots(2, 1, figsize = (12,10))

axs[0].plot(D_loss2, label = "Discriminator Loss")
axs[1].plot(G_loss2, label = "Generator Loss", color = "orange")

axs[0].set_xlabel("Epochs")
axs[1].set_xlabel("Epochs")

axs[0].set_ylabel("Loss")
axs[1].set_ylabel("Loss")

axs[0].set_title("Discriminator Loss")
axs[1].set_title("Generator Loss")
plt.show()
fig = plt.figure(figsize = (12,8))
noise = np.random.normal(loc = 0, scale = 1, size=[100,100])
generated_images = generator.predict(noise)
generated_images = generated_images 
plt.imshow(generated_images[66].reshape(128,128,3))
plt.axis("off")
plt.title("{} Epochs Result".format(str(e+1)),fontsize=15)
plt.show()