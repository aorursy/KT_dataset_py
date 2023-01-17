import numpy as np 

import matplotlib.pyplot as plt 

import keras 

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Dropout 

from keras.layers import BatchNormalization, Activation, ZeroPadding2D 

from keras.layers.advanced_activations import LeakyReLU 

from keras.layers.convolutional import UpSampling2D, Conv2D 

from keras.models import Sequential, Model 

from keras.optimizers import Adam,SGD 

from keras.layers.convolutional import UpSampling2D, Conv2D 

from keras.models import Sequential, Model 

from keras.optimizers import Adam,SGD 
#Loading the CIFAR10 data 

(x_train, y_train), (_, _) = keras.datasets.cifar100.load_data() 



save_path = '/kaggle/working/'

print(x_train)
print(y_train)
import matplotlib.pyplot as plt

n = 10

plt.figure(figsize=(20,8))

for i in range(n):

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_train[i])

    ax.get_xaxis().set_visible(True)

    ax.get_yaxis().set_visible(True)

plt.show()
#Defining the Input shape 

image_shape = (32, 32, 3) 



dimension = 100

def build_generator(): 



		model = Sequential() 



		#Building the input layer 

		model.add(Dense(128 * 8 * 8, activation="relu", 

						input_dim=dimension)) 

		model.add(Reshape((8, 8, 128))) 

		

		model.add(UpSampling2D()) 

		

		model.add(Conv2D(128, kernel_size=3, padding="same")) 

		model.add(BatchNormalization(momentum=0.78)) 

		model.add(Activation("relu")) 

		

		model.add(UpSampling2D()) 

		

		model.add(Conv2D(64, kernel_size=3, padding="same")) 

		model.add(BatchNormalization(momentum=0.78)) 

		model.add(Activation("relu")) 

		

		model.add(Conv2D(3, kernel_size=3, padding="same")) 

		model.add(Activation("tanh")) 



		noise = Input(shape=(dimension,)) 

		image = model(noise) 



		return Model(noise, image) 

def build_discriminator(): 



		model = Sequential() 



		model.add(Conv2D(32, kernel_size=3, strides=2, 

						input_shape=image_shape, padding="same")) 

		model.add(LeakyReLU(alpha=0.2)) 

		model.add(Dropout(0.25)) 

		

		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) 

		model.add(ZeroPadding2D(padding=((0,1),(0,1)))) 

		model.add(BatchNormalization(momentum=0.82)) 

		model.add(LeakyReLU(alpha=0.25)) 

		model.add(Dropout(0.25)) 

		

		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) 

		model.add(BatchNormalization(momentum=0.82)) 

		model.add(LeakyReLU(alpha=0.2)) 

		model.add(Dropout(0.25)) 

		

		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) 

		model.add(BatchNormalization(momentum=0.8)) 

		model.add(LeakyReLU(alpha=0.25)) 

		model.add(Dropout(0.25)) 

		

		#Building the output layer 

		model.add(Flatten()) 

		model.add(Dense(1, activation='sigmoid')) 



		image = Input(shape=image_shape) 

		validity = model(image) 



		return Model(image, validity) 

def display_images(): 

        r, c = 5,5

        noise = np.random.normal(0, 1, (r * c,dimension)) 

        generated_images = generator.predict(noise) 



        #Scaling the generated images 

        generated_images = 0.5 * generated_images + 0.5

                    

        fig, axs = plt.subplots(r, c) 

        count = 0

        for i in range(r): 

            for j in range(c): 

                axs[i,j].imshow(generated_images[count, :,:,]) 

                axs[i,j].axis('off') 

                plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')

                count += 1

        plt.show() 

        plt.close()

# Building and compiling the discriminator 

discriminator = build_discriminator() 

discriminator.compile(loss='binary_crossentropy', 

					optimizer=Adam(0.0002,0.5), 

					metrics=['accuracy']) 



#Making the Discriminator untrainable 

#so that the generator can learn from fixed gradient 

discriminator.trainable = False



# Building the generator 

generator = build_generator() 



#Defining the input for the generator and generating the images 

dummy = Input(shape=(dimension,)) 

image = generator(dummy) 





#Checking the validity of the generated image 

valid = discriminator(image) 



#Defining the combined model of the Generator and the Discriminator 

combined_network = Model(dummy, valid) 

combined_network.compile(loss='binary_crossentropy', 

						optimizer=Adam(0.0002,0.5)) 

from keras.utils import plot_model

generator.summary()

#plt.savefig('/kaggle/working/generator.png')

plot_model(generator, to_file='/kaggle/working/generator.png', show_shapes=True,show_layer_names=True)
from keras.utils import plot_model

discriminator.summary()

plot_model(discriminator, to_file='/kaggle/working/discriminator.png', show_shapes=True,show_layer_names=True)
num_epochs=50000

batch_size=32

display_after=1000

losses=[] 



#Normalizing the input 

x_train = (x_train / 127.5) - 1.
#Defining the Adversarial ground truths 

valid = np.ones((batch_size, 1)) 



#Adding some noise 

valid += 0.05 * np.random.random(valid.shape) 

fake = np.zeros((batch_size, 1)) 

fake += 0.05 * np.random.random(fake.shape) 



for epoch in range(num_epochs): 

            

            #Training the Discriminator 

              

            #Sampling a random half of images 

            index = np.random.randint(0, x_train.shape[0], batch_size) 

            images = x_train[index] 



            #Sampling noise and generating a batch of new images 

            noise = np.random.normal(0, 1, (batch_size, dimension)) 

            generated_images = generator.predict(noise) 





            #Training the discriminator to detect more accurately 

            #whether a generated image is real or fake 

            discm_loss_real = discriminator.train_on_batch(images, valid) 

            discm_loss_fake = discriminator.train_on_batch(generated_images, fake) 

            discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake) 

          

            #Training the Generator 

            

            #Training the generator to generate images 

            #which pass the authenticity test 

            genr_loss = combined_network.train_on_batch(noise, valid) 

             

            #Tracking the progress				 

            if epoch % display_after == 0: 

              display_images()

            ''''plt.savefig(f'{save_path}/gan-images_epoch-{epoch}.png')

            plt.show()'''

            

       

                
#Plotting some of the original images 

solid = x_train[:30] 

solid = 0.5 * solid + 0.5

f, ax = plt.subplots(5,6, figsize=(15,8)) 

for i, image in enumerate(solid): 

	ax[i//6, i%6].imshow(image) 

	ax[i//6, i%6].axis('on') 

plt.savefig("/kaggle/working/originalimages.png")

plt.show() 

#Plotting some of the last batch of generated images 

noise = np.random.normal(size=(30, dimension)) 

generated_images = generator.predict(noise) 

generated_images = 0.5 * generated_images + 0.5

f, ax = plt.subplots(5,6, figsize=(15,8)) 

for i, image in enumerate(generated_images): 

	ax[i//6, i%6].imshow(image) 

	ax[i//6, i%6].axis('on') 

plt.savefig("/kaggle/working/reconstructedImages.png")

plt.show() 

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import os
import re

import sys

from PIL import Image, ImageDraw



def func(x):

  for i in x:

    if(i.isdigit()):

      print(i)

      return int(i)

  return 10000000000



image_names = os.listdir(save_path)



frames = []

#for image in sorted(image_names, key=lambda name: int(''.join(i for i in name if i.isdigit()))):

for image in sorted(image_names, key=func):

    if(bool(re.search(r'\d', image))):

        print(image)

        frames.append(Image.open(save_path + '/' + image))



frames[0].save('/kaggle/working/reconstruction_process.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
from IPython.display import Image

Image("../working/reconstruction_process.gif")
discriminator.save('/kaggle/working/discriminator.h5')

generator.save('/kaggle/working/dcgenerator.h5')