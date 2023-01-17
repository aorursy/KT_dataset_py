# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importando las librerías

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

import random as rn



from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Activation, Dropout, BatchNormalization, Input, LeakyReLU, Reshape

from keras.models import Model

from keras.optimizers import Adam

from keras.initializers import glorot_normal
beagle_dir = '../input/stanford-dogs-dataset/images/Images/n02088364-beagle'

irish_terrier_dir = '../input/stanford-dogs-dataset/images/Images/n02093991-Irish_terrier'

collie_dir = '../input/stanford-dogs-dataset/images/Images/n02106030-collie'

boxer_dir = '../input/stanford-dogs-dataset/images/Images/n02108089-boxer'

shitzu_dir = '../input/stanford-dogs-dataset/images/Images/n02086240-Shih-Tzu'

blenheim_spaniel_dir = '../input/stanford-dogs-dataset/images/Images/n02086646-Blenheim_spaniel'

bluetick_dir = '../input/stanford-dogs-dataset/images/Images/n02088632-bluetick'

toy_terrier_dir = '../input/stanford-dogs-dataset/images/Images/n02087046-toy_terrier'

afghan_hound_dir = '../input/stanford-dogs-dataset/images/Images/n02088094-Afghan_hound'

borzoi_dir = '../input/stanford-dogs-dataset/images/Images/n02090622-borzoi'





X = []

imgsize = 64
# Preparación del Training Data

def training_data(data_dir):

    for img in tqdm(os.listdir(data_dir)):

        path = os.path.join(data_dir,img)

        img = cv2.imread(path)

        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(imgsize,imgsize))

        img = cv2.normalize(img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        

        X.append(np.array(img))
#llamado a la función previamente definida

training_data(beagle_dir)

training_data(irish_terrier_dir)

training_data(collie_dir)

training_data(boxer_dir)

training_data(shitzu_dir)

training_data(blenheim_spaniel_dir)

training_data(bluetick_dir)

training_data(toy_terrier_dir)

training_data(afghan_hound_dir)

training_data(borzoi_dir)
X=np.array(X)

# para mostrar las imágenes

#De mi vector de imagenes saca índices aleatorios y te muestra las imágenes que están en ellos

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(X))

        ax[i,j].imshow(X[l])

            

plt.tight_layout()
image_shape=(64,64,3)

channels=3

#definimos el tamano de la imagen que es de 64x64x3, como tiene 3 canales es a color

latent_dimension= 100

#por cada imagen hay un vector de ruido, en este caso cada uno tendrá 100 números aleatorios
# Definiendo la Arquitectura del Generador



def generator_creator(latent_dimension):

    z= Input(shape=(latent_dimension,)) #capa de entrada, recibe vectores de ruido de x,100 siendo x la cantidad de img que quiero generar (tamano del batch)

    x= Dense(4*4*512, input_dim=latent_dimension)(z)

    x= Reshape ((4,4,512))(x)    #pasa de ser un vector a una matriz

    

    #Ahora que tenemos esra matriz, se hacen las deconvoluciones

    

    x=Conv2DTranspose(256, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    # el parámetro 256 corresponde al numero de canales del filtro

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    # el parámetro 128 corresponde al numero de canales del filtro

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    # el parámetro 128 corresponde al numero de canales del filtro

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    # el parámetro 128 corresponde al numero de canales del filtro

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    image = Conv2DTranspose(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    

    #en este paso ajustamos los valores de la matriz para que estén entre -1 y 1

    model= Model(z, image) #recibe la capa de entrada y la de salida

    

    return model
# Definiendo la Arquitectura del Discriminador



def discriminator_creator(image_shape):

    img=Input(shape = image_shape)

    

    # hacemos convoluciones

    

    x=Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(img)

    #Xavier Initialization (glorot normal)

    #32 es el numero de canales del filtro

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x=Conv2D(512, kernel_size=3, strides=2, padding='same', kernel_initializer=glorot_normal())(x)

    #Xavier Initialization (glorot normal)

    x=BatchNormalization()(x)

    x=LeakyReLU()(x)

    

    x= Flatten()(x)  #Pasamos de matriz a vector

    y= Dense(1, activation='sigmoid')(x) 

    #capa de salida, que tiene 1 sola neurona y usa una función de activación sigmoide porque es clasificacion binaria

    

    model= Model(img, y)

    return model
# Definiendo la GAN



def gan_creator (generator, discriminator):

    

    z= Input (shape=(latent_dimension,)) #La entrada es el ruido

    #Ahora, se le pasa el ruido al generador

    img = generator(z)

    #Ahora, se le pasa esa imagen al discriminador para que haga su predicción

    prediction= discriminator(img)

    

    model= Model(z, prediction) #la entrada es el ruido y la salida la prediccion

    return model
#Se crea el discriminador

discriminator= discriminator_creator(image_shape)

discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0005, beta_1=0.5))

# definimos la funcion de perdida y usaremos adam con learning rate de 0,001



discriminator.trainable= False



generator=generator_creator(latent_dimension)

generator.summary()

dcgan= gan_creator(generator, discriminator) #usamos dcgan



dcgan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, beta_1=0.5))



# el generador y el discriminador no se entrenan simultaneamente.
epochs = 200

batch_size = 32

smooth = 0.1

real = np.ones(shape=(batch_size, 1))

fake = np.zeros(shape=(batch_size, 1))



d_loss = []

g_loss = []



np.random.shuffle(X)



for e in range(epochs + 1):

    for i in range(len(X) // batch_size):

        discriminator.trainable = True

        # Real samples

        X_batch = X[i*batch_size:(i+1)*batch_size]

        d_loss_real = discriminator.train_on_batch(x=X_batch,

                                                   y=real * (1 - smooth))

        

        # Fake Samples

        z = np.random.normal(size=(batch_size, latent_dimension))

        X_fake = generator.predict_on_batch(z)

        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)

         

        # Discriminator loss

        d_loss_batch = 0.5 * (d_loss_real + d_loss_fake)

        

        discriminator.trainable = False

        # Train Generator weights

        g_loss_batch = dcgan.train_on_batch(x=z, y=real)



        print(

            'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(X) // batch_size, d_loss_batch, g_loss_batch),

            100*' ',

            end='\r'

        )

    

    d_loss.append(d_loss_batch)

    g_loss.append(g_loss_batch)

    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')



    if e % 10 == 0:

        samples = 10

        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dimension)))



        for k in range(samples):

            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])

            plt.imshow((x_fake[k]))



        plt.tight_layout()

        plt.show()