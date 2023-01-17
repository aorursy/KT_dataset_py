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
import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

import random as rn



from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Activation, Dropout, BatchNormalization, Input, LeakyReLU, Reshape

from keras.models import Model

from keras.optimizers import Adam

from keras.initializers import glorot_normal
chihuahua_dir = '../input/stanford-dogs-dataset/images/Images/n02085620-Chihuahua'

japanese_spaniel_dir = '../input/stanford-dogs-dataset/images/Images/n02085782-Japanese_spaniel'

maltese_dir = '../input/stanford-dogs-dataset/images/Images/n02085936-Maltese_dog'

pekinese_dir = '../input/stanford-dogs-dataset/images/Images/n02086079-Pekinese'

shitzu_dir = '../input/stanford-dogs-dataset/images/Images/n02086240-Shih-Tzu'

blenheim_spaniel_dir = '../input/stanford-dogs-dataset/images/Images/n02086646-Blenheim_spaniel'

papillon_dir = '../input/stanford-dogs-dataset/images/Images/n02086910-papillon'

toy_terrier_dir = '../input/stanford-dogs-dataset/images/Images/n02087046-toy_terrier'

afghan_hound_dir = '../input/stanford-dogs-dataset/images/Images/n02088094-Afghan_hound'

basset_dir = '../input/stanford-dogs-dataset/images/Images/n02088238-basset'

golden_retriever_dir='../input/stanford-dogs-dataset/images/Images/n02099601-golden_retriever'

pug_dir='../input/stanford-dogs-dataset/images/Images/n02110958-pug'

Doberman_dir='../input/stanford-dogs-dataset/images/Images/n02107142-Doberman'

silky_terrier_dir='../input/stanford-dogs-dataset/images/Images/n02097658-silky_terrier'

Border_collie_dir='../input/stanford-dogs-dataset/images/Images/n02106166-Border_collie'

Rottweiler_dir='../input/stanford-dogs-dataset/images/Images/n02106550-Rottweiler'

Siberian_husky_dir='../input/stanford-dogs-dataset/images/Images/n02110185-Siberian_husky'

French_bulldog_dir='../input/stanford-dogs-dataset/images/Images/n02108915-French_bulldog'

Yorkshire_terrier_dir='../input/stanford-dogs-dataset/images/Images/n02094433-Yorkshire_terrier'

Norwich_terrier_dir='../input/stanford-dogs-dataset/images/Images/n02094258-Norwich_terrier'

Scotch_terrier_dir='../input/stanford-dogs-dataset/images/Images/n02097047-miniature_schnauzer'

miniature_schnauzer_dir='../input/stanford-dogs-dataset/images/Images/n02097298-Scotch_terrier'

Pomeranian_dir='../input/stanford-dogs-dataset/images/Images/n02112018-Pomeranian'

Shetland_sheepdog_dir='../input/stanford-dogs-dataset/images/Images/n02105855-Shetland_sheepdog'

Irish_wolfhound_dir='../input/stanford-dogs-dataset/images/Images/n02090721-Irish_wolfhound'

borzoi_dir='../input/stanford-dogs-dataset/images/Images/n02090622-borzoi'

redbone_dir='../input/stanford-dogs-dataset/images/Images/n02090379-redbone'

Walker_hound_dir='../input/stanford-dogs-dataset/images/Images/n02089867-Walker_hound'

black_and_tan_coonhound_dir='../input/stanford-dogs-dataset/images/Images/n02089078-black-and-tan_coonhound'

giant_schnauzer_dir='../input/stanford-dogs-dataset/images/Images/n02097130-giant_schnauzer'









X = []

imgsize = 64
def training_data(data_dir):

    for img in tqdm(os.listdir(data_dir)):

        path = os.path.join(data_dir,img)

        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (imgsize, imgsize))

        img = cv2.normalize(img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



        X.append(np.array(img))
training_data(Siberian_husky_dir)

training_data(silky_terrier_dir)

training_data(Rottweiler_dir)

training_data(chihuahua_dir)

training_data(French_bulldog_dir)

training_data(golden_retriever_dir)

training_data(maltese_dir)

training_data(pug_dir)

training_data(Doberman_dir)

training_data(blenheim_spaniel_dir)

training_data(papillon_dir)

training_data(afghan_hound_dir)

training_data(basset_dir)

training_data(Yorkshire_terrier_dir)

training_data(Norwich_terrier_dir)

training_data(Scotch_terrier_dir)

training_data(miniature_schnauzer_dir)

training_data(Pomeranian_dir)

training_data(Shetland_sheepdog_dir)

training_data(Border_collie_dir)

training_data(Irish_wolfhound_dir)

training_data(borzoi_dir)

training_data(redbone_dir)

training_data(Walker_hound_dir)

training_data(black_and_tan_coonhound_dir)

training_data(giant_schnauzer_dir)









X = np.array(X)

print(X.shape)
fig,ax=plt.subplots(7,2)

fig.set_size_inches(15,15)

for i in range(7):

    for j in range (2):

        l=rn.randint(0,len(X))

        ax[i,j].imshow(X[l])

            

plt.tight_layout()
image_shape= (64,64,3)

channels=3

noise_vector=100

#por cada imagen hay un vector de ruido
#funcion encargada que define el generador



def generador(noise_vector):

    #capa de entrada

    z=Input(shape=(noise_vector,))#va a recibir vectores de tmanao x,100 siendo x la cantidad de imagenes a generar en el batch

    x= Dense(4*4*512, input_dim=noise_vector)(z)

    x= Reshape ((4,4,512))(x)# x pasa de ser un vector a una matriz

    

    #con esta nueva matriz se hacen las deconvoluciones

    x=Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    #256 es el numero de canales de filtro

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

        #con esta nueva amtriz se hacen las deconvoluciones

    x=Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)



        #con esta nueva  matriz se hacen las deconvoluciones

    x=Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

        #con esta nueva matriz se hacen las deconvoluciones

    x=Conv2DTranspose(32, kernel_size=5, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

    imagen= Conv2DTranspose(channels,kernel_size=5, strides=1, padding='same', activation='tanh')(x)

    #se ajustarpn los valores de la matriz para que esten entre -1 1 porque de esta forma se entrena mas rapido

    model=Model(z,imagen)#recibe la capa de entrada y la de salida

    return model

#se define arquitectura del discriminador

def discriminador(image_shape):

    img=Input(shape= image_shape)

    

     #con esta nueva matriz se hacen las convoluciones

    x=Conv2D(32, kernel_size=3, strides=2, padding='same',kernel_initializer=glorot_normal())(img)

    #inicializacion con xavier con el glorot normal

    #256 es el numero de canales de filtro

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

        #con esta nueva matriz se hacen las convoluciones

    x=Conv2D(64, kernel_size=3, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

        #con esta nueva matriz se hacen las convoluciones

    x=Conv2D(128, kernel_size=3, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

        #con esta nueva matriz se hacen las convoluciones

    x=Conv2D(256, kernel_size=3, strides=1, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

    x=Conv2D(512, kernel_size=3, strides=2, padding='same',kernel_initializer=glorot_normal())(x)

    #inicializacion con xavier con el glorot normal

    x=BatchNormalization()(x)

    x=LeakyReLU(0.2)(x)

    

    x= Flatten()(x)#pasa de matriz a vector

    

    y= Dense(1,activation='sigmoid')(x)# capa de salida de 1 neurona y usa sigmoid por ser clasificacion binaria

    

    model= Model(img, y)

    return model

    
#se define la GAN

def gan_creator (generator , discriminator):

    z=Input (shape=(noise_vector,)) #entrada de ruido

    #paso el ruido al generador

    img=generator(z)

    #paso imagen al discriminador para que haga la prediccion

    prediction= discriminator(img)

    model= Model(z,prediction) #entrado el ruido y salida prediccion

    return model
#crwnado el discriminador

discriminator =discriminador(image_shape)

discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002)) # funcion de perdida y usamos adam con ese learnng rate



discriminator.trainable= False

#creando el generador

generator=generador(noise_vector)

generator.summary()

#creando la gan

dcgan= gan_creator(generator, discriminator)



dcgan.compile(loss='binary_crossentropy', optimizer=Adam(0.0004))
epochs = 500

batch_size = 16

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

        d_loss_real = discriminator.train_on_batch(x=X_batch,y=real * (1 - smooth))

        

        # Fake Samples

        z = np.random.normal(size=(batch_size, noise_vector))

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

        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, noise_vector)))



        for k in range(samples):

            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])

            plt.imshow((x_fake[k]))



        plt.tight_layout()

        plt.show()