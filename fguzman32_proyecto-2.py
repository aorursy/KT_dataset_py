# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt, zipfile

from PIL import Image

import time

from numpy import expand_dims

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ComputeLB = True

DogsOnly = True



import numpy as np, pandas as pd, os

import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



ROOT = '../input/stanford-dogs-dataset/'

if not ComputeLB: ROOT = '../input/'

IMAGES = os.listdir(ROOT + 'images/Images/')

breeds = os.listdir(ROOT + 'annotations/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))



# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# https://www.kaggle.com/paulorzp/show-annotations-and-breeds

if DogsOnly:

    for breed in breeds:

        #print(breed)

        for dog in os.listdir(ROOT+'annotations/Annotation/'+breed):

            try: img = Image.open(ROOT+'images/Images/'+ breed + '/' +dog+'.jpg')

            except: continue           

            tree = ET.parse(ROOT+'annotations/Annotation/'+breed+'/'+dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                w = np.min((xmax - xmin, ymax - ymin))

                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                #hay una foto que tiene shape (64,64,4)

                #para chequear si se tiene shape indicado, hay una foto que tiene shape 64,64,4

                if(np.asarray(img2).shape == (64,64,3)):

                    imagesIn[idxIn,:,:,:] = np.asarray(img2)

                    imagesIn[idxIn,:,:,:] = np.asarray(img2)

                    namesIn.append(breed)

                    idxIn += 1

                else:

                    #foto no tiene shape 64,64,3

                    None

                #if idxIn%1000==0: print(idxIn)

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    namesIn = np.array(namesIn)[idx]

    

# RANDOMLY CROP FULL IMAGES

else:

    x = np.random.choice(np.arange(25000),10000)

    for k in range(len(x)):

        img = Image.open(ROOT + 'images/Images/' + IMAGES[x[k]])

        w = img.size[0]

        h = img.size[1]

        sz = np.min((w,h))

        a=0; b=0

        if w<h: b = (h-sz)//2

        else: a = (w-sz)//2

        img = img.crop((0+a, 0+b, sz+a, sz+b))  

        img = img.resize((64,64), Image.ANTIALIAS)

        imagesIn[idxIn,:,:,:] = np.asarray(img)

        namesIn.append(IMAGES[x[k]])

        if idxIn%1000==0: print(idxIn)

        idxIn += 1

    

# DISPLAY CROPPED IMAGES

print('value of idxIn: ',idxIn)

x = np.random.randint(0,idxIn,25)

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )

        plt.axis('off')

        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)

        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)

        plt.imshow(img)

    plt.show()
daImages = imagesIn[10000:12000,:,:,:]

datagen = ImageDataGenerator(horizontal_flip=True)

it = datagen.flow(daImages, batch_size=1, shuffle=False)



daArray = np.ones((1,64,64,3))



#agregar a imagesIn las nuevas imagenes

for k in range(2000):

    print('k: ',k)

    batch = it.next()

    image = batch[0]

    image = np.expand_dims(image,axis=0)

    imagesIn = np.append(imagesIn, image, axis=0)



print(imagesIn.shape)
from keras.models import Model, Sequential

from keras.layers import Input,Dropout,Activation, Conv2DTranspose,Dense, Conv2D, Reshape, Flatten, concatenate, UpSampling2D, BatchNormalization, LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD, Adam

from keras.initializers import RandomNormal
#Generator

dog = Input((100,))

x = Dense(2048,activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(dog)

x = Reshape((4,4,128))(x)

#0.02 standard deviation

x = Conv2DTranspose(512, kernel_size=5, use_bias=False,padding='same', kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None), strides=(1,1))(x)

#add dropout

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Dropout(0.5)(x)

x = Conv2DTranspose(256, kernel_size=5, use_bias=False,padding='same', kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None), strides= (2,2))(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Dropout(0.5)(x)

x = Conv2DTranspose(128, kernel_size=5, use_bias=False,padding='same', kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None), strides=(2,2))(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2DTranspose(64, kernel_size=5, use_bias=False,padding='same', kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None), strides=(2,2))(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2DTranspose(32, kernel_size=5, use_bias=False,padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None), strides=(2,2))(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

x = Conv2DTranspose(3, kernel_size=5, activation='tanh', padding='same', kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None), strides=(1,1))(x)



generator = Model(dog, x)

generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

generator.summary()
#Discriminator

inp = Input((12288,))

x = Reshape((64,64,3))(inp)



#x = Conv2D(128, (2,2), use_bias=False, activation='relu')(inp)

x = Conv2D(64, strides=(1,1),kernel_size=5, padding='same',use_bias=False, kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)

x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(64, kernel_size=5, strides=(2,2), padding='same', use_bias=False, kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)

x = BatchNormalization()(x)

x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(128, kernel_size=4, strides=(2,2), padding='same', use_bias=False, kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)

x = BatchNormalization()(x)

x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(256, kernel_size=4, strides=(2,2), padding='same', use_bias=False, kernel_initializer= RandomNormal(mean=0.0, stddev=0.02, seed=None))(x)

x = BatchNormalization()(x)

x = LeakyReLU(alpha=0.2)(x)

x = Flatten()(x)

x = Dense(1, activation='sigmoid')(x)



discriminator = Model(inp, x)



discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

discriminator.summary()
#Gan recibe input, codigo de 100 valores

inp = Input(shape=(100,))

# se emplea el generador para generar una foto

x = generator(inp)

#la foto se pone en forma de 12288 para pasarla por discriminador

x = Reshape((12288,))(x)

#el discriminador determina si la foto es perro o no (discriminador tiene entrenamiento)

#discriminator.trainable = False

discriminator.trainable=False

gan_output= discriminator(x)



#modelo recibe codigo y devuelve los resultados de si ese codigo genera foto de perro o no

gan = Model(inp, gan_output)

gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

gan.summary()
num_epochs = 200


#50 epochs. 





indexOfData= 128

start = time.time()

for j in range(num_epochs):

    print(' ############################### GAN Epoch ############################### :',j+1)

    

    #188 steps to complete an epoch

    for d in range(188):

        print('step: ',d+1)

        print('Training Discriminator')

        #128 imagenes

        x_real = ((imagesIn[indexOfData-128:indexOfData,:,:,:]-127.5)/127.5).reshape(-1,12288)

        #aplicando label smoothing a labels

        y_real = np.random.uniform(low=0.7, high=1, size=(128,1))

        #y_real = np.ones((22125,1))

    

        if d==187:

            indexOfData=128

        else:

            indexOfData=indexOfData+128

        

        #128 codigos random de 100

        noises = np.random.rand(128,100)

        x_fake = generator.predict(x=noises).reshape(-1,12288)

        #y_fake = np.zeros((128,1))

        #aplicando label smoothing a labels

        y_fake = np.random.uniform(low=0, high=0.3, size=(128,1))





        #training the with the real data

        #h1 = discriminator.fit(x=x_real, y=y_real, batch_size=128, epochs=1)

        h1 = discriminator.train_on_batch(x=x_real, y=y_real)

        #training with fake data from the generator

        print('loss on real images: ',h1)

        #h2 = discriminator.fit(x=x_fake, y=y_fake, batch_size=128, epochs=1)

        h2 = discriminator.train_on_batch(x=x_fake, y=y_fake)

        print('loss on fake images: ',h2)



        code2 = np.random.rand(128,100)

        #y_trick = np.ones((7000,1))

        #label smoothing

        y_trick = np.random.uniform(low=0.7, high=1, size=(128,1))



        print('Training GAN (Generator)')

        #h3 = gan.fit(x=code2, y=y_trick, batch_size=128, epochs=1)

        h3 = gan.train_on_batch(x=code2, y=y_trick)

        print('loss on generator: ',h3)

    



end = time.time()



totalTime = end-start

totalTime = totalTime/60



print('total elapsed time: ' + str(totalTime) +' minutes')
result = generator.predict_on_batch(x=np.random.rand(25,100))



print(result.shape)



#denormalizando desde [-1,1] a [0,255]

result = (result*127.5)+(127.5)

kk=0

for k in range(5):

  plt.figure(figsize=(15,3))

  for j in range(5):

    plt.subplot(1,5,j+1)

    img = Image.fromarray(result[kk,:,:,:].astype('uint8'))

    plt.axis('off')

    plt.imshow(img)

    kk=kk+1

plt.show()