# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



from tqdm import tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

from tensorflow import keras

from keras.models import Model

from keras.initializers import RandomNormal, Constant

from keras.layers import Input, Dense, Dropout, Flatten, Reshape , LeakyReLU, PReLU

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, UpSampling2D, Conv2DTranspose,BatchNormalization

from keras import regularizers

from keras.optimizers import SGD, Adam



import time

start_time = time.time()



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PATH = '../input/all-dogs/all-dogs/'

imageNames = os.listdir(PATH)

#print(images)

img = plt.imread(PATH + imageNames[np.random.randint(0,len(imageNames))])

plt.imshow(img)

print(img.size)
PATH_ANNO = '../input/annotation/Annotation/'

breeds = os.listdir(PATH_ANNO)

imagesInput = np.zeros((len(imageNames)*2,64,64,3))

images_breed = []

i = 0

print(imagesInput.shape)

for breed in breeds:

    for dog in os.listdir(PATH_ANNO+breed):

        tree = ET.parse(PATH_ANNO + breed + '/' + dog)

        root = tree.getroot()

        try: img = Image.open(PATH + root.find('filename').text +'.jpg')

        except: continue

        for obj in root.findall('object'):

            bndbox = obj.find('bndbox')

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

            img_crop = img.crop((xmin, ymin, xmax, ymax))

            w = img_crop.size[0]; h = img_crop.size[1];

            a=0; b=0

            if w<h:

                w2 = 64; h2 = int((64/w)*h)

                #b = np.random.randint(0,(h2-64)) if (h2-64 > 0) else 0

            else:

                h2 = 64; w2 = int((64/h)*w)

                #a = np.random.randint(0,(w2-64)) if (w2-64 > 0) else 0

            img_crop = img_crop.resize((w2,h2), Image.ANTIALIAS)

            img_crop = img_crop.crop((0+a, 0+b, 64+a, 64+b))

            imagesInput[i,:,:,:] = np.asarray(img_crop)

            images_breed.append(obj.find('name').text)

            i += 1

imagesInput = imagesInput[:i,:,:,:]        

flip_imagesInput = np.flip(imagesInput,2)

imagesInput = np.vstack((imagesInput,flip_imagesInput))

images_breed = images_breed + images_breed

imagesInput = imagesInput / (255 / 2) - 1

print(imagesInput.shape,len(images_breed))
rnd = np.random.randint(0,imagesInput.shape[0])

plt.imshow(imagesInput[rnd]/2 + .5)

print(imagesInput[0].shape)
def load_img_full():    

    imagesInput = np.zeros((len(imageNames),64,64,3))

    i = 0

    for i, dog in enumerate(imageNames):

        img = Image.open(PATH + dog)

        w = img.size[0]; h = img.size[1];

        a=0; b=0

        if w<h:

            w2 = 64; h2 = int((64/w)*h)

            b = np.random.randint(0,(h2-64)) if (h2-64 > 0) else 0

        else:

            h2 = 64; w2 = int((64/h)*w)

            a = np.random.randint(0,(w2-64)) if (w2-64 > 0) else 0

        img = img.resize((w2,h2), Image.ANTIALIAS)

        img = img.crop((0+a, 0+b, 64+a, 64+b))

        imagesInput[i,:,:,:] = np.asarray(img)

    imagesInput = imagesInput / (255 / 2) - 1

    print(imagesInput.shape)
drop = 0.1

init = RandomNormal(mean=0.0, stddev=0.02) #'glorot_uniform'#



gen_input = Input(shape=(100,))

x = Dense(1024, activation='relu',)(gen_input)

x = BatchNormalization(momentum=0.8)(x)

x = Reshape((4,4,64))(x)



x = Conv2DTranspose(128, (3, 3),strides=(1, 1),padding='same',kernel_initializer=init)(x)

#x = Conv2D(128, (3, 3), padding='same',kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)#momentum=0.8

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

#x = UpSampling2D((2, 2))(x)

x = Dropout(drop)(x, training=True)#(x, training=True)





x = Conv2DTranspose(128, (3, 3),strides=(2, 2),padding='same',kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x, training=True)



x = Conv2DTranspose(128, (5,5),strides=(2, 2),padding='same',kernel_initializer=init)(x)

#x = Conv2D(128, (3, 3), padding='same',kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

#x = UpSampling2D((2, 2))(x)

x = Dropout(drop)(x, training=True)



x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',kernel_initializer=init)(x)

x = BatchNormalization()(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x, training=True)





x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x, training=True)



#x = AveragePooling2D(pool_size=(2, 2),strides=(1, 1),padding='same')(x)



x = Conv2D(3, (5, 5), activation='tanh', padding='same',kernel_initializer=init)(x)



generator = Model(gen_input, x)

generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

generator.summary()
inpt = np.random.random(100*2).reshape(2,100)

img = generator.predict(inpt)

print(img.shape)

plt.imshow(img[0]/2+.5)

plt.imshow(img[1]/2+.5)

#print(img[1]/2+.5)
init = RandomNormal(mean=0.0, stddev=0.02)

drop = 0.0

dis_input = Input(shape=(64,64,3,))

x = Conv2D(32, (5, 5), padding='same',kernel_initializer=init)(dis_input)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x)



#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

x = Conv2D(64, (5, 5), padding='same',strides=(2, 2),kernel_initializer=init)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = BatchNormalization(momentum=0.8)(x)

x = Dropout(drop)(x)



#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

x = Conv2D(64, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x)



#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

x = Conv2D(128, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x)



#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)

x = Conv2D(256, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x)



#x = Conv2D(64, (3, 3), padding='valid',strides=(1, 1),kernel_initializer=init)(x)

#x = LeakyReLU()(x)

#x = BatchNormalization()(x)

#x = Dropout(drop)(x)



x = Conv2D(512, (3, 3), padding='same',strides=(1, 1),kernel_initializer=init)(x)

x = BatchNormalization(momentum=0.8)(x)

x = PReLU(alpha_initializer=Constant(value=0.3))(x)

x = Dropout(drop)(x)



#x = AveragePooling2D(pool_size=(2, 2),strides=(1, 1),padding='same')(x)



x = Flatten()(x)

x = Dense(1 , activation = "sigmoid")(x)

discriminator = Model(dis_input, x)

discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

discriminator.summary()
gan_input = Input(shape=(100,))

discriminator.trainable=False

x = generator(gan_input)

x = discriminator(x)

gan = Model(gan_input, x)

gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

gan.summary()
def gen_noise(batch_size=128):

    noise = np.random.randn(batch_size,100)

    #noise = np.random.normal(size = 100 * batch_size).reshape(batch_size,100)

    # noise =  np.random.random(100 * batch_size).reshape(batch_size,100)

    return noise 
print("--- %s seconds ---" % (time.time() - start_time))
def training(epochs=3, batch_size=256):

    imagesProgress = np.zeros((epochs,64,64,3))

    progress_noise = gen_noise(1)

    for e in range(1,epochs+1):

        #print(imagesInput.shape[0]//batch_size)

        #print('Epoch:', e)

        np.random.shuffle(imagesInput)

        for b in tqdm(range(imagesInput.shape[0]//batch_size)):

            noise = gen_noise(batch_size)

            gen_imgs = generator.predict(noise)

            real_imgs = imagesInput[b*batch_size:(b+1)*batch_size]

            #X = np.concatenate([real_imgs, gen_imgs])

            #y_dis = np.zeros(2*batch_size) 

            #y_dis[:batch_size] = .8 +  np.random.normal(loc=0, scale=.050)

            

            y_dis = np.ones(batch_size) - np.abs(np.random.normal(loc=.2, scale=.1))

            #y_dis = 1 - y_dis if np.random.random() <= .05 else y_dis

            discriminator.trainable=True

            discriminator.train_on_batch(real_imgs,y_dis)

            

            y_dis.fill(0.0)

            y_dis += np.abs(np.random.normal(loc=.2, scale=.1))

            #y_dis = 1 - y_dis if np.random.random() <= .05 else y_dis

            discriminator.trainable=True

            discriminator.train_on_batch(gen_imgs,y_dis)

                        

            noise = gen_noise(batch_size//4)

            #noise[:,np.random.randint(100)] += 1

            y_gen = np.ones(batch_size//4) - .1

            

            discriminator.trainable=False

            gan.train_on_batch(noise, y_gen)

            inpt = np.random.random(100).reshape(1,100)

                    

        inpt = gen_noise(1)

        i = np.random.randint(0,real_imgs.shape[0])

        print('Epoch: ', e,

        ' || Gen: ' , discriminator.predict(generator.predict(inpt))[0,0],

        ' || Dog: ' , discriminator.predict(real_imgs[i:i+1])[0,0])

        #print(np.average(y_dis))

        imagesProgress[e-1,:,:,:] = generator.predict(progress_noise)[0]

    return imagesProgress

try: imagesProgress

except NameError: imagesProgress = np.zeros((0,64,64,3))

imagesProgress = np.vstack((imagesProgress,training(epochs=36)))
columns = 6 ; rows = min(6,(imagesProgress.shape[0] // columns) + 1);

fig=plt.figure(figsize=(32, 5 * rows))

j=0

for i in range(0 , min(36,imagesProgress.shape[0])):

    fig.add_subplot(rows,columns,i+1)

    plt.imshow(imagesProgress[int(j)]/2+.5)

    j += max(1,imagesProgress.shape[0] / 36)

plt.show()
inpt = gen_noise(12)

#inpt[:,np.random.randint(100)] += 1

img = generator.predict(inpt)

#print(discriminator.predict(imagesInput[:10]))

#print(discriminator.predict(img))

#print(img[0,0,:10])

fig=plt.figure(figsize=(32, 10))

columns = 6 ; rows = 2;

for i in range(0 , columns * rows):

    fig.add_subplot(rows,columns,i+1)

    plt.imshow(img[i]/2+.5)

plt.show()
z = zipfile.PyZipFile('images.zip', mode='w')

inpt = gen_noise(10000)

imgs = generator.predict(inpt) 

imgs = imgs + 1

imgs = imgs / 2

print(imgs.shape)

for k in range(10000):

    f = str(k)+'.png'

    img = imgs[k,:,:,:]#.numpy()

    tf.keras.preprocessing.image.save_img(

        f,

        img,

        scale=True

    )

    z.write(f); os.remove(f)

z.close()

#!ls
#inpt = gen_noise(10000)

#img = generator.predict(inpt) 

#img = img + 1

#img = img * (255 / 2)

#np.array(img).min(), np.array(img).max()

#if not os.path.exists('../tmp'):

#    os.mkdir('../tmp')

#for i in range(0,img.shape[0]):

#    plt.imsave('../tmp/dog_'+ str (i)+'.png' , img[i].astype(np.uint8))

#import shutil

#shutil.make_archive('images', 'zip', '../tmp')
#print(img[9].astype(np.uint8))

#im = Image.fromarray(img[0].astype(np.uint8))

#plt.imshow(im)
print("--- %s seconds ---" % (time.time() - start_time))