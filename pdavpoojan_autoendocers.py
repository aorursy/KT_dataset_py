# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk("../input/landset-8/data_jpeg/"):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os, gc, zipfile

import numpy as np, pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



PATH = '../input/landset-8/data_jpeg/'

IMAGES = os.listdir(PATH)

print('There are',len(IMAGES),'images. Here are 5 example filesnames:')

print(IMAGES[:5])
i=0

for i in range(len(IMAGES)):

    if IMAGES[i] == "svn-r6Yb5c":

        print(i)
del IMAGES[18128]
import glob



files = glob.glob('../tmp/**/*.*', recursive=True)



for f in files:

    try:

        os.remove(f)

    except OSError as e:

        print("Error: %s : %s" % (f, e.strerror))
os.rmdir('../tmp/images')

os.rmdir('../tmp')
os.mkdir('../tmp')

os.mkdir('../tmp/images')



# CREATE RANDOMLY CROPPED IMAGES

for i in range(300000):

    try:

        img = Image.open(PATH + IMAGES[i%len(IMAGES)])

        img = img.resize(( 100,int(img.size[1]/(img.size[0]/100) )), Image.ANTIALIAS)

        w = img.size[0]; h = img.size[1]; a=0; b=0

        if w>64: a = np.random.randint(0,w-64)

        if h>64: b = np.random.randint(0,h-64)

        img = img.crop((a, b, 64+a, 64+b))

        img.save('../tmp/images/'+str(i)+'.png','PNG')

        if i%50000==0: print('created',i,'cropped images')

    except:

        print(i)

        pass

print('created 300000 cropped images')
from keras.models import Model

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.callbacks import ModelCheckpoint



BATCH_SIZE = 256; EPOCHS = 20

train_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory('../tmp/',

        target_size=(64,64), shuffle=True, class_mode='input', batch_size=BATCH_SIZE)
# ENCODER

input_img = Input(shape=(64, 64, 3))  

x = Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)



# LATENT SPACE

latentSize = (8,8,32)



# DECODER

direct_input = Input(shape=latentSize)

x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)

x = UpSampling2D((2, 2))(x)

x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)



# COMPILE

encoder = Model(input_img, encoded)

decoder = Model(direct_input, decoded)

autoencoder = Model(input_img, decoder(encoded))



autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=False)

history = autoencoder.fit_generator(train_batches,

        steps_per_epoch = train_batches.samples // BATCH_SIZE,

        epochs = EPOCHS, verbose=2,callbacks=[checkpointer])
print(history.history)
autoencoder.save('my_model.h5')

autoencoder.save_weights('my_model_weights.h5')
import os, gc, zipfile

import numpy as np, pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



images = next(iter(train_batches))[0]

for i in range(5):



    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)

    

    # ORIGINAL IMAGE

    orig = images[i,:,:,:].reshape((-1,64,64,3))

    img = Image.fromarray( (255*orig).astype('uint8').reshape((64,64,3)))

    plt.title('Original')

    plt.imshow(img)



    # LATENT IMAGE

    latent_img = encoder.predict(orig)

    mx = np.max( latent_img[0] )

    mn = np.min( latent_img[0] )

    latent_flat = ((latent_img[0] - mn) * 255/(mx - mn)).flatten(order='F')

    img = Image.fromarray( latent_flat[:2025].astype('uint8').reshape((45,45)), mode='L') 

    plt.subplot(1,3,2)

    plt.title('Latent')

    plt.xlim((-10,55))

    plt.ylim((-10,55))

    plt.axis('off')

    plt.imshow(img)



    # RECONSTRUCTED IMAGE

    decoded_imgs = decoder.predict(latent_img[0].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.subplot(1,3,3)

    plt.title('Reconstructed')

    plt.imshow(img)

    

    plt.show()
from matplotlib.patches import Ellipse



# PROJECT LATENT INTO 2D, AVOID DEAD RELU

latent_img = encoder.predict(images)

latent_img2 = latent_img.reshape((-1,latentSize[0]*latentSize[1]*latentSize[2]))

d = 0; s = 0

while s<0.1:

    x = latent_img2[:,d]

    s = np.std(x); d += 1

s = 0

while s<0.1:

    y = latent_img2[:,d]

    s = np.std(y); d += 1



# CALCULATE ELLIPSOID FROM 256 IMAGES

cov = np.cov(x, y)

lambda_, v = np.linalg.eig(cov)

lambda_ = np.sqrt(lambda_)

for j in [1,2,3]:

    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=lambda_[0]*j*2, 

            height=lambda_[1]*j*2, angle=np.rad2deg(np.arccos(v[0, 0])))

    ell.set_facecolor('None')

    ell.set_edgecolor('black')

    plt.gca().add_artist(ell)

    

# PLOT 256 IMAGES AS DOTS IN LATENT SPACE

plt.scatter(x,y)

d = np.random.multivariate_normal([np.mean(x),np.mean(y)],cov,9)

plt.scatter(d[:,0],d[:,1],color='red',s=100)

plt.title('Leaf Images form an Ellipsoid in Latent Space')

plt.show()
# CREATE 10000 CROPPED IMAGES

x = np.random.choice(np.arange(20000),10000)

images = np.zeros((10000,64,64,3))

for i in range(len(x)):

    try:

        img = Image.open(PATH + IMAGES[x[i]])

        img = img.resize((100,int(img.size[1]/(img.size[0]/100))), Image.ANTIALIAS)

        img = img.crop((18, 0, 82, 64))

        images[i,:,:,:] = np.asarray(img).astype('float32') / 255.

    except:

        pass

    #if i%1000==0: print(i)

        

# CALCULATE ELLIPSOID FROM 10000 IMAGES        

encoded_imgs = encoder.predict(images)

sz = latentSize[0] * latentSize[1] * latentSize[2]

encoded_imgs = encoded_imgs.reshape((-1,sz))

mm = np.mean(encoded_imgs,axis=0)

ss = np.cov(encoded_imgs,rowvar=False)



# GENERATE 9 RANDOM DOG IMAGES

generated = np.random.multivariate_normal(mm,ss,9)

generated = generated.reshape((-1,latentSize[0],latentSize[1],latentSize[2]))
# PLOT 9 RANDOM DOG IMAGES

for k in range(3):

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)

    decoded_imgs = decoder.predict(generated[k*3].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.subplot(1,3,2)

    decoded_imgs = decoder.predict(generated[k*3+1].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.subplot(1,3,3)

    decoded_imgs = decoder.predict(generated[k*3+2].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.show()

# DISTANCE TO MOVE AWAY FROM EXISTING TRAIN IMAGES

beta = 0.35

# GENERATE 9 RANDOM DOG IMAGES

generated = np.random.multivariate_normal(mm,ss,9)

generated = beta*generated + (1-beta)*encoded_imgs[:9]
for k in range(3):

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)

    decoded_imgs = decoder.predict(generated[k*3].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.subplot(1,3,2)

    decoded_imgs = decoder.predict(generated[k*3+1].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.subplot(1,3,3)

    decoded_imgs = decoder.predict(generated[k*3+2].reshape((-1,latentSize[0],latentSize[1],latentSize[2])))

    img = Image.fromarray( (255*decoded_imgs[0]).astype('uint8').reshape((64,64,3)))

    plt.imshow(img)

    plt.show()