# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import random

from tqdm import tqdm

import shutil



ComputeLB = False

DogsOnly = True



import numpy as np, pandas as pd, os

import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



#ROOT = '../input/generative-dog-images/'

ROOT = '../input/'

#if not ComputeLB: ROOT = '../input/'

IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))

imagesIn2 = np.zeros((25000,64,64))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# https://www.kaggle.com/paulorzp/show-annotations-and-breeds

if DogsOnly:

    for breed in breeds:

        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

            except: continue           

            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

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

                img3 = img2

                img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)

                img3 = img3.resize((64,64), Image.ANTIALIAS)

                img3 = img3.convert('LA')

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                imagesIn2[idxIn,:,:] = np.asarray(img3)[:,:,0]

                #if idxIn%1000==0: print(idxIn)

                namesIn.append(breed)

                idxIn += 1

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    imagesIn2 = imagesIn2[idx,:,:]

    namesIn = np.array(namesIn)[idx]

    

# RANDOMLY CROP FULL IMAGES

else:

    IMAGES = np.sort(IMAGES)

    np.random.seed(810)

    x = np.random.choice(np.arange(20579),10000)

    np.random.seed(None)

    for k in range(len(x)):

        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])

        w = img.size[0]; h = img.size[1];

        if (k%2==0)|(k%3==0):

            w2 = 100; h2 = int(h/(w/100))

            a = 18; b = 0          

        else:

            a=0; b=0

            if w<h:

                w2 = 64; h2 = int((64/w)*h)

                b = (h2-64)//2

            else:

                h2 = 64; w2 = int((64/h)*w)

                a = (w2-64)//2

        img = img.resize((w2,h2), Image.ANTIALIAS)

        img = img.crop((0+a, 0+b, 64+a, 64+b))    

        imagesIn[idxIn,:,:,:] = np.asarray(img)

        namesIn.append(IMAGES[x[k]])

        #if idxIn%1000==0: print(idxIn)

        idxIn += 1

    
img3.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(imagesIn2[101,:,:], cmap='Greys')
#plt.imshow(Image.fromarray( (imagesIn[101]).astype('uint8').reshape((64,64))), cmap='Greys')
plt.imshow(Image.fromarray( (imagesIn[101]).astype('uint8').reshape((64,64,3))))
from keras.models import Model

from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD, Adam

def define_discriminator(in_shape=(28,28,1)):

    model = Sequential()

# downsample

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))

    model.add(LeakyReLU(alpha=0.2))

    # downsample

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))

    model.add(LeakyReLU(alpha=0.2))

# classifier

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

# compile model

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
def define_discriminator(in_shape=(28,28,1), n_classes=10):

	# label input

	in_label = Input(shape=(1,))

	# embedding for categorical input

	li = Embedding(n_classes, 50)(in_label)

	# scale up to image dimensions with linear activation

	n_nodes = in_shape[0] * in_shape[1]

	li = Dense(n_nodes)(li)

	# reshape to additional channel

	li = Reshape((in_shape[0], in_shape[1], 1))(li)

	# image input

	in_image = Input(shape=in_shape)

	# concat label as a channel

	merge = Concatenate()([in_image, li])

	# downsample

	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)

	fe = LeakyReLU(alpha=0.2)(fe)

	# downsample

	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)

	fe = LeakyReLU(alpha=0.2)(fe)

	# flatten feature maps

	fe = Flatten()(fe)

	# dropout

	fe = Dropout(0.4)(fe)

	# output

	out_layer = Dense(1, activation='sigmoid')(fe)

	# define model

	model = Model([in_image, in_label], out_layer)

	# compile model

	opt = Adam(lr=0.0002, beta_1=0.5)

	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model
NNimage = 4096
# BUILD DISCRIMINATIVE NETWORK

#dog = Input((12288,))

dog = Input((NNimage,))

dogName = Input((10000,))

#x = Dense(12288, activation='sigmoid')(dogName)

x = Dense(NNimage, activation='sigmoid')(dogName)

#x = Reshape((2,12288,1))(concatenate([dog,x]))

x = Reshape((2,NNimage,1))(concatenate([dog,x]))

x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)

discriminated = Flatten()(x)



# COMPILE

discriminator = Model([dog,dogName], discriminated)

discriminator.get_layer('conv').trainable = False

discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# TRAINING DATA

#NNimage = 12288

#NNimage = 16384

train_y = (imagesIn2[:10000,:,:]/255.).reshape((-1,NNimage))

train_X = np.zeros((10000,10000))

for i in range(10000): train_X[i,i] = 1

zeros = np.zeros((10000,NNimage))



# TRAIN NETWORK

lr = 0.5

for k in range(4):

    annealer = LearningRateScheduler(lambda x: lr)

    h = discriminator.fit([zeros,train_X], train_y, epochs = 20, batch_size=256, callbacks=[annealer], verbose=0)

    print('Epoch',(k+1)*10,'/50 - loss =',h.history['loss'][-1] )

    if h.history['loss'][-1]<0.530: lr = 0.1
print('Epoch',(k+1)*10,'/50 - loss =',h.history['loss'][-1] )
def define_generator(latent_dim):

	model = Sequential()

	# foundation for 7x7 image

	n_nodes = 128 * 7 * 7

	model.add(Dense(n_nodes, input_dim=latent_dim))

	model.add(LeakyReLU(alpha=0.2))

	model.add(Reshape((7, 7, 128)))

	# upsample to 14x14

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

	model.add(LeakyReLU(alpha=0.2))

	# upsample to 28x28

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

	model.add(LeakyReLU(alpha=0.2))

	# generate

	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))

	return model
def define_generator(latent_dim, n_classes=120):

	# label input

	in_label = Input(shape=(1,))

	# embedding for categorical input

	li = Embedding(n_classes, 50)(in_label)

	# linear multiplication

	n_nodes = 7 * 7

	li = Dense(n_nodes)(li)

	# reshape to additional channel

	li = Reshape((7, 7, 1))(li)

	# image generator input

	in_lat = Input(shape=(latent_dim,))

	# foundation for 7x7 image

	n_nodes = 128 * 7 * 7

	gen = Dense(n_nodes)(in_lat)

	gen = LeakyReLU(alpha=0.2)(gen)

	gen = Reshape((7, 7, 128))(gen)

	# merge image gen and label input

	merge = Concatenate()([gen, li])

	# upsample to 14x14

	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)

	gen = LeakyReLU(alpha=0.2)(gen)

	# upsample to 28x28

	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)

	gen = LeakyReLU(alpha=0.2)(gen)

	# output

	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)

	# define model

	model = Model([in_lat, in_label], out_layer)

	return model
# BUILD GENERATOR NETWORK

seed = Input((10000,))

generated = Dense(NNimage, activation='linear')(seed)



# COMPILE

generator = Model(seed, [generated,Reshape((10000,))(seed)])

generator.summary()
#gan.summary()
discriminator.trainable=False    

gan_input = Input(shape=(10000,))

x = generator(gan_input)

gan_output = discriminator(x)



# COMPILE GAN

gan = Model(gan_input, gan_output)

gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])

gan.compile(optimizer=Adam(5), loss='mean_squared_error')

train = np.zeros((10000,10000))

for i in range(10000): train[i,i] = 1

zeros = np.zeros((10000,NNimage))



# TRAIN NETWORKS

lr = 5.

#for k in range(50):  

for k in range(20):

    # BEGIN DISCRIMINATOR COACHES GENERATOR

    annealer = LearningRateScheduler(lambda x: lr)

    h = gan.fit(train, zeros, epochs = 1, batch_size=256, callbacks=[annealer], verbose=0)

    if (k<10)|(k%5==4):

        print('Epoch',(k+1)*10,'/500 - loss =',h.history['loss'][-1] )

    if h.history['loss'][-1] < 25: lr = 1.

    if h.history['loss'][-1] < 1.5: lr = 0.5

        

    # DISPLAY GENERATOR LEARNING PROGRESS

    if k<10:        

        plt.figure(figsize=(15,3))

        for j in range(5):

            xx = np.zeros((10000))

            xx[np.random.randint(10000)] = 1

            plt.subplot(1,5,j+1)

            #img = generator.predict(xx.reshape((-1,10000)))[0].reshape((-1,64,64,3))

            img = generator.predict(xx.reshape((-1,10000)))[0]

            #img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))

            #img = Image.fromarray( (img).astype('uint8').reshape((128,128)))

            img = Image.fromarray( (img).astype('uint8').reshape((64,64)))

            plt.axis('off')

            plt.imshow(img, cmap='Greys')

        plt.show()  
import os

import sys

import random

import warnings



import numpy as np

import pandas as pd

import cv2



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

import skimage

from PIL import Image

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.util import crop, pad

from skimage.morphology import label

from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

from sklearn.model_selection import train_test_split



from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.models import Model, load_model,Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape, Embedding

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K



import tensorflow as tf



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed

os.listdir('../')
# inception = InceptionResNetV2(weights=None, include_top=True)

# inception.load_weights('../input/image-colorization/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')

# inception.graph = tf.get_default_graph()
IMG_WIDTH = 64

IMG_HEIGHT = 64

IMG_CHANNELS = 3

INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 1)

idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))



IMAGES = np.sort(IMAGES)

#np.random.seed(2019)

x = np.random.choice(np.arange(20579),10000)

#np.random.seed(None)

for k in range(len(x)):

    img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])

    w = img.size[0]; h = img.size[1];

    if (k%2==0)|(k%3==0):

        w2 = 100; h2 = int(h/(w/100))

        a = 18; b = 0          

    else:

        a=0; b=0

        if w<h:

            w2 = 64; h2 = int((64/w)*h)

            b = (h2-64)//2

        else:

            h2 = 64; w2 = int((64/h)*w)

            a = (w2-64)//2

    img = img.resize((w2,h2), Image.ANTIALIAS)

    img = img.crop((0+a, 0+b, 64+a, 64+b))    

    imagesIn[idxIn,:,:,:] = np.asarray(img)

    namesIn.append(IMAGES[x[k]])

    #if idxIn%1000==0: print(idxIn)

    idxIn += 1
from keras.utils import to_categorical

import sklearn.preprocessing

L_enc = sklearn.preprocessing.LabelEncoder()

labels = L_enc.fit_transform(namesIn)
labels = np.array(labels).reshape((len(labels),1))
labels.shape
def Colorize():

    in_label = Input(shape=(1,))

    embed_input = Embedding(120, 1000)(in_label)

    

    

    #Encoder

    encoder_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1,))

    encoder_output = Conv2D(32, (3,3), activation='relu', padding='same',strides=1)(encoder_input)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(32, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(32, (3,3), activation='relu', padding='same',strides=1)(encoder_output)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same',strides=1)(encoder_output)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(encoder_output)

    

    #Fusion

    #fusion_output = RepeatVector(2 * 2)(embed_input) 

    fusion_output = Dense(4096, activation='relu')(embed_input) 

    fusion_output = Reshape(([8, 8, 64]))(fusion_output)

    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 

    fusion_output = Conv2D(64, (1, 1), activation='relu', padding='same')(fusion_output)

    

    #Decoder

    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(fusion_output)

    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(3, (3, 3), activation='tanh', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    return Model(inputs=[encoder_input, in_label], outputs=decoder_output)



model = Colorize()

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
X_train = imagesIn / 256.
datagen = ImageDataGenerator(

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=20,

        horizontal_flip=True)



# #Create embedding

# def create_inception_embedding(grayscaled_rgb):

#     def resize_gray(x):

#         return resize(x, (299, 299, 3), mode='constant')

#     grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])

#     grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)

#     with inception.graph.as_default():

#         embed = inception.predict(grayscaled_rgb_resized)

#     return embed



#Generate training data

def image_a_b_gen(dataset=X_train, batch_size = 20):

    for batch in datagen.flow(dataset, batch_size=batch_size):

        

        X_batch = rgb2gray(batch)

        

        #label_batch = batch

        #grayscaled_rgb = gray2rgb(X_batch)

        lab_batch = rgb2lab(batch)

        X_batch = lab_batch[:,:,:,0]

        X_batch = X_batch.reshape(X_batch.shape+(1,))

        Y_batch = lab_batch[:,:,:,1:] / 128

        yield [X_batch, label_batch], Y_batch

    



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5,

                                            min_lr=0.00001)

filepath = "Art_Colorization_Model.h5"

checkpoint = ModelCheckpoint(filepath,

                             save_best_only=True,

                             monitor='loss',

                             mode='min')



model_callbacks = [learning_rate_reduction,checkpoint]
X_batch = rgb2gray(X_train)

X_batch = X_batch.reshape(X_batch.shape+(1,))        

#label_batch = batch

#grayscaled_rgb = gray2rgb(X_batch)

# lab_batch = rgb2lab(batch)

# X_batch = lab_batch[:,:,:,0]

# X_batch = X_batch.reshape(X_batch.shape+(1,))

# Y_batch = lab_batch[:,:,:,1:] / 128

# [X_batch, label_batch], Y_batch
model.fit([X_batch,labels],X_train,

            epochs=6,

            verbose=1)
# BATCH_SIZE = 128

# model.fit_generator(image_a_b_gen(X_train,BATCH_SIZE),

#             epochs=2,

#             verbose=1,

#             steps_per_epoch=X_train.shape[0]/BATCH_SIZE,

#              callbacks=model_callbacks

#                    )
def get_rgb(img):

    color_me_embed = np.array(np.random.randint(120))#create_inception_embedding([np.asarray(img)])

    #color_me = np.array([img])

    img = img.reshape((64,64,1))/256

    #color_me = gray2rgb(img)

    #color_me = gray2rgb(np.asarray(img))

    #color_me = rgb2lab([color_me])[:,:,:,0]

    #color_me = color_me.reshape(color_me.shape+(1,))

    

    color_me_embed = color_me_embed.reshape(1,1)

    #print(color_me.shape)

    output = model.predict([[img], color_me_embed.reshape(1,1)])

    #output = output[0] * 256

    #print(output)

#     decoded_imgs = np.zeros((len(output),64, 64, 3))







#     cur = np.zeros((64, 64, 3))

#     cur[:,:,0] = color_me[0][:,:,0]

#     cur[:,:,1:] = output[0]

#    return lab2rgb(cur)

    output = output*256.

    return Image.fromarray( (output).astype('uint8').reshape((64,64,3)))
img = imagesIn2[101,:,:]

img2 = get_rgb(img)

#plt.imshow(Image.fromarray( (img2).astype('uint8').reshape((64,64,3))))

plt.imshow(img2)
plt.figure(figsize=(15,3))

for j in range(5):

    xx = np.zeros((10000))

    xx[np.random.randint(10000)] = 1

    plt.subplot(1,5,j+1)

    #img = generator.predict(xx.reshape((-1,10000)))[0].reshape((-1,64,64,3))

    img = generator.predict(xx.reshape((-1,10000)))[0]

    #img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))

    #img = Image.fromarray( (img).astype('uint8').reshape((128,128)))

    img = Image.fromarray( (img).astype('uint8').reshape((64,64)))

    plt.axis('off')

    plt.imshow(img, cmap='Greys')

    plt.imshow(get_rgb(img))

plt.show()  
class DogGenerator:

    index = 0   

    def getDog(self,seed):

        xx = np.zeros((10000))

        #xx[self.index] = 0.70

        #xx[np.random.randint(10000)] = 0.30

        xx = np.zeros((10000))

        xx[np.random.randint(10000)] = 1

        #img = generator.predict(xx.reshape((-1,10000)))[0].reshape((64,64,3))

        img = generator.predict(xx.reshape((-1,10000)))[0].reshape((64,64))

        self.index = (self.index+1)%10000

        return Image.fromarray( (img).astype('uint8').reshape((64,64)))

    #Image.fromarray( img.astype('uint8') ) 
# SAVE TO ZIP FILE NAMED IMAGES.ZIP

z = zipfile.PyZipFile('images.zip', mode='w')

d = DogGenerator()

for k in range(10000):

    img = d.getDog(np.random.normal(0,1,100))

    img = get_rgb(img)

    

    f = str(k)+'.png'

    #img.save(f,'PNG'); z.write(f); os.remove(f)

    cv2.imwrite(f, img); z.write(f); os.remove(f)

    #if k % 1000==0: print(k)

z.close()


img = d.getDog(np.random.normal(0,1,100))

img = get_rgb(img)

plt.imshow(img)