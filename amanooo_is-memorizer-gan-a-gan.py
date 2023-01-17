import os

import sys

from tqdm import tqdm, tqdm_notebook

import glob

import shutil



import numpy as np

import pandas as pd

import random



import matplotlib.pyplot as plt

import cv2



import xml.etree.ElementTree as ET



from keras.models import Model

from keras.layers import Input, Dense, Conv2D, Reshape, Flatten

from keras.layers import concatenate, UpSampling2D

from keras.preprocessing.image import image, load_img, ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD, Adam



print(os.listdir("../input"))
img_size = 64

channels = 3

img_shape = (img_size, img_size, channels)



dim = img_size * img_size * channels     #
DIR = os.getcwd()

DIRimg = "../input/all-dogs/all-dogs"

DIRanno = "../input/annotation/Annotation"

DIRout = "../output_images"
def loadImage(fPath, resize = True):

    img = cv2.imread(fPath)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # BGR to RGB

    ok = True

    if resize:

        xmin,ymin,xmax,ymax = clipImage(fPath)        # clip to square

        if xmin >= 0:                                 # exist Annotation

            img = img[ymin:ymax, xmin:xmax, :]        # [h,w,c]

            # Interpolation method

            if xmax - xmin > img_size:

                interpolation = cv2.INTER_AREA            # shrink

            else:

                interpolation = cv2.INTER_CUBIC           # expantion

            img = cv2.resize(img, (img_size, img_size),

                        interpolation = interpolation)    # resize

        else:

            ok = False

    return ok, img
def clipImage(fPath):

    imgName = os.path.basename(fPath)[:-4].split("_")

    breed = imgName[0]

    dog = imgName[1]

    path = glob.glob(os.path.join(DIRanno, breed + "*", breed +"_" + dog))

    if len(path) > 0:

        tree = ET.parse(path[0])

        root = tree.getroot()    # get <annotation>

        size = root.find('size')

        width = int(size.find('width').text)

        height = int(size.find('height').text)

#        objects = root.findall('object')      # ToDo: correspond multi objects

#        for object in objects:

        object = root.find('object')

        bndbox = object.find('bndbox') 

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)



        xmin = max(0, xmin - 4)        # 4 : margin

        xmax = min(width, xmax + 4)

        ymin = max(0, ymin - 4)

        ymax = min(height, ymax + 4)



        w = max(xmax - xmin, ymax - ymin, img_size)   # ideal w

        

        if w > min(width, height):

            xmin = -1; ymin = -1; xmax = -1; ymax = -1;

        else:

            w = min(w, width, height)                     # available w

    

            if w > xmax - xmin:

                xmin = min(max(0, xmin - int((w - (xmax - xmin))/2)), width - w)

                xmax = xmin + w

            if w > ymax - ymin:

                ymin = min(max(0, ymin - int((w - (ymax - ymin))/2)), height - w)

                ymax = ymin + w



    else:

        xmin = -1; ymin = -1; xmax = -1; ymax = -1;       # nothing Annotation

        

    return xmin,ymin,xmax,ymax
all_fNames = os.listdir(DIRimg)



# train data

x_train = np.zeros((len(all_fNames),img_size,img_size,3))

j = 0

for i in tqdm(range(len(all_fNames))):

    path = os.path.join(DIRimg, all_fNames[i])

#    x_train[i] = loadImage(path)

    ok, img = loadImage(path)

    if ok:

        x_train[j] = img

        j += 1



print(j)

x_train = x_train[:j] / 255.
input = Input((10000,))

x = Dense(2048, activation='elu')(input)

x = Reshape((8,8,32))(x)

x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)



# COMPILE

decoder = Model(input, decoded)

decoder.compile(optimizer=Adam(lr=0.005), loss='binary_crossentropy')



# DISPLAY ARCHITECTURE

decoder.summary()
# TRAINING DATA

ids = np.random.randint(0,len(x_train),10000)

#train_y = x_train[ids, :,:,:].reshape((-1,dim))

train_y = x_train[ids, :,:,:]

train_X = np.eye(10000)
# TRAIN NETWORK

lr = 0.01

batch = 256; ep = 30; it = 10

d_loss = []



for k in range(10):

    annealer = LearningRateScheduler(lambda x: lr)

    h = decoder.fit(train_X, train_y, epochs = ep, batch_size = batch,

                    callbacks=[annealer], verbose=0)

    d_loss.extend(h.history['loss'])

    print('Epoch',(k+1)*ep,'/',ep*it,'  loss =',h.history['loss'][-1], '/ lr =',lr)

    if h.history['loss'][-1] / h.history['loss'][0] > 0.99: lr = max(lr/2., 0.0005)



plt.plot(d_loss)

plt.show()
del x_train, train_y, train_X
def getDog(ids, mix_rate):

    imgs = []

    for id in ids:

        xx = np.zeros((10000))

        xx[id] = mix_rate

        xx[np.random.randint(10000)] = 1.0 - mix_rate

        imgs.append(decoder.predict(xx.reshape((-1,10000)))[0].reshape(img_shape))

    return imgs
def sumple_images(imgs, rows=3, cols=5, figsize=(12,10)):

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    for indx, axis in enumerate(axes.flatten()):

        img = image.array_to_img(imgs[indx])    # ndarray → PIL

        imgplot = axis.imshow(img)

#        axis.set_title(all_fNames[sample_ids[indx]])

        axis.set_axis_off()

    plt.tight_layout()
ids = np.random.randint(0,10000, 35)

g_imgs = getDog(ids, 0.99)

sumple_images(g_imgs, rows=5, cols=7, figsize=(12,8))
if os.path.exists(DIRout):

    shutil.rmtree(DIRout)

if not os.path.exists(DIRout):

    os.mkdir(DIRout)
batch = 64

e = batch

id = list(range(10000))



for s in tqdm(range(0, 10000, batch)):

    g_imgs = getDog(id[s:e], 0.99)

    for j in range(batch):

        img = image.array_to_img(g_imgs[j])    # ndarray → PIL

        img.save(os.path.join(DIRout, 'image_' + str(s+j+1).zfill(5) + '.png'))

        if s+j+1 == 10000:

            break

    e += batch

    

print(len(os.listdir(DIRout)))
shutil.make_archive('images', 'zip', DIRout)