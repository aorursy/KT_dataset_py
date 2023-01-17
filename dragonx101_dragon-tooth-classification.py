# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running texithis (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

category_id = [1000017037,

1000017053,

1000017055,

1000017057,

1000017059,

1000017181,

1000017061,

1000017183,

1000017063,

1000017065,

1000017067,

1000017069,

1000017071,

1000017075,

1000017077]

# 1000017081,1000017083,1000017093,1000017097,1000017101,1000017107,1000017111,1000017133,1000017143,1000017173
import io

import bson                       # this is installed with the pymongo package

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data

from multiprocessing import cpu_count

import concurrent.futures as cf 

from tqdm import tqdm_notebook as ttnote

import cv2

num_processes     =    16

im_adw            =    180

im_adh            =    180

#pool             =     mp.Pool(processes = num_processes)

num_cpus          =     cpu_count()

print(num_cpus)
# multiprocess version to count the number of images in this 25 categories

def parallel_count(filepath, num_images, category_id):

    bar = ttnote(total = num_images)

    with open(filepath, 'rb') as f, cf.ThreadPoolExecutor(num_cpus) as executor:

        data = bson.decode_file_iter(f)

        delayed_load = []

        i = 0

        try:

            for c, d in enumerate(data):

                target = d['category_id']

                if target in category_id:

                    for e, pic in enumerate(d['imgs']):

                        i = i + 1

                        if i >= num_images:

                            print('too many images found')

                            raise IndexError



        except IndexError:

            pass;

        print('the number of images to train is %d'%i)

 

    return i



img_cnt = parallel_count('../input/train.bson', 100000, category_id)

#X_test, Y_test = parallel_read('../input/test.bson', num_images_test, im_adw, im_adh)



                
# multiprocess version

# a little bug is it applies operation block by block, and have multiple variables

# I guess we could accelerate it by vecterization

def imread(buf):

    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)



def img2adjust(im):

    x = cv2.resize(im, (im_adw, im_adh), interpolation = cv2.INTER_AREA)

    return np.float32(x)/255



def load_image(pic, target, bar):

    picture = imread(pic)

    x = img2adjust(picture)

    bar.update() # seems that we are using the function from

    return x, target

    

def parallel_read(filepath, num_images, im_adw, im_adh):

    X = np.empty((num_images, im_adw, im_adh, 3), dtype = np.float32)

    Y = []

    bar = ttnote(total = num_images)

    with open(filepath, 'rb') as f, cf.ThreadPoolExecutor(num_cpus) as executor:

        data = bson.decode_file_iter(f)

        delayed_load = []

        i = 0

        try:

            for c, d in enumerate(data):

                target = d['category_id']

                if target in category_id:

                    for e, pic in enumerate(d['imgs']):

                        delayed_load.append(executor.submit(load_image, pic['picture'],target, bar))

                        i = i + 1

                        if i >= num_images:

                            raise IndexError



        except IndexError:

            print('the maximium is the best we could do')

            pass;

 

        for i, item in enumerate(cf.as_completed(delayed_load)):

            x, target = item.result()

            X[i] = x

            Y.append(target)

    return X, Y



num_images_train  =    40000

num_images_test   =    round(num_images_train/10)

X_train = np.empty((num_images_train, im_adw, im_adh, 3), dtype = np.float32)

Y_train = []

X_test = np.empty((num_images_test, im_adw, im_adh, 3), dtype = np.float32)

Y_test = []

X_train, Y_train =     parallel_read('../input/train.bson', 10000, im_adw, im_adh)

#X_test, Y_test = parallel_read('../input/test.bson', num_images_test, im_adw, im_adh)

        

                
from IPython.display import HTML, Image

print('X_train shape', X_train.shape)

print(len(Y_train))
rows = 16

cols = 6

fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))

fig.suptitle('Image from Each Product', fontsize=20)

for i in range(rows):

    for j in range(cols):

        product_image = X_train[i*cols + j,:,:,:]

        product_cater = Y_train[i*cols + j]

        ax[i][j].imshow(product_image)

        ec = (0, .6, .1)

        fc = (0, .7, .2)

        ax[i][j].text(0, -20, product_cater, size=10, rotation=0,

                ha="left", va="top", 

                bbox=dict(boxstyle="round", ec=ec, fc=fc))

plt.setp(ax, xticks=[], yticks=[])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
from ipywidgets import interact, interactive, fixed

import ipywidgets as widgets



@interact(n = (0, len(X_train)))

def show_pic(n):

    plt.imshow(X_train[n])

    print('Category :', Y_train[n])
from keras.layers import Flatten, Dense, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D

from keras.layers import MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras.models import Model, Sequential

from keras.optimizers import SGD, Adam

from keras.activations import softmax

from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.preprocessing import image

#keras,tensorflow, pytorch, 



#second part

from keras.models import load_model

import keras.backend as K

from keras.metrics import categorical_crossentropy, top_k_categorical_accuracy

import tensorflow as tf

# from multiGPU import MultiGPUModel



num_class = 500

#Y_train_cat = to_categorical(str(Y_train[:]),num_classes = num_class)

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

import math



# encoder the label 

encoder = LabelEncoder()

encoder.fit(Y_train)

encoded_Y = encoder.transform(Y_train)

# convert integers to dummy variables (i.e. one hot encoded)

Y_train_cat = np_utils.to_categorical(encoded_Y, num_classes = 1000)

print(Y_train_cat[:1])

modelname = 'InceptionV5'

savemodel = './models/' + modelname

# Generate batches of tensor image data with real-time data augmentation, looped over

TrainDatagen = ImageDataGenerator(

    rescale           = 1,

    zoom_range        = 1,

    width_shift_range = 0.1,# range of random horizontal shift 

    height_shift_range= 0.1,

    horizontal_flip   = True,

    data_format=K.image_data_format() # channels last default

)

# batches of augmented/normalized data

train_generator = TrainDatagen.flow(

    X_train,

    Y_train_cat,

    batch_size = 32,

    seed = 11

)
print(X_train.shape)

print(len(Y_train))

print(len(Y_train_cat))

print(Y_train[1])

print(Y_train_cat[100])
# Choice 1: DIY model;

# Choice 2: reuse model;

from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.layers import Input

model = Sequential()

model.add(Conv2D(16, 3, activation='relu', padding = 'same', input_shape=(140, 140, 3)))

model.add(Conv2D(16, 3, activation='relu', padding = 'same'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.25))

model.add(Conv2D(32, 3, activation='relu', padding = 'same'))

model.add(Conv2D(32, 3, activation='relu', padding = 'same'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.2))

model.add(Flatten())

# 

model.add(Dense(num_class*2, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_class*2, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_class*2, activation='softmax'))





optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optim, 

              loss  = 'categorical_crossentropy',

              metrics = ['accuracy'])

model.fit(X_train, Y_train_cat,

          epochs=10,

          batch_size=128)

# base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=None, input_shape=(im_adw, im_adh, 3))

# base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

model.save_weights('model.h5')

model.save_weights('model.h5')

score = model.evaluate(x_test, y_test, batch_size=28)

print(score)