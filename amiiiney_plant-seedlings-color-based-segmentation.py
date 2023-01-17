# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%env JOBLIB_TEMP_FOLDER=/tmp



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

from matplotlib import colors

import seaborn as sns

sns.set_style('whitegrid')



from PIL import Image

from imageio import imread

import imageio

import skimage

import skimage.io

import skimage.transform

from imageio import imread





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score



from skimage.morphology import closing, disk, opening

import random

import time

import copy

from tqdm import tqdm_notebook as tqdm

from os import listdir

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau



from keras import backend as K

from keras.models import Sequential

from keras.models import Model

from keras.layers import Activation

from keras.layers.core import Dense, Flatten

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dropout

from keras.layers.convolutional import *

from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input

from keras.applications.inception_v3 import decode_predictions

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from keras.models import model_from_json

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import optimizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import np_utils

from keras.optimizers import SGD



from IPython.core.display import display, HTML

from PIL import Image

from io import BytesIO

import base64



from os import listdir

from skimage.segmentation import mark_boundaries



import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Directory='../input/v2-plant-seedlings-dataset/nonsegmentedv2/'

subfolders = listdir(Directory)

print(os.listdir('../input/v2-plant-seedlings-dataset/nonsegmentedv2/'))
sfc="Small-flowered Cranesbill"

plt.figure(figsize=(15,12))



for n in range(12):

        folder=subfolders[n]

        plt.subplot(3,4,n+1)

        files = listdir(Directory + folder + "/") 

        image=cv2.imread(Directory + folder + "/" + files[n+221])

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.axis("off")

        plt.title(folder, fontsize=14, weight='bold')
p1 = cv2.imread('/kaggle/input/v2-plant-seedlings-dataset/nonsegmentedv2/Small-flowered Cranesbill/135.png')

p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2RGB)

plt.imshow(p1)

plt.grid(False)

plt.show()
def set_size(w,h, ax=None):

    """ w, h: width, height in inches """

    if not ax: ax=plt.gca()

    l = ax.figure.subplotpars.left

    r = ax.figure.subplotpars.right

    t = ax.figure.subplotpars.top

    b = ax.figure.subplotpars.bottom

    figw = float(w)/(r-l)

    figh = float(h)/(t-b)

    ax.figure.set_size_inches(figw, figh)

    

pixel_colors = p1.reshape((np.shape(p1)[0]*np.shape(p1)[1], 3))

norm = colors.Normalize(vmin=-1.,vmax=1.)

norm.autoscale(pixel_colors)

pixel_colors = norm(pixel_colors).tolist()
r, g, b = cv2.split(p1)

fig = plt.figure()

axis = fig.add_subplot(1, 1, 1, projection="3d")

set_size(6,6)

pixel_colors = p1.reshape((np.shape(p1)[0]*np.shape(p1)[1], 3))

norm = colors.Normalize(vmin=-1.,vmax=1.)

norm.autoscale(pixel_colors)

pixel_colors = norm(pixel_colors).tolist()



axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")

axis.set_xlabel("Red", weight='bold')

axis.set_ylabel("Green", weight='bold')

axis.set_zlabel("Blue", weight='bold')



plt.show()
hsv_p1 = cv2.cvtColor(p1, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_p1)

fig = plt.figure()

axis = fig.add_subplot(1, 1, 1, projection="3d")





axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")

axis.set_xlabel("Hue",weight='bold')

axis.set_ylabel("Saturation", weight='bold')

axis.set_zlabel("Value", weight='bold')

set_size(6,6)

plt.show()
def plot_mask(image, colormin, colormax):

        hsv_p1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)   

        mask = cv2.inRange(hsv_p1, colormin , colormax)

        result = cv2.bitwise_and(image, image, mask=mask)

        plt.figure(figsize=(15,10))

        plt.subplot(1, 3, 1)

        plt.imshow(image)

        plt.grid(False)

        plt.subplot(1, 3, 2)

        plt.imshow(mask, cmap="gray")

        plt.grid(False)

        plt.subplot(1, 3, 3)

        plt.imshow(result)

        plt.grid(False)

        return plt.show()
colormin=(36, 25, 25)

colormax=(70, 255,255)



plot_mask(p1, colormin, colormax)
new_colormin=(25,50,50)

new_colormax=(80,255,255)

plot_mask(p1, new_colormin, new_colormax)
def segmented(image):

    foto = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_foto = cv2.cvtColor(foto, cv2.COLOR_RGB2HSV)

    #print("hsvh",hsv_foto.dtype)

    colormin=(25,50,50)

    colormax=(86,255,255)



    mask = cv2.inRange(hsv_foto, colormin , colormax)

    #print("mask",mask.dtype)

    result = cv2.bitwise_and(foto, foto, mask=mask)

    #print("result",result.dtype)

    pil_image= Image.fromarray(result)





    return result
plt.figure(figsize=(15,10))



for n in range(12):

        folder = subfolders[n]

        plt.subplot(3,4,n+1)

        files = listdir(Directory + folder + "/") 

        image=cv2.imread(Directory + folder + "/" + files[n+221])

        plt.imshow(segmented(image))

        plt.axis("off")

        plt.title(folder, weight='bold', fontsize=14)

def segmented2(image):

    image=np.array(image)

    #foto=image.copy().astype(np.uint8)

    #foto = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_foto = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    colormin=(25,50,50)

    colormax=(90,255,255)

    mask = cv2.inRange(hsv_foto, colormin , colormax)

    

    #result = cv2.bitwise_and(foto, foto, mask=mask)

    result = cv2.bitwise_and(image, image, mask=mask)

    result2=np.array(result)

    #pil_image= Image.fromarray(result, mode='RGB')

    #pil_image= Image.fromarray(result)





    return result2
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.1,

        horizontal_flip=True,

        validation_split=0.2,

        #preprocessing_function = segmented2

                     )





train_generator = train_datagen.flow_from_directory(

    '../input/v2-plant-seedlings-dataset/nonsegmentedv2',

        target_size=(64,64),

        batch_size=32,

        class_mode='categorical',

        subset='training')



validation_generator = train_datagen.flow_from_directory(

        '../input/v2-plant-seedlings-dataset/nonsegmentedv2',

        target_size=(64, 64),

        batch_size=32,

        class_mode='categorical',

        subset='validation')
#x,y = train_generator.next()

#for i in range(0,2):

#    image = x[i]

#    plt.imshow(image)

#    plt.show()
#weights_incv3 = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



# load pre-trained weights and add global average pooling layer

#base_model_incv3 = InceptionV3(weights=weights_incv3, input_shape=(150,150,3), include_top=False, pooling='avg')



# freeze convolutional layers

#for layer in base_model_incv3.layers:

#    layer.trainable = False



#define classification layers

#x = Dense(1024, activation='relu')(base_model_incv3.output)

#predictions = Dense(1, activation='sigmoid')(x)

#x = Dense(256, activation='relu')(base_model_incv3.output)

#x = Dropout(0.5)(x)

#predictions = Dense(12, activation='softmax')(x)



#model = Model(inputs=base_model_incv3.input, outputs=predictions)

#print(model.summary())
from keras.applications.vgg16 import VGG16

from keras.layers import Dropout

from keras.models import Sequential

from keras.layers import Dense, Flatten



vgg_conv = VGG16(weights=None, include_top=False, input_shape=(64, 64, 3))

vgg_conv.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')



for layer in vgg_conv.layers[:-4]:

    layer.trainable = False



model = Sequential()

model.add(vgg_conv)

 

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(12, activation='softmax'))

 

model.summary()
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', 

              metrics=['accuracy'])
# We'll stop training if no improvement after some epochs

earlystopper1 = EarlyStopping(monitor='loss', patience=10, verbose=1)



# Save the best model during the traning

checkpointer1 = ModelCheckpoint('best_model1.hdf5'

                                ,monitor='val_accuracy'

                                ,verbose=1

                                ,save_best_only=True

                                ,save_weights_only=True)
history = model.fit_generator(train_generator, steps_per_epoch=800, 

                    validation_data=validation_generator,

                    validation_steps=128,

                    epochs=15, verbose=1,

                   callbacks=[checkpointer1])
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,10))

#First Model

ax1 = plt.subplot2grid((2,2),(0,0))

train_loss = history.history['loss']

test_loss = history.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model 1: Loss vs. Epoch',weight='bold', fontsize=18)

ax1 = plt.subplot2grid((2,2),(0,1))

train_acc = history.history['accuracy']

test_acc = history.history['val_accuracy']

x = list(range(1, len(test_acc) + 1))

plt.plot(x, test_acc, color = 'cyan', label = 'Test accuracy')

plt.plot(x, train_acc, label = 'Training accuracy')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Model 1: Accuracy vs. Epoch', weight='bold', fontsize=18)

plt.show()