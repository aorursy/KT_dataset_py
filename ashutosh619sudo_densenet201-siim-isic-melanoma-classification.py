# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input director



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
train.head()
test.head()
import seaborn as sn
import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

sn.countplot(train["benign_malignant"])
train["benign_malignant"].value_counts()
train["diagnosis"].value_counts()
plt.figure(figsize=(15,9))

sn.countplot(train["diagnosis"])
plt.figure(figsize=(10,6))

sn.countplot(train["anatom_site_general_challenge"])
train.head()
plt.figure(figsize=(15,9))

malignant_body_part = train[train["benign_malignant"]=="malignant"]["anatom_site_general_challenge"]

sn.countplot(malignant_body_part)
plt.figure(figsize=(15,9))

sn.boxplot(x="benign_malignant",y="age_approx",data=train)
plt.figure(figsize=(10,6))

sn.distplot(train["age_approx"])
plt.figure(figsize=(15,9))

sn.boxplot(x="anatom_site_general_challenge",y="age_approx",data=train)
train.head()
sn.countplot(x = "benign_malignant",hue="sex",data=train)
import pydicom

from pydicom import dcmread
ex1 = "/kaggle/input/siim-isic-melanoma-classification/train/ISIC_6692344.dcm"

ex2 = "/kaggle/input/siim-isic-melanoma-classification/train/ISIC_6652710.dcm"

ex1_img = dcmread(ex1)

ex2_img = dcmread(ex2)
fig,(ax1,ax2) = plt.subplots(1,2)

ax1.imshow(ex1_img.pixel_array,cmap=plt.cm.bone)

ax2.imshow(ex2_img.pixel_array,cmap=plt.cm.bone)
import os

def show_images(n = 5, rows=1, cols=5, title="Skin Cancer"):

    plt.figure(figsize=(16,4))



    for k, path in enumerate(list(os.listdir('../input/siim-isic-melanoma-classification/train'))[:n]):

        image = pydicom.read_file('../input/siim-isic-melanoma-classification/train/'+path)

        image = image.pixel_array



        plt.suptitle(title, fontsize = 16)

        plt.subplot(rows, cols, k+1)

        plt.imshow(image)

        plt.axis('off')
show_images(n=10, rows=2, cols=5)
import matplotlib.image as mpimg

def show_images_jpeg(n = 5, rows=1, cols=5, title="Skin Cancer"):

    plt.figure(figsize=(16,4))



    for k, path in enumerate(list(os.listdir('../input/siim-isic-melanoma-classification/jpeg/train'))[:n]):

        image = mpimg.imread('../input/siim-isic-melanoma-classification/jpeg/train/'+path)

        plt.suptitle(title, fontsize = 16)

        plt.subplot(rows, cols, k+1)

        plt.imshow(image)

        plt.axis('off')
show_images_jpeg(n=10, rows=2, cols=5)
path = '../input/siim-isic-melanoma-classification/jpeg/train/'

train["path"] = path+train["image_name"]+'.jpg'
from sklearn.model_selection import train_test_split
X_train,X_val = train_test_split(train,test_size=0.2)
X_train.shape,X_val.shape
X_train["target"] = X_train["target"].astype(str)
from keras.models import Sequential, Model,load_model

from keras.applications.vgg16 import VGG16,preprocess_input

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation

from keras.layers import GlobalMaxPooling2D

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc

import skimage.io

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras import backend as K
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    X_train,

    x_col='path',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    shuffle=True,

    class_mode='raw')



validation_generator = val_datagen.flow_from_dataframe(

    X_val,

    x_col='path',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode='raw')
from tensorflow.keras.applications import DenseNet121
dense_model = DenseNet121(include_top=False,

    weights="imagenet",

    input_shape=(224,224,3))
model = Sequential()

model.add(dense_model)

model.add(GlobalMaxPooling2D())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.metrics import AUC
def focal_loss(alpha=0.25,gamma=2.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        

        alpha_factor = 1

        modulating_factor = 1



        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))

        modulating_factor = K.pow((1-p_t), gamma)



        # compute the final loss and return

        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)

    return focal_crossentropy
opt = Adam(lr=1e-5)

model.compile(loss=focal_loss(), metrics=[AUC()],optimizer=opt)
nb_epochs = 2

batch_size=8

nb_train_steps = X_train.shape[0]//batch_size

nb_val_steps=X_val.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))