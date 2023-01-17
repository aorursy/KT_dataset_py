from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, BatchNormalization,DepthwiseConv2D,ZeroPadding2D,ReLU,GlobalAveragePooling2D, Dense

import numpy as np

import pandas as pd

import tensorflow.keras

from PIL import Image, ImageOps

from tensorflow.keras.layers import Concatenate,Input,Lambda

import tensorflow as tf

from tensorflow.keras import layers

import keras

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import initializers

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

import itertools

import os

import shutil

import random

import glob

import matplotlib.pyplot as plt

import warnings





import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score, auc, roc_auc_score, roc_curve

import sklearn

import scipy

import tensorflow as tf

from tqdm import tqdm

from keras.preprocessing import image

from keras.models import Model

from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense



%matplotlib inline





warnings.simplefilter(action='ignore', category=FutureWarning)

#no zero pad

#_URL = 'https://drive.google.com/file/d/1VZQe1rP0A7z4Xxa4omeyo89n7NgSaHXN/view?usp=sharing'
global batch_size

global epochs

global IMG_HEIGHT

global IMG_WIDTH

IMG_HEIGHT = 224

IMG_WIDTH = 224

np.random.seed(2019)

tf.random.set_seed(2019)

TEST_SIZE = 0.25

SEED = 2019

BATCH_SIZE = 8
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

train_df.head(7)
x_train = np.load('../input/four-fold-aptos/train_all_four.npy')

x_test = np.load('../input/four-fold-aptos/test_all_four.npy')
y_train = train_df['diagnosis'].values

y_train
y_train_one_hot = pd.get_dummies(train_df['diagnosis']).values



y_train_multi = np.empty(y_train_one_hot.shape, dtype=y_train_one_hot.dtype)

y_train_multi[:, 4] = y_train_one_hot[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train_one_hot[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train_one_hot.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
# y_tr = y_train

x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=TEST_SIZE, 

    random_state=SEED

)
# train_image_generator = ImageDataGenerator(

#                     rescale=1./255,

#                     rotation_range=45,

#                     width_shift_range=.15,

#                     height_shift_range=.15,

#                     horizontal_flip=True,

#                     zoom_range=0.5

#                     )

#  # Generator for our training data

# validation_image_generator =ImageDataGenerator(rescale=1./255)



#  # Generator for our validation data

# global train_data_gen

# train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

#                                                            directory=train_dir,

#                                                            shuffle=True,

#                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),

#                                                            class_mode='binary')

# global val_data_gen

# val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,

#                                                               directory=validation_dir,

#                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),

#                                                               class_mode='binary')
def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)
# class string1(tensorflow.keras.layers.Layer):

#     def __init__(self,filters1,filters2,chanDim=-1):

#         super(string1, self).__init__()

#         self.conv1s1=Conv2D(filters=filters1,kernel_size=(1,1),padding='same')

#         self.bn1s1=BatchNormalization(axis=3,momentum=0.999)

#         self.a1s1=ReLU(max_value=6,negative_slope=0,threshold=0)

#         self.dcv1s1=DepthwiseConv2D(kernel_size=(3,3),padding='same')

#         self.bn2s1=BatchNormalization(axis=3,momentum=0.999)

#         self.a2s1=ReLU(max_value=6,negative_slope=0,threshold=0)

#         self.conv2s1=Conv2D(filters=filters2,kernel_size=(1,1),padding='same')

#         self.bn3s1=BatchNormalization(axis=3,momentum=0.999)



#     def call(self,inputs):

#         x=self.conv1s1(inputs)

#         x=self.bn1s1(x)

#         x=self.a1s1(x)

#         x=self.dcv1s1(x)

#         x=self.bn2s1(x)

#         x=self.a2s1(x)

#         x=self.conv2s1(x)

#         x=self.bn3s1(x)

#         return x



# class string2(tensorflow.keras.layers.Layer):

#     def __init__(self,filters1,filters2,chanDim=-1):

#         super(string2, self).__init__()

#         self.conv1s2=Conv2D(filters=filters1,kernel_size=(1,1),padding='same')

#         self.bn1s2=BatchNormalization(axis=3,momentum=0.999)

#         self.a1s2=ReLU(max_value=6,negative_slope=0,threshold=0)

#         self.pads2=ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last')

#         self.dcv1s2=DepthwiseConv2D(kernel_size=(3,3),padding='same')

#         self.bn2s2=BatchNormalization(axis=3,momentum=0.999)

#         self.a2s2=ReLU(max_value=6,negative_slope=0,threshold=0)

#         self.conv2s2=Conv2D(filters=filters2,kernel_size=(1,1),padding='same')

#         self.bn3s2=BatchNormalization(axis=3,momentum=0.999)



#     def call(self,inputs):

#         x=self.conv1s2(inputs)

#         x=self.bn1s2(x)

#         x=self.a1s2(x)

#         x=self.pads2(x)

#         x=self.dcv1s2(x)

#         x=self.bn2s2(x)

#         x=self.a2s2(x)

#         x=self.conv2s2(x)

#         x=self.bn3s2(x)

#         return x

# #model=Sequential()



# input1=Input(shape=(224,224,3))

# zpd1=ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last')(input1)

# cv1=Conv2D(filters=16,kernel_size=(1,1),padding='same')(zpd1)

# bn1=BatchNormalization(axis=3,momentum=0.999)(cv1)

# a1=ReLU(max_value=6,negative_slope=0,threshold=0)(bn1)

# dcv1=DepthwiseConv2D(kernel_size=(3,3),padding='same')(a1)

# bn2=BatchNormalization(axis=3,momentum=0.999)(dcv1)

# a2=ReLU(max_value=6,negative_slope=0,threshold=0)(bn2)

# cv2=Conv2D(filters=8,kernel_size=(1,1),padding='same')(a2)

# bn3=BatchNormalization(axis=3,momentum=0.999)(cv2)

# cv3=Conv2D(filters=48,kernel_size=(1,1),padding='same')(bn3)

# bn4=BatchNormalization(axis=3,momentum=0.999)(cv3)

# a3=ReLU(max_value=6,negative_slope=0,threshold=0)(bn4)

# zpd2=ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last')(a3)

# dcv2=DepthwiseConv2D(kernel_size=(3,3),padding='same')(zpd2)

# bn5=BatchNormalization(axis=3,momentum=0.999)(dcv2)

# a4=ReLU(max_value=6,negative_slope=0,threshold=0)(bn5)

# cv4=Conv2D(filters=8,kernel_size=(1,1),padding='same')(a4)

# bn6=BatchNormalization(axis=3,momentum=0.999)(cv4)

# snz1=string1(48,8)(bn6)

# merger1=Concatenate(axis=-1)([bn6,snz1])

# sz1=string2(48,16)(merger1)

# snz2=string1(96,16)(sz1)

# merger2=Concatenate(axis=-1)([sz1,snz2])

# snz3=string1(96,16)(merger2)

# merger3=Concatenate(axis=-1)([merger2,snz3])

# sz2=string2(96,24)(merger3)

# snz4=string1(144,24)(sz2)

# merger4=Concatenate(axis=-1)([sz2,snz4])

# snz5=string1(144,24)(merger4)

# merger5=Concatenate(axis=-1)([merger4,snz5])

# snz6=string1(144,24)(merger5)

# merger6=Concatenate(axis=-1)([merger5,snz6])

# snz7=string1(144,32)(merger6)

# snz8=string1(192,32)(snz7)

# merger7=Concatenate(axis=-1)([snz7,snz8])

# snz9=string1(192,32)(merger7)

# merger8=Concatenate(axis=-1)([merger7,snz9])

# sz3=string2(192,56)(merger8)

# snz10=string1(336,56)(sz3)

# merger9=Concatenate(axis=-1)([sz3,snz10])

# snz11=string1(336,56)(merger9)

# merger10=Concatenate(axis=-1)([merger9,snz11])

# cv5=Conv2D(filters=336,kernel_size=(1,1),padding='same')(merger10)

# bn7=BatchNormalization(axis=3,momentum=0.999)(cv5)

# a5=ReLU(max_value=6,negative_slope=0,threshold=0)(bn7)

# dcv3=DepthwiseConv2D(kernel_size=(3,3),padding='same')(a5)

# bn8=BatchNormalization(axis=3,momentum=0.999)(dcv3)

# a6=ReLU(max_value=6,negative_slope=0,threshold=0)(bn8)

# cv7=Conv2D(filters=112,kernel_size=(1,1),padding='same')(a6)

# bn9=BatchNormalization(axis=3,momentum=0.999)(cv7)

# cv8=Conv2D(filters=1280,kernel_size=(1,1),padding='same')(bn9)

# bn10=BatchNormalization(axis=3,momentum=0.999)(cv8)

# a7=ReLU(max_value=6,negative_slope=0,threshold=0)(bn10)

# pool1=GlobalAveragePooling2D(data_format='channels_last')(a7)

# initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="normal", seed=None)

# D1=Dense(units=100,activation='relu',batch_input_shape=(None,1280),kernel_initializer=initializer)(pool1)

# D2=Dense(units=1,activation='softmax',use_bias='false',kernel_initializer=initializer)(D1)

# model = tensorflow.keras.Model(inputs=input1, outputs=D2)



# #keras.utils.plot_model(cmodel, "my_first_model.png")

# model.compile(optimizer=Adam(lr=0.1),loss=tf.keras.losses.CategoricalCrossentropy(),metrics='accuracy')
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    BatchNormalization(axis=3,momentum=0.999),

        ReLU(max_value=6,negative_slope=0,threshold=0),

        Dropout(0.2),

    DepthwiseConv2D(kernel_size=(3,3),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    Conv2D(filters=8,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    Conv2D(filters=48,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last'),

    DepthwiseConv2D(kernel_size=(3,3),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    Conv2D(filters=8,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    Conv2D(filters=192,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last'),

    DepthwiseConv2D(kernel_size=(3,3),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    Conv2D(filters=56,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    Conv2D(filters=192,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

      Conv2D(filters=1024,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    ZeroPadding2D(padding=([0,1],[0,1]),data_format='channels_last'),

    DepthwiseConv2D(kernel_size=(3,3),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Conv2D(filters=56,kernel_size=(1,1),padding='same'),

    BatchNormalization(axis=3,momentum=0.999),

    ReLU(max_value=6,negative_slope=0,threshold=0),

    Dropout(0.2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(5, activation='sigmoid')

])



# model.compile(optimizer=Adam(lr=0.0001),loss=tf.keras.losses.SquaredHinge(),metrics='accuracy')

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00005), metrics=['accuracy','AUC'])
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')

# Reducing the Learning Rate if result is not improving. 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',

                              verbose=1)



# kappa_metrics = Metrics()
# history = model.fit_generator(

#     train_data_gen,

#         epochs=2,

#     validation_data=val_data_gen,

    

# )



history = model.fit_generator(

    data_generator,

    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,

    epochs=60,

    validation_data=(x_val, y_val),

    callbacks=[early_stop, reduce_lr]

)
model.save('net.h5')
y_predict = model.predict(x_val)

y_predict
val_y = y_predict > 0.5

val_y = val_y.astype(int).sum(axis=1) - 1

val_y
y_real = [4 if (list(i)[4]==1) else list(i).index(0)-1 for i in y_val]




from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

  

actual = y_real

predicted = val_y

results = confusion_matrix(actual, predicted) 

  

print ('Confusion Matrix :')

print(results)

print ('Accuracy Score :',accuracy_score(actual, predicted) )

print ('Report : ')

print (classification_report(actual, predicted))
predict_probab = [y_predict[i][val_y[i]] for i in range(len(val_y))]
y_val_one_hot = []

for i in range(len(y_real)):

    y_val_one_hot.append(list(np.zeros(5, dtype = 'uint8')))

y_val_one_hot = np.array(y_val_one_hot)

for i in range(y_val_one_hot.shape[0]):

    y_val_one_hot[i][y_real[i]] = 1
def plot_roc(label):

    y_probab = y_predict[:, label]

    y_label = y_val_one_hot[:,label]

    fpr, tpr, thresholds = roc_curve(y_label, y_probab)

    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot([0,1],[0,1], 'k--')

    plt.plot(fpr,tpr, label = 'AUC SCORE : {:.3f}'.format(auc))

    plt.title('AUC ROC Curve of class '+str(label))

    plt.xlabel('False Positive rate')

    plt.ylabel('True Positive rate')

    plt.legend(loc = 'best')
plot_roc(0)
plot_roc(1)
plot_roc(2)
plot_roc(3)
plot_roc(4)
model.evaluate(x_val,y_val)
y_test_p = model.predict(x_test)
test_y = y_test_p > 0.5

test_y = test_y.astype(int).sum(axis=1) - 1

test_y
cohen_kappa_score(

            y_real,

            val_y, 

            weights='quadratic'

        )
hist = history.history
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["loss"], label="loss")

plt.plot(hist["val_loss"], label="val_loss")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("log_loss")

plt.legend();




plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["accuracy"], label="accuracy")

plt.plot(hist["val_accuracy"], label="val_acc")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("accuracy")

plt.legend();
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(hist["auc"], label="loss")

plt.plot(hist["val_auc"], label="val_loss")

# plt.plot(np.argmin(hist["val_loss"]), np.min(hist["val_loss"]), marker="x", color="r",

#          label="best model")

plt.xlabel("Epochs")

plt.ylabel("auc")

plt.legend();