import os,shutil

import glob

import numpy as np

import cv2

import tensorflow as tf

import keras

import matplotlib.pyplot as plt

from keras.models import *

from keras.layers import *

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing.image import *

from keras.callbacks import EarlyStopping

from keras import regularizers,optimizers

from keras.callbacks import LearningRateScheduler

from keras import *

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

#walk is 1, run is 0

original_dataset_dir = "../input/walk-or-run"



train_dir = os.path.join(original_dataset_dir,'walk_or_run_train/train')

#'../input/walk_or_run_train/train'

test_dir= os.path.join(original_dataset_dir,'walk_or_run_test/test')

#'../input/walk_or_run_test/test'

os.listdir(train_dir)
BATCH_SIZE=16

EPOCHS=50
LR_START = 0.00001

LR_MAX = 0.00005 

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = keras.callbacks.LearningRateScheduler(lrfn(30), verbose = True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    def eraser(input_img):

        img_h, img_w, img_c = input_img.shape

        p_1 = np.random.rand()



        if p_1 > p:

            return input_img



        while True:

            s = np.random.uniform(s_l, s_h) * img_h * img_w

            r = np.random.uniform(r_1, r_2)

            w = int(np.sqrt(s / r))

            h = int(np.sqrt(s * r))

            left = np.random.randint(0, img_w)

            top = np.random.randint(0, img_h)



            if left + w <= img_w and top + h <= img_h:

                break



        if pixel_level:

            c = np.random.uniform(v_l, v_h, (h, w, img_c))

        else:

            c = np.random.uniform(v_l, v_h)



        input_img[top:top + h, left:left + w, :] = c



        return input_img



    return eraser
train_datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,\

                             height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,\

                                 horizontal_flip=True,vertical_flip=False,validation_split=0.2,

                                  preprocessing_function = get_random_eraser(v_l=0, v_h=255))

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=16,

    class_mode='categorical',

    seed=2019,

    color_mode='rgb'

)

validation_generator = train_datagen.flow_from_directory(

    train_dir, # same directory as training data

    target_size=(224, 224),

    batch_size=16,

    class_mode='categorical',color_mode='rgb',

    subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(224, 224),

    batch_size=16,

    class_mode='categorical',color_mode='rgb')
import tensorflow as tf, tensorflow.keras.backend as K

from keras.applications import DenseNet201
strategy = tf.distribute.get_strategy()
def get_model():

    with strategy.scope():

        model = keras.Sequential([

                    DenseNet201(input_shape=(224,224, 3),include_top=False,weights='imagenet'),

                    keras.layers.GlobalAveragePooling2D(),

                    keras.layers.Dense(2, activation='softmax')

        ])

        model.compile(

                optimizer='adam',

                loss = 'categorical_crossentropy',

                metrics=['categorical_accuracy']

        )

    return model
D_net = get_model()
STEPS_PER_EPOCH = 600/BATCH_SIZE

history = D_net.fit_generator(train_generator,epochs = 30,

            callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience =3, 

                                                           min_lr=0.00001, verbose=1, mode='min'),early_stopping],

                              validation_data=test_generator)
res = D_net.evaluate_generator(test_generator)
print("test_loss:",res[0],"test acc:",res[1])