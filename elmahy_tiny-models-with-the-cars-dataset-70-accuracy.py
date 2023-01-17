# for working with files 

import glob

import os

import shutil

import itertools  

from tqdm import tqdm



# for working with images

from PIL import Image

import numpy as np

import pandas as pd

from skimage import transform

import matplotlib.pyplot as plt

import cv2 as cv

import scipy.io

import random



# tensorflow stuff

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization, GlobalAveragePooling2D, Add

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import regularizers, optimizers, Model

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.activations import relu, softmax







# for evaluation

from sklearn.metrics import classification_report, confusion_matrix
!mkdir car_data_cropped/

!mkdir car_data_cropped/train

!mkdir car_data_cropped/test
cars_annos = pd.read_csv('../input/anno_train.csv',header=None)

fnames = []

class_ids = []

bboxes = []

labels = []





for i in range(len(cars_annos)):

    annotation = cars_annos.iloc[i]

    bbox_x1 = annotation[1]

    bbox_y1 = annotation[2]

    bbox_x2 = annotation[3]

    bbox_y2 = annotation[4]

    class_id = annotation[5]

    labels.append('%04d' % (class_id,))

    fname = annotation[0]

    bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))

    class_ids.append(class_id)

    fnames.append(fname)





l = glob.glob('../input/car_data/car_data/train/*/*')



for j in tqdm(range(len(l))):

    i = fnames.index(l[j].split('/')[-1])

    labels[i]

    (x1, y1, x2, y2) = bboxes[i]

    fname=l[j].split('/')[-1]

    class_name = l[j].split('/')[-2]

    src_path = os.path.join('../input/car_data/car_data/train/'+class_name+'/', fname)

    src_image = cv.imread(src_path)



    height, width = src_image.shape[:2]



    # margins of 16 pixels

    margin = 16

    x1 = max(0, x1 - margin)

    y1 = max(0, y1 - margin)

    x2 = min(x2 + margin, width)

    y2 = min(y2 + margin, height)

    # print("{} -> {}".format(fname, label))





    dst_path = os.path.join('car_data_cropped/train/', class_name)

    if not os.path.exists(dst_path):

        os.makedirs(dst_path)





    dst_path = os.path.join(dst_path, fname)





    crop_image = src_image[y1:y2, x1:x2]

    #dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))

    cv.imwrite(dst_path, crop_image)
cars_annos = pd.read_csv('../input/anno_test.csv',header=None)

fnames = []

class_ids = []

bboxes = []

labels = []





l = glob.glob('../input/car_data/car_data/test/*/*')





for i in range(len(cars_annos)):

    annotation = cars_annos.iloc[i]

    bbox_x1 = annotation[1]

    bbox_y1 = annotation[2]

    bbox_x2 = annotation[3]

    bbox_y2 = annotation[4]

    class_id = annotation[5]

    labels.append('%04d' % (class_id,))

    fname = annotation[0]

    bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))

    class_ids.append(class_id)

    fnames.append(fname)

for j in tqdm(range(len(l))):

    i = fnames.index(l[j].split('/')[-1])



    (x1, y1, x2, y2) = bboxes[i]

    fname=l[j].split('/')[-1]



    class_name = l[j].split('/')[-2]

    src_path = os.path.join('../input/car_data/car_data/test/'+class_name+'/', fname)

    src_image = cv.imread(src_path)



    height, width = src_image.shape[:2]



    # margins of 16 pixels

    margin = 16

    x1 = max(0, x1 - margin)

    y1 = max(0, y1 - margin)

    x2 = min(x2 + margin, width)

    y2 = min(y2 + margin, height)

    # print("{} -> {}".format(fname, label))





    dst_path = os.path.join('car_data_cropped/test/', class_name)

    if not os.path.exists(dst_path):

        os.makedirs(dst_path)



    dst_path = os.path.join(dst_path, fname)





    crop_image = src_image[y1:y2, x1:x2]

    #dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))



    cv.imwrite(dst_path, crop_image)
attempt = 1

if attempt == 0:

    train_datagen=ImageDataGenerator(rotation_range=20,

                                     width_shift_range=0.1,

                                     height_shift_range=0.1,

                                     zoom_range=0.2,

                                     horizontal_flip=True)





    valid_datagen=ImageDataGenerator(rotation_range=20,

                                    zoom_range=0.15,

                                    horizontal_flip=True)





    train_generator=train_datagen.flow_from_directory(

        directory="car_data_cropped/train/",

        batch_size=64,

        seed=42,

        target_size=(224,224))





    valid_generator=valid_datagen.flow_from_directory(

        directory="car_data_cropped/test/",

        batch_size=64,

        seed=42,

        target_size=(224,224))



#--------------------------------------------------------------------------

if attempt == 1:

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



    train_datagen=ImageDataGenerator(rotation_range=15,

                                     width_shift_range=0.1,

                                     height_shift_range=0.1,

                                     zoom_range=0.2,

                                     horizontal_flip=True,

                                     preprocessing_function=preprocess_input)



    valid_datagen=ImageDataGenerator(horizontal_flip=True, 

                                     preprocessing_function=preprocess_input)





    train_generator=train_datagen.flow_from_directory(

        directory="car_data_cropped/train/",

        batch_size=64,

        seed=42,

        target_size=(224,224))





    valid_generator=valid_datagen.flow_from_directory(

        directory="car_data_cropped/test/",

        batch_size=300,

        seed=42,

        target_size=(224,224))

def block(n_output, upscale=False):

    # n_output: number of feature maps in the block

    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    

    # keras functional api: return the function of type

    # Tensor -> Tensor

    

    def f(x):

        

        # H_l(x):

        # first pre-activation

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # first convolution

        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        

        # second pre-activation

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # second convolution

        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        

        # f(x):

        if upscale:

            # 1x1 conv2d

            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)

        else:

            # identity

            f = x

        

        # F_l(x) = f(x) + H_l(x):

        return Add()([f, h])

    

    return f
# input tensor is the 28x28 grayscale image

input_tensor = Input((224, 224, 3))



# first conv2d with post-activation to transform the input data to some reasonable form

x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)

x = BatchNormalization()(x)

x = Activation(relu)(x)



# F_1

x = block(16)(x)

# F_2

x = block(16)(x)



# F_3

# H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32

# and we can't add together tensors of inconsistent sizes, so we use upscale=True

x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation

# F_4

x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

# F_5

x = block(32)(x)                     # !!! <------- Uncomment for local evaluation



# F_6

x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation

# F_7

x = block(48)(x)                     # !!! <------- Uncomment for local evaluation



# last activation of the entire network's output

x = BatchNormalization()(x)

x = Activation(relu)(x)



# average pooling across the channels

# 28x28x48 -> 1x48

x = GlobalAveragePooling2D()(x)



# dropout for more robust learning

x = Dropout(0.2)(x)



# last softmax layer

x = Dense(units=196, kernel_regularizer=regularizers.l2(0.01))(x)

x = Activation(softmax)(x)



model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
len(model.layers)
model.summary()
for x,y in valid_generator:

    x_val = x

    y_val = y

    break;
IMAGE_SIZE = 224

# Base model with MobileNetV2

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)



base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,alpha = .5,

                                               include_top=False, 

                                               weights='imagenet')



x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = Dropout(.6)(x)

prediction_layer = tf.keras.layers.Dense(196, activation='softmax')(x)



learning_rate = 0.0001



model=Model(inputs=base_model.input,outputs=prediction_layer)



for layer in model.layers[:80]:

    layer.trainable=False

for layer in model.layers[80:]:

    layer.trainable=True

# 



optimizer=tf.keras.optimizers.Adam(lr=learning_rate,clipnorm=0.001)

def get_lr_metric(optimizer):

    def lr(y_true, y_pred):

        return optimizer.lr

    return lr



lr_metric = get_lr_metric(optimizer)



model.compile(optimizer=optimizer,

             loss='categorical_crossentropy',

             metrics=['accuracy',lr_metric])



reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=1, verbose=1)



model.fit(train_generator,

          steps_per_epoch=100,

          validation_data=(x_val,y_val),

          epochs=40,verbose=1)



scoreSeg = model.evaluate_generator(valid_generator)

print("Accuracy = ",scoreSeg[1])



for i,j in valid_generator:

    print(i.shape, j.shape)

    p = model.predict(i)

    p = p.argmax(-1)

    t = j.argmax(-1)

    print(classification_report(t,p))

    print(confusion_matrix(t,p))

    break;

p
{i:j for j,i in valid_generator.class_indices.items()}
t
tf.keras.models.save_model(

    model,

    "cars1_half.h5"

)

1 /196
ls -la
from IPython.display import FileLink

FileLink('cars1_half2.h5')