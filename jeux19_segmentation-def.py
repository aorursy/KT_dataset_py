# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print("Setup finished")



# Any results you write to the current directory are saved as output.
import os

import random

import numpy as np

from lxml import etree

from skimage import io

from skimage.transform import resize



# parameters that you should set before running this script

filter = ['aeroplane', 'boat', 'bus', 'motorbike']

# select class, this default should yield 1489 training and 1470 validation images

aux_folder= os.path.join("../input", os.listdir("../input")[0])

voc_root_folder=os.path.join(aux_folder, os.listdir(aux_folder)[0])



image_size = 128    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)





# step1 - build list of filtered filenames

annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")

annotation_files = os.listdir(annotation_folder)

filtered_filenames = []



# for a_f in annotation_files[:1500]:

for a_f in annotation_files:

    tree = etree.parse(os.path.join(annotation_folder, a_f))

    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):

        filtered_filenames.append(a_f)

print("Finished filtered images")







# step3 - build segmentation dataset

annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")



filtered_segmentation=[]



for a_f in filtered_filenames:

    tree = etree.parse(os.path.join(annotation_folder, a_f))

    for tag in tree.iterfind(".//segmented"):

        if tag.text=="1" :

            filtered_segmentation.append(a_f[:-4])

print("Finished filtered images")

import cv2

# step3 - build (x,y) for TRAIN/VAL (segmentation)

segmentation_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/")

segmentation_files = os.listdir(segmentation_folder)



segmentation_train=os.path.join(segmentation_folder,'train.txt')

segmentation_val=os.path.join(segmentation_folder,'val.txt')









def build_segmentation_dataset(seg_file,filtered_segmentation):

    """ build training or validation set



    :param list_of_files: list of filenames to build trainset with

    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 1) and  y np.ndarray of shape 

    (n_images, image_size, image_size, 1) with segmented masks

    """



    with open(seg_file) as file:

        lines = file.read().splitlines()

      

    train_filter = [item for item in lines if (item in filtered_segmentation)]



#     "Real Images"

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")

    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if f in file]

    

#     Segmented Images

    seg_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass/")

    seg_filenames = [os.path.join(seg_folder, file) for f in train_filter for file in os.listdir(seg_folder) if f in file]

    

    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype('float32')

#     x=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aux_x]

    

    aux_y = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in seg_filenames]).astype('float32')           

    aux_y_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in aux_y])



    mask_val = [0.294, 0.418, 0.572, 0.596]

    masks = []

    for elm in aux_y_gray:

        mask = np.zeros((128,128))

        for mask_val_int in mask_val:

            mask[np.array(elm>(mask_val_int-0.005)) * np.array(elm<(mask_val_int+0.005))] = 1

        masks.append(mask.reshape(image_size, image_size, 1))

        

    masks = np.array(masks)

    return x, masks
# Unet Network

import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("ggplot")

%matplotlib inline



# from tqdm import tqdm_notebook, tnrange

from itertools import chain

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split



import tensorflow as tf



from keras.models import Model, load_model

from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout

from keras.layers.core import Lambda, RepeatVector, Reshape

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D

from keras.layers.merge import concatenate, add

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",

               padding="same")(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    # second layer

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",

               padding="same")(x)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # contracting path

    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    p1 = MaxPooling2D((2, 2)) (c1)

    p1 = Dropout(dropout*0.5)(p1)



    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    p2 = MaxPooling2D((2, 2)) (c2)

    p2 = Dropout(dropout)(p2)



    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    p3 = MaxPooling2D((2, 2)) (c3)

    p3 = Dropout(dropout)(p3)



    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    p4 = Dropout(dropout)(p4)

    

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    

    # expansive path

    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)

    u6 = concatenate([u6, c4])

    u6 = Dropout(dropout)(u6)

    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)



    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)

    u7 = concatenate([u7, c3])

    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)



    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    u8 = Dropout(dropout)(u8)

    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)



    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    u9 = Dropout(dropout)(u9)

    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model
import keras.backend as K

# def dice_coef(y_true, y_pred):

#     y_true_f = K.flatten(y_true)

#     y_pred_f = K.flatten(y_pred)

#     intersection = K.sum(y_true_f * y_pred_f)

#     return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())



def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)
# Dice metric 

im_height=image_size

im_width=image_size



input_img = Input((im_height, im_width, 3), name='img')

model = get_unet(input_img, n_filters=32, dropout=0.05, batchnorm=True)



model.summary()
x_train_seg, y_train_seg = build_segmentation_dataset(segmentation_train,filtered_segmentation)

x_test_seg, y_test_seg = build_segmentation_dataset(segmentation_val,filtered_segmentation)

print("Finished")



x_train_seg_bak, y_train_seg_bak = x_train_seg, y_train_seg

x_test_seg_bak, y_test_seg_bak = x_test_seg, y_test_seg 
# #PreProcess

from keras.utils import np_utils

x_train_seg, y_train_seg = x_train_seg_bak, y_train_seg_bak

x_test_seg, y_test_seg = x_test_seg_bak, y_test_seg_bak



#z-score

mean = np.mean(x_train_seg,axis=(0,1,2,3))

std = np.std(x_train_seg,axis=(0,1,2,3))

x_train_seg = (x_train_seg-mean)/(std+1e-7)

x_test_seg = (x_test_seg-mean)/(std+1e-7)



#Data augmentation

# datagen = ImageDataGenerator(

#     rotation_range=15,

#     width_shift_range=0.1,

#     height_shift_range=0.1,

#     horizontal_flip=True,

#     )



datagen = ImageDataGenerator()

datagen.fit(x_train_seg)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

import keras



def lr_schedule(epoch):

    lrate = 0.001

    if epoch > 75:

        lrate = 0.0005

    if epoch > 100:

        lrate = 0.0003

    return lrate



# Prepare callbacks for model saving and for learning rate adjustment.

# checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

es_callback = EarlyStopping(patience=15, verbose=1),

cp_callback = ModelCheckpoint('model_dice.h5', verbose=1, save_best_only=True, save_weights_only=True)



opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

callbacks = [cp_callback,lr_reducer, lr_scheduler]



model.compile(optimizer=opt_rms, loss="mse", metrics=[dice_coef_loss,'mse','binary_crossentropy',"accuracy"])



# callbacks = [

#     EarlyStopping(patience=15, verbose=1),

#     ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

#     #ModelCheckpoint('model_seg_dice.h5', verbose=1, save_best_only=True, save_weights_only=True)

# ]
batch_size = 32

n_epochs = 30

n_steps = 30 #15*x_train_seg.shape[0] // batch_size



results = model.fit_generator(datagen.flow(x_train_seg, y_train_seg, batch_size=batch_size), epochs=n_epochs, steps_per_epoch=n_steps, shuffle=True, validation_data=datagen.flow(x_test_seg, y_test_seg, batch_size=batch_size), validation_steps=10,  verbose=1,callbacks=callbacks)
model.load_weights('model_dice.h5')
plt.figure(figsize=(8, 8))

plt.title("Learning curve")

plt.plot(results.history["acc"], label="acc")

plt.plot(results.history["val_acc"], label="val_acc")

plt.plot(np.argmax(results.history["val_acc"]), np.max(results.history["val_acc"]), marker="x", color="r", label="best model")

plt.xlabel("Epochs")

plt.ylabel("Acc")

plt.legend();
# Predict on train, val and test

preds_train = model.predict(x_train_seg, verbose=1)

preds_val = model.predict(x_test_seg, verbose=1)





# Threshold predictions

preds_train_t = (preds_train > 0.05).astype(np.uint8)

preds_val_t = (preds_val > 0.05).astype(np.uint8)





scores_train = model.evaluate(x_train_seg, y_train_seg, verbose=0)

scores_val = model.evaluate(x_test_seg, y_test_seg, verbose=0)

print(scores_train,scores_val)

import numpy as np

import matplotlib.pyplot as plt

def plot_Results_seg(img11,img12,img13):

    plt.figure(figsize=(15,20))

    plt.subplot(1, 3, 1)

    plt.imshow(img11)

    plt.title('Orig')

    plt.axis('off')

    

    plt.subplot(1, 3, 2)

    plt.imshow(img12,cmap="gray")

    plt.title("Orig Segmentation")

    plt.axis('off')

    

    plt.subplot(1, 3, 3)

    plt.imshow(img13,cmap="gray")

    plt.title("Predicted Segmentation")

    plt.axis('off')

    plt.show()



x_to_test = x_test_seg_bak

y_to_test = y_test_seg_bak

mask_pred = preds_val_t



# fijamos random para comparar dice con mse

# idx_rnd=[]

# for el in range(5):

#     idx_rnd.append(np.random.randint(len(x_test_seg)))

idx_rnd=[105, 46, 43, 51, 86]

for i in range(5):

    img11=x_to_test[idx_rnd[i]]

    img12=y_to_test[idx_rnd[i]]

    img12 = img12.reshape(image_size, image_size) 

    img13 = mask_pred[idx_rnd[i]]

    img13 = img13.reshape(image_size, image_size) 

    plot_Results_seg(img11,img12,img13)



# idx_rnd=[]

# for el in range(10):

#     idx_rnd.append(np.random.randint(len(x_test_seg)))

print(idx_rnd)
x_train_seg,org_mask,b_mask= build_segmentation_dataset(segmentation_train,filtered_segmentation)

import numpy as np

import matplotlib.pyplot as plt

def plot_Results_seg(img11,img12,img13):

    plt.figure(figsize=(15,20))

    plt.subplot(1, 3, 1)

    plt.imshow(img11)

    plt.title('Orig')

    plt.axis('off')

    

    plt.subplot(1, 3, 2)

    plt.imshow(img12)

    plt.title("Orig Mask")

    plt.axis('off')

    

    plt.subplot(1, 3, 3)

    plt.imshow(img13,cmap="gray")

    plt.title("Binary Mask")

    plt.axis('off')

    plt.show()



idx = np.random.randint(len(x_test_seg))

org=x_train_seg[idx]

mask=org_mask[idx]

binary_mask=b_mask[idx]

binary_mask= binary_mask.reshape(image_size, image_size) 



plot_Results_seg(org,mask,binary_mask)
