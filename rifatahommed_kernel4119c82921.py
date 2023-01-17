# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
import os

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use("ggplot")

%matplotlib inline



import cv2

from tqdm import tqdm_notebook, tnrange

from glob import glob

from itertools import chain

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split



import tensorflow as tf

from skimage.color import rgb2gray

from tensorflow.keras import Input

from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Set some parameters

im_width = 256

im_height = 256
train_files = []

mask_files = glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')



for i in mask_files:

    train_files.append(i.replace('_mask',''))



print(train_files[:10])

print(mask_files[:10])
#Lets plot some samples

rows,cols=3,3

fig=plt.figure(figsize=(10,10))

for i in range(1,rows*cols+1):

    fig.add_subplot(rows,cols,i)

    img_path=train_files[i]

    msk_path=mask_files[i]

    img=cv2.imread(img_path)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    msk=cv2.imread(msk_path)

    plt.imshow(img)

    plt.imshow(msk,alpha=0.4)

plt.show()
smooth=100



def dice_coef(y_true, y_pred):

    y_truef=K.flatten(y_true)

    y_predf=K.flatten(y_pred)

    And=K.sum(y_truef* y_predf)

    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



def iou(y_true, y_pred):

    intersection = K.sum(y_true * y_pred)

    sum_ = K.sum(y_true + y_pred)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac



def jac_distance(y_true, y_pred):

    y_truef=K.flatten(y_true)

    y_predf=K.flatten(y_pred)



    return - iou(y_true, y_pred)
def unet(input_size=(256,256,1)):

    inputs = Input(input_size)

    

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)

    bn1 = Activation('relu')(conv1)

    conv1 = Conv2D(32, (3, 3), padding='same')(bn1)

    bn1 = BatchNormalization(axis=3)(conv1)

    bn1 = Activation('relu')(bn1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)



    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)

    bn2 = Activation('relu')(conv2)

    conv2 = Conv2D(64, (3, 3), padding='same')(bn2)

    bn2 = BatchNormalization(axis=3)(conv2)

    bn2 = Activation('relu')(bn2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)



    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)

    bn3 = Activation('relu')(conv3)

    conv3 = Conv2D(128, (3, 3), padding='same')(bn3)

    bn3 = BatchNormalization(axis=3)(conv3)

    bn3 = Activation('relu')(bn3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)



    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)

    bn4 = Activation('relu')(conv4)

    conv4 = Conv2D(256, (3, 3), padding='same')(bn4)

    bn4 = BatchNormalization(axis=3)(conv4)

    bn4 = Activation('relu')(bn4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)



    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)

    bn5 = Activation('relu')(conv5)

    conv5 = Conv2D(512, (3, 3), padding='same')(bn5)

    bn5 = BatchNormalization(axis=3)(conv5)

    bn5 = Activation('relu')(bn5)



    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), padding='same')(up6)

    bn6 = Activation('relu')(conv6)

    conv6 = Conv2D(256, (3, 3), padding='same')(bn6)

    bn6 = BatchNormalization(axis=3)(conv6)

    bn6 = Activation('relu')(bn6)



    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), padding='same')(up7)

    bn7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, (3, 3), padding='same')(bn7)

    bn7 = BatchNormalization(axis=3)(conv7)

    bn7 = Activation('relu')(bn7)



    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), padding='same')(up8)

    bn8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, (3, 3), padding='same')(bn8)

    bn8 = BatchNormalization(axis=3)(conv8)

    bn8 = Activation('relu')(bn8)



    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), padding='same')(up9)

    bn9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, (3, 3), padding='same')(bn9)

    bn9 = BatchNormalization(axis=3)(conv9)

    bn9 = Activation('relu')(bn9)



    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)



    return Model(inputs=[inputs], outputs=[conv10])
# From: https://github.com/zhixuhao/unet/blob/master/data.py

def train_generator(data_frame, batch_size, aug_dict,

        image_color_mode="rgb",

        mask_color_mode="grayscale",

        image_save_prefix="image",

        mask_save_prefix="mask",

        save_to_dir=None,

        target_size=(256,256),

        seed=1):

    '''

    can generate image and mask at the same time use the same seed for

    image_datagen and mask_datagen to ensure the transformation for image

    and mask is the same if you want to visualize the results of generator,

    set save_to_dir = "your path"

    '''

    image_datagen = ImageDataGenerator(**aug_dict)

    mask_datagen = ImageDataGenerator(**aug_dict)

    

    image_generator = image_datagen.flow_from_dataframe(

        data_frame,

        x_col = "filename",

        class_mode = None,

        color_mode = image_color_mode,

        target_size = target_size,

        batch_size = batch_size,

        save_to_dir = save_to_dir,

        save_prefix  = image_save_prefix,

        seed = seed)



    mask_generator = mask_datagen.flow_from_dataframe(

        data_frame,

        x_col = "mask",

        class_mode = None,

        color_mode = mask_color_mode,

        target_size = target_size,

        batch_size = batch_size,

        save_to_dir = save_to_dir,

        save_prefix  = mask_save_prefix,

        seed = seed)



    train_gen = zip(image_generator, mask_generator)

    

    for (img, mask) in train_gen:

        img, mask = adjust_data(img, mask)

        yield (img,mask)



def adjust_data(img,mask):

    img = img / 255

    mask = mask / 255

    mask[mask > 0.5] = 1

    mask[mask <= 0.5] = 0

    

    return (img, mask)
from sklearn.model_selection import KFold

import pandas



kf = KFold(n_splits = 5, shuffle=False)

df = pandas.DataFrame(data={"filename": train_files, 'mask' : mask_files})



df2 = df.sample(frac=1).reset_index(drop=True)


train_generator_args = dict(rotation_range=0.2,

                            width_shift_range=0.05,

                            height_shift_range=0.05,

                            shear_range=0.05,

                            zoom_range=0.05,

                            horizontal_flip=True,

                            fill_mode='nearest')



histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



EPOCHS = 40

BATCH_SIZE = 1



for k, (train_index, test_index) in enumerate(kf.split(df2)):

    train_data_frame = df2.iloc[train_index]

    test_data_frame = df2.iloc[test_index]

    

    train_gen = train_generator(train_data_frame, BATCH_SIZE,

                                train_generator_args,

                                target_size=(im_height, im_width))

    

    test_gener = train_generator(test_data_frame, BATCH_SIZE,

                                dict(),

                                target_size=(im_height, im_width))

    

    model = unet(input_size=(im_height, im_width, 3))

    

    model.compile(optimizer=SGD(lr=0.001, nesterov=True), loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])

    callbacks = [ModelCheckpoint(str(k+1) + '_unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]

    history = model.fit(train_gen,

                                  steps_per_epoch=len(train_data_frame) / BATCH_SIZE, 

                                  epochs=EPOCHS, 

                                  callbacks=callbacks,

                                  validation_data = test_gener,

                                  validation_steps=len(test_data_frame) / BATCH_SIZE)

    

    model = load_model(str(k+1) + '_unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    

    test_gen = train_generator(test_data_frame, BATCH_SIZE,

                                dict(),

                                target_size=(im_height, im_width))

    results = model.evaluate(test_gen, steps=len(test_data_frame) / BATCH_SIZE)

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['binary_accuracy'])

    losses.append(results['loss'])

    dicecoefs.append(results['dice_coef'])

    ious.append(results['iou'])



    # for dev purpose

    #break



    del model
print('accuracies : ', accuracies)

print('losses : ', losses)

print('dicecoefs : ', dicecoefs)

print('ious : ', ious)



print('-----------------------------------------------------------------------------')

print('-----------------------------------------------------------------------------')



print('average accuracy : ', np.mean(np.array(accuracies)))

print('average loss : ', np.mean(np.array(losses)))

print('average dicecoefs : ', np.mean(np.array(dicecoefs)))

print('average ious : ', np.mean(np.array(ious)))

print()



print('standard deviation of accuracy : ', np.std(np.array(accuracies)))

print('standard deviation of loss : ', np.std(np.array(losses)))

print('standard deviation of dicecoefs : ', np.std(np.array(dicecoefs)))

print('standard deviation of ious : ', np.std(np.array(ious)))
for h, history in enumerate(histories):



    keys = history.history.keys()

    fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))

    fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)



    for k, key in enumerate(list(keys)[:len(keys)//2]):

        training = history.history[key]

        validation = history.history['val_' + key]



        epoch_count = range(1, len(training) + 1)



        axs[k].plot(epoch_count, training, 'r--')

        axs[k].plot(epoch_count, validation, 'b-')

        axs[k].legend(['Training ' + key, 'Validation ' + key])
model = load_model('1_unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
for i in range(20):

    index=np.random.randint(1,len(test_data_frame.index))

    img = cv2.imread(test_data_frame['filename'].iloc[index])

    img = cv2.resize(img ,(im_height, im_width))

    img = img / 255

    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(np.squeeze(img))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.imread(test_data_frame['mask'].iloc[index])))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()
 