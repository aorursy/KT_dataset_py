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

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#Set Parameters

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
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})

df_train, df_test = train_test_split(df,test_size = 0.1)

df_train, df_val = train_test_split(df_train,test_size = 0.2)

print(df_train.values.shape)

print(df_val.values.shape)

print(df_test.values.shape)
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
def conv_bn_relu_block(num_kernel,input_layer):

    x = Conv2D(num_kernel, (3,3), padding='same')(input_layer)

    x = BatchNormalization(axis=3)(x)

    x = Activation('relu')(x)

    x = Conv2D(num_kernel, (3,3), padding='same')(x)

    x = BatchNormalization(axis=3)(x)

    x = Activation('relu')(x)

    return x

def maxpool_conv_bn_relu_block(num_kernel, input_layer):

    x = MaxPooling2D(pool_size=(2,2))(input_layer)

    x = conv_bn_relu_block(num_kernel,x)

    return x

def upconv_conv_bn_relu_block(num_kernel,input_layer,concat_layer):

    x = concatenate([Conv2DTranspose(num_kernel, (2, 2), strides=(2, 2), padding='same')(input_layer), concat_layer], axis=3)

    x = conv_bn_relu_block(num_kernel,x)

    return x

    
def unet(input_size = (256,256,3)):

    inputs = Input(input_size)

    

    conv_bn_relu_block_64 = conv_bn_relu_block(64,inputs)

    maxpool_conv_bn_relu_128 = maxpool_conv_bn_relu_block(128,conv_bn_relu_block_64)

    maxpool_conv_bn_relu_256 = maxpool_conv_bn_relu_block(256,maxpool_conv_bn_relu_128)

    maxpool_conv_bn_relu_512 = maxpool_conv_bn_relu_block(512,maxpool_conv_bn_relu_256)

    maxpool_conv_bn_relu_1024 = maxpool_conv_bn_relu_block(1024,maxpool_conv_bn_relu_512)

    

    upconv_conv_bn_relu_512 = upconv_conv_bn_relu_block(512,maxpool_conv_bn_relu_1024,maxpool_conv_bn_relu_512)

    upconv_conv_bn_relu_256 = upconv_conv_bn_relu_block(256,upconv_conv_bn_relu_512,maxpool_conv_bn_relu_256)

    upconv_conv_bn_relu_128 = upconv_conv_bn_relu_block(128,upconv_conv_bn_relu_256,maxpool_conv_bn_relu_128)

    upconv_conv_bn_relu_64 = upconv_conv_bn_relu_block(64,upconv_conv_bn_relu_128,conv_bn_relu_block_64)

    

    last_conv = Conv2D(1, (1, 1), activation='sigmoid')(upconv_conv_bn_relu_64)



    return Model(inputs=[inputs], outputs=[last_conv])

    

    
EPOCHS = 150

BATCH_SIZE = 16

learning_rate = 1e-4
train_generator_args = dict(rotation_range=0.2,

                            width_shift_range=0.05,

                            height_shift_range=0.05,

                            shear_range=0.05,

                            zoom_range=0.05,

                            horizontal_flip=True,

                            fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE,

                                train_generator_args,

                                target_size=(im_height, im_width))

    

test_gener = train_generator(df_val, BATCH_SIZE,

                                dict(),

                                target_size=(im_height, im_width))

    

model = unet(input_size=(im_height, im_width, 3))







decay_rate = learning_rate / EPOCHS

opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])



callbacks = [ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]



history = model.fit(train_gen,

                    steps_per_epoch=len(df_train) / BATCH_SIZE, 

                    epochs=EPOCHS, 

                    callbacks=callbacks,

                    validation_data = test_gener,

                    validation_steps=len(df_val) / BATCH_SIZE)
a = history.history





list_trainjaccard = a['iou']

list_testjaccard = a['val_iou']



list_trainloss = a['loss']

list_testloss = a['val_loss']

plt.figure(1)

plt.plot(list_testloss, 'b-',label = "Val loss")

plt.plot(list_trainloss,'r-', label = "Train loss")

plt.xlabel('Iteration')

plt.ylabel('Loss')

plt.title('Loss graph', fontsize = 15)

plt.figure(2)

plt.plot(list_trainjaccard, 'r-',label = "Train IOU")

plt.plot(list_testjaccard, 'b-',label = "Val IOU" )

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.title('Accuracy graph', fontsize = 15)

plt.show()
model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
test_gen = train_generator(df_test, BATCH_SIZE,

                                dict(),

                                target_size=(im_height, im_width))

results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)

print("Test lost: ",results[0])

print("Test IOU: ",results[2])

print("Test Dice Coefficent: ",results[3])
for i in range(30):

    index=np.random.randint(1,len(df_test.index))

    img = cv2.imread(df_test['filename'].iloc[index])

    img = cv2.resize(img ,(im_height, im_width))

    img = img / 255

    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(np.squeeze(img))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()