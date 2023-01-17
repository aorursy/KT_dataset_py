# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

%matplotlib inline

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# Set some parameters
img_size_ori = 101
img_size_target = 101
im_width = 101
im_height = 101
im_chan = 1
basicpath = '../input/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'
# Loading of training/testing ids and depths

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)
train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
train_df.head()
fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_df.coverage, kde=False, ax=axs[0])
sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")
# Create train/validation split stratified by salt coverage

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)
depth_train.shape
ACTIVATION = "relu"

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation(ACTIVATION)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = Activation(ACTIVATION)(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x
# DeepUnet model
def build_DeepUnet_model(inputs):
    # 101 -> 50
    down0 = Conv2D(64, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0) #
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0) #
    down0 = Activation('relu')(down0)
    plus_0 = Conv2D(32, (2, 2), padding='same')(down0)
    down0 = Add()([down0,inputs]) #
    down0 = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    down0 = Dropout(0.25)(down0) #
    down0 = Activation('relu')(down0)

    # 50 -> 25

    down1 = Conv2D(64, (3, 3), padding='same')(down0)
    down1 = BatchNormalization()(down1) #
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1) #
    plus_1 = concatenate([down1, down0], axis=3)
    down1 = Add()([down1, down0]) #
    down1= MaxPooling2D((2, 2), strides=(2, 2))(down1)
    down1 = Dropout(0.5)(down1) #
    down1 = Activation('relu')(down1)
    

    # 25 -> 12

    down2 = Conv2D(64, (3, 3), padding='same')(down1)
    down2 = BatchNormalization()(down2) #
    down2 = Activation('relu')(down2)
    down2 = Conv2D(32, (2, 2), padding='same')(down2)
    down2 = BatchNormalization()(down2) #
    plus_2 = concatenate([down2, down1], axis=3)
    down2 = Add()([down2, down1]) #
    down2 = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    down2 = Dropout(0.5)(down2) #
    down2 = Activation('relu')(down2)

    # 12 -> 6

    down3 = Conv2D(64, (3, 3), padding='same')(down2)
    down3 = BatchNormalization()(down3) #
    down3 = Activation('relu')(down3)
    down3 = Conv2D(32, (2, 2), padding='same')(down3)
    down3 = BatchNormalization()(down3) #
    plus_3 = concatenate([down3, down2], axis=3)
    down3 = Add()([down3,down2]) #
    down3 = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    down3 = Dropout(0.5)(down3) #
    down3 = Activation('relu')(down3)

    # 6 - > 12
    up4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(down3)

    # 12 -> 25

    up3 = concatenate([up4, plus_3], axis=3)
    up3 = Dropout(0.5)(up3) #
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3) #
    up3 = Activation('relu')(up3)
    up3 = Conv2D(32, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3) #
    #up3 = concatenate([up3, up4], axis=3)
    up3 = Add()([up3,up4]) #
    up3 = Activation('relu')(up3)
    up3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="valid")(up3)

    # 25 -> 50

    up2 = concatenate([up3, plus_2], axis=3)
    up2 = Dropout(0.5)(up2) #
    up2 = Conv2D(64, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2) #
    up2 = Activation('relu')(up2)
    up2 = Conv2D(32, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2) #
    #up2 = concatenate([up2, up3], axis=3)
    up2 = Add()([up2,up3]) #
    up2 = Activation('relu')(up2)
    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(up3)

    # 50 -> 101

    up1 = concatenate([up2, plus_1], axis=3)
    up1 = Dropout(0.5)(up1) #
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1) #
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1) #
    #up1 = concatenate([up1, up2], axis=3)
    up1 = Add()([up1,up2]) #
    up1 = Activation('relu')(up1)
    up1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="valid")(up1)

    up1 = Dropout(0.25)(up1) #
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(up1)
    model = Model(input_layer, output_layer)
    return model   
# Build Unet model
def build_Unet_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16)
    convm = Activation(ACTIVATION)(convm)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = Activation(ACTIVATION)(uconv4)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = Activation(ACTIVATION)(uconv2)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = Activation(ACTIVATION)(uconv1)
    
    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    model = Model(input_layer, output_layer)
    return model #output_layer
iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0)
    if u == 0:
        return u
    return i/u

def iou_metric(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
            
    return scores.mean()
#Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
print(x_train.shape)
print(y_valid.shape)
def compile_and_train(model, epochs,batch_size,model_name): 
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    early_stopping = EarlyStopping(monitor='val_acc', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("./" + model_name + "_best.model",monitor='val_acc', 
                                   mode = 'max', save_best_only=True, verbose=1)
    #model_checkpoint = ModelCheckpoint("./" + model_name + "_best.h5",monitor='val_acc', 
    #                               mode = 'max', save_weights_only=True, save_best_only=True, period=1, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max',factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1)
    #tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    #history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.2)
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], 
                    verbose=1)
    return history
input_layer = Input((img_size_target, img_size_target, 1))
Unet_model = build_Unet_model(input_layer, 16,0.5)

history_unet = compile_and_train(Unet_model, 200, 32,"Unet")
input_layer = Input((img_size_target, img_size_target, 1))
DeepUnet_model = build_DeepUnet_model(input_layer)
history_deepunet = compile_and_train(DeepUnet_model, 50, 32,"DeepUnet")
import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history_unet.history['acc'][1:])
plt.plot(history_unet.history['val_acc'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.show()
import matplotlib.pyplot as plt
# summarize history for loss
plt.plot(history_deepunet.history['acc'][1:])
plt.plot(history_deepunet.history['val_acc'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.show()
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history_unet.epoch, history_unet.history["loss"], label="Train loss")
ax_loss.plot(history_unet.epoch, history_unet.history["val_loss"], label="Validation loss")
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history_deepunet.epoch, history_deepunet.history["loss"], label="Train loss")
ax_loss.plot(history_deepunet.epoch, history_deepunet.history["val_loss"], label="Validation loss")
def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(a) for a in model.predict(np.array([np.fliplr(x) for x in x_test])).reshape(-1, img_size_target, img_size_target)])
    return preds_test/2.0
preds_valid = predict_result(DeepUnet_model,x_valid,img_size_target)
def filter_image(img):
    if img.sum() < 100:
        return np.zeros(img.shape)
    else:
        return img

## Scoring for last model
thresholds = np.linspace(0.3, 0.7, 31)
ious = np.array([iou_metric(y_valid.reshape((-1, img_size_target, img_size_target)), [filter_image(img) for img in preds_valid > threshold]) for threshold in tqdm_notebook(thresholds)])
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
import gc

del x_train, x_valid, y_train, y_valid, preds_valid
gc.collect()
x_test = np.array([(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)

#preds_test = predict_result(model,x_test,img_size_target)
preds_test_unet = predict_result(Unet_model,x_test,img_size_target)
preds_test_deepunet = predict_result(DeepUnet_model,x_test,img_size_target)
preds_test_unet
preds_test_deepunet
# take the average of the models - ensemble
preds_test  = np.mean([preds_test_unet, preds_test_deepunet], axis=0)
preds_test
import time
t1 = time.time()
pred_dict = {idx: rle_encode(filter_image(preds_test[i] > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
sub.head(10)
