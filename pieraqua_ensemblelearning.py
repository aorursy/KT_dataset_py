!rm -r /kaggle/temp
!mkdir /kaggle/temp /kaggle/temp/train /kaggle/temp/test /kaggle/temp/train_labels

!unzip ../input/data-science-bowl-2018/stage2_test_final.zip -d /kaggle/temp/test

!unzip ../input/data-science-bowl-2018/stage1_test.zip -d /kaggle/temp/test

!unzip ../input/data-science-bowl-2018/stage1_train.zip -d /kaggle/temp/train

!unzip ../input/data-science-bowl-2018/stage1_train_labels.csv.zip /kaggle/temp/masks


!ls /kaggle/temp
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install Pillow

#!/usr/bin/env python

__author__ = "Sreenivas Bhattiprolu"

__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"



# https://www.youtube.com/watch?v=0kiroPnV1tM

# https://www.youtube.com/watch?v=cUHPL_dk17E

# https://www.youtube.com/watch?v=RaswBvMnFxk







"""

@author: Sreenivas Bhattiprolu

"""

import sklearn.model_selection

import tensorflow as tf

import os

import random

import numpy as np

import cv2

 

from tqdm import tqdm 

import datetime

from skimage.io import imread, imshow

from skimage.transform import resize

from skimage.morphology import label

import PIL

import matplotlib.pyplot as plt



from keras import backend as K



#seed = 42

#np.random.seed = seed



IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3



LIMIAR = 0.5



TRAIN_PATH = '/kaggle/temp/train/'

TEST_PATH = '/kaggle/temp/test/'



#iou_metric = sklearn.metrics.jaccard_score
"""

Here is a dice loss for keras which is smoothed to approximate a linear (L1) loss.

It ranges from 1 to 0 (no error), and returns results similar to binary crossentropy

"""





smooth = 1.





def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]



X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



print('Resizing training images and masks')

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   

    path = TRAIN_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img  #Fill empty X_train with values from img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)  

            

    Y_train[n] = mask   







cv2.imwrite("/kaggle/working/mask_1.png", np.float32(Y_train[0])*255)

cv2.imwrite("/kaggle/working/mask_2.png", np.float32(Y_train[1])*255)

img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  

path = TRAIN_PATH + train_ids[0]

cv2.imwrite("/kaggle/working/img_1.png",  imread(path + '/images/' + train_ids[0] + '.png')[:,:,:IMG_CHANNELS] )

path = TRAIN_PATH + train_ids[1]

cv2.imwrite("/kaggle/working/img_2.png",  imread(path + '/images/' + train_ids[1] + '.png')[:,:,:IMG_CHANNELS] )
# test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Resizing test images') 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = cv2.imread(path + '/images/' + id_ + '.png', cv2.IMREAD_COLOR)[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_train, Y_train, train_size = 0.8, shuffle = True)



X_train = None

Y_train = None
len(test_ids)
#learn.crit = MixedLoss(10.0, 2.0)

#learn.metrics=[accuracy_thresh(0.5),dice,IoU]





#Build the model

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)



#Contraction path

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)

c1 = tf.keras.layers.Dropout(0.1)(c1)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)



c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)

c2 = tf.keras.layers.Dropout(0.1)(c2)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

 

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

c3 = tf.keras.layers.Dropout(0.2)(c3)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

 

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)

c4 = tf.keras.layers.Dropout(0.2)(c4)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

 

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)

c5 = tf.keras.layers.Dropout(0.3)(c5)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)



    #Expansive path 

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

u6 = tf.keras.layers.concatenate([u6, c4])

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)

c6 = tf.keras.layers.Dropout(0.2)(c6)

c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

 

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

u7 = tf.keras.layers.concatenate([u7, c3])

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)

c7 = tf.keras.layers.Dropout(0.2)(c7)

c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

 

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

u8 = tf.keras.layers.concatenate([u8, c2])

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)

c8 = tf.keras.layers.Dropout(0.1)(c8)

c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

 

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)

c9 = tf.keras.layers.Dropout(0.1)(c9)

c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

 

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

 

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])



lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate=1e-5,

    decay_steps=10000,

    decay_rate=0.5)

opt = tf.keras.optimizers.Adam(learning_rate=(10e-4))

iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

#opt = tf.keras.optimizers.Adam(learning_rate=0.01)

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])

model.compile(optimizer=opt, loss=dice_coef_loss, metrics=['accuracy', iou_metric , dice_coef])

#model.summary()





#model.load_weights("/kaggle/working/model_dice.h5")

def reset_weights(model):

    session = K.get_session()

    for layer in model.layers: 

        if hasattr(layer, 'kernel_initializer'):

            layer.kernel.initializer.run(session=session)




################################

#Modelcheckpoint

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),

        tf.keras.callbacks.TensorBoard(log_dir='logs/body'),

        

]



results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=3, callbacks=callbacks)

#while float(results.history['loss'][-1]) > -0.6:

#    reset_weights(model)

#    model = build()

#    results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=3, callbacks=callbacks)

####################################
model.save_weights("/kaggle/working/model_dice.h5")




idx = random.randint(0, len(x_train))





preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(x_test, verbose=1)



 

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)





# Perform a sanity check on some random training samples

ix = random.randint(0, len(preds_train_t))

imshow(x_train[ix])

plt.show()

imshow(np.squeeze(y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()





# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))

#imshow(x_train[int(x_train.shape[0]*0.9):][ix])

plt.show()

imshow(np.squeeze(y_train[int(y_train.shape[0]*0.9):][ix]))

plt.show()

imshow(np.squeeze(preds_val_t[ix]))

plt.show()
#model.save_weights("/kaggle/working/model_dice.h5")
def build_model():

    #Build the model

    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)



    #Contraction path

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)

    c1 = tf.keras.layers.Dropout(0.1)(c1)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)



    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)

    c2 = tf.keras.layers.Dropout(0.1)(c2)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

 

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

    c3 = tf.keras.layers.Dropout(0.2)(c3)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

 

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)

    c4 = tf.keras.layers.Dropout(0.2)(c4)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

 

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)

    c5 = tf.keras.layers.Dropout(0.3)(c5)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)



    #Expansive path 

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = tf.keras.layers.concatenate([u6, c4])

    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)

    c6 = tf.keras.layers.Dropout(0.2)(c6)

    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

 

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = tf.keras.layers.concatenate([u7, c3])

    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)

    c7 = tf.keras.layers.Dropout(0.2)(c7)

    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

 

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = tf.keras.layers.concatenate([u8, c2])

    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)

    c8 = tf.keras.layers.Dropout(0.1)(c8)

    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

 

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)

    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)

    c9 = tf.keras.layers.Dropout(0.1)(c9)

    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

 

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

 

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    loss = "dice"

    if loss == "dice":

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 10e-4), loss=dice_coef_loss, metrics=['accuracy',iou_metric, dice_coef])

    else:

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 10e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy',iou_metric, dice_coef])

    return model
datagen_1 = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 60,fill_mode = "wrap")

datagen_2 = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.2,1.0], validation_split = 0.2, fill_mode = "wrap")

datagen_2_masks = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = 0.2, fill_mode = "wrap")

datagen_3 = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range = [0.5, 1.5],fill_mode = "wrap")

datagen_4 = tf.keras.preprocessing.image.ImageDataGenerator(vertical_flip = True,fill_mode = "wrap")

datagen_5 = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True, fill_mode = "wrap")

datagen_6 = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 60, zoom_range = [0.5, 1.5], fill_mode = "wrap")

datagen_7 = tf.keras.preprocessing.image.ImageDataGenerator(vertical_flip = True, horizontal_flip = True, rotation_range = 60, fill_mode = "wrap")





#for batch in datagen.flow(x, batch_size=1,seed=1337 ):



 

#data_1 = datagen_1.flow(x_train, seed=1337)

#masks_1 = datagen_1.flow(y_train, seed=1337)



#data_2 = datagen_2.flow(x_train, seed=1337)

#masks_2 = datagen_2.flow(y_train, seed=1337)



#data_3 = datagen_3.flow(x_train, seed=1337)

#masks_3 = datagen_3.flow(y_train, seed=1337)

logdir = os.path.join("logs/head_1", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_1 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]

logdir = os.path.join("logs/head_2", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_2 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]

logdir = os.path.join("logs/head_3", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_3 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]

logdir = os.path.join("logs/head_4", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_4 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]

logdir = os.path.join("logs/head_5", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_5 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]

logdir = os.path.join("logs/head_6", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_6 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]



logdir = os.path.join("logs/head_7", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_7 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]



logdir = os.path.join("logs/head_8", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_8 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]



logdir = os.path.join("logs/head_9", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_9 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]



logdir = os.path.join("logs/head_10", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_10 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]



logdir = os.path.join("logs/head_11", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  

callbacks_11 = [

        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),

        

        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

]
model_1 = build_model()

model_2 = build_model()

model_3 = build_model()

model_4 = build_model()

model_5 = build_model()

model_6 = build_model()

model_7 = build_model()

model_8 = build_model()

model_9 = build_model()

model_10 = build_model()

model_11 = build_model()



model_1.load_weights("/kaggle/working/model_dice.h5")

model_2.load_weights("/kaggle/working/model_dice.h5")

model_3.load_weights("/kaggle/working/model_dice.h5")

model_4.load_weights("/kaggle/working/model_dice.h5")

model_5.load_weights("/kaggle/working/model_dice.h5")

model_6.load_weights("/kaggle/working/model_dice.h5")

model_7.load_weights("/kaggle/working/model_dice.h5")

model_8.load_weights("/kaggle/working/model_dice.h5")

model_9.load_weights("/kaggle/working/model_dice.h5")



    

EPOCHS = 10

#for batch in datagen_1.flow(x_train, batch_size=1,seed=1337 ):

#    data_1.append(batch)

    

#for batch in datagen_1.flow(y_train, batch_size=1,seed=1337 ):

#    masks_1.append(batch)



 

datagen_img = datagen_1.flow(x_train, seed = 1337)

datagen_masks = datagen_1.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_1 = model_1.fit(datagen, batch_size=32, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_1)

datagen_img = datagen_4.flow(x_train, seed = 1337)

datagen_masks = datagen_4.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_4 = model_4.fit(datagen, batch_size=32, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_4)







datagen_img = datagen_2.flow(x_train, seed = 1337)

datagen_masks = datagen_2_masks.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_2 = model_2.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_2)

datagen_img = datagen_5.flow(x_train, seed = 1337)

datagen_masks = datagen_5.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_5 = model_5.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_5)



datagen_img = datagen_3.flow(x_train, seed = 1337)

datagen_masks = datagen_3.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_3 = model_3.fit(datagen, batch_size=32, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_3)

results_6 = model_6.fit(datagen, batch_size=32, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_6)

results_7 = model_7.fit(datagen, batch_size=32, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks = callbacks_7)

datagen_img = datagen_6.flow(x_train, seed = 1337)

datagen_masks = datagen_6.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_8 = model_8.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_8)

results_9 = model_9.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_9)
datagen_img = datagen_7.flow(x_train, seed = 1337)

datagen_masks = datagen_7.flow(y_train, seed = 1337)

datagen = zip(datagen_img, datagen_masks)

results_8 = model_10.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_10)

results_9 = model_11.fit(datagen, batch_size=16, steps_per_epoch = len(x_train)/16, epochs=EPOCHS, callbacks=callbacks_11)
#model_1.predict(x_test)
models = [ model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11]

         

#predictions = []

predictions = (model_5.predict(x_test)> LIMIAR).astype(int)

x=2

for model in models:

    x+=1

    predictions += (model.predict(x_test)>LIMIAR).astype(int)

    model.save_weights("model_"+str(x)+".h5")



predictions_new = (predictions >= 5).astype(int)

print(predictions_new)

models.append(model_5)
corpo_predict = model.predict(x_test)



# Perform a sanity check on some random training samples

ix = random.randint(0, len(corpo_predict))

imshow(x_test[ix])

plt.show()

imshow(np.squeeze(y_test[ix]))

plt.show()

imshow(np.squeeze(predictions_new[ix]))

plt.show()

imshow(np.squeeze(corpo_predict[ix]))

plt.show()
#model_1.metrics_names
results = []



hydra_results = []

acc = tf.keras.metrics.Accuracy()

iou = 0

dice = 0

x = -1

for image in predictions_new > 0.5:

    x+= 1

    if(x == 134):

        x = 133

    acc.update_state( y_test[x], image )

    iou += iou_metric(y_test[x], image )

    dice += dice_coef(tf.dtypes.cast(y_test[x], tf.float32), tf.dtypes.cast(image, tf.float32) )

    

x+=1

iou = iou/x

dice = dice/x





print( str(iou) + " " + str(dice) + " " + str(acc.result().numpy()))

#['loss', 'accuracy', 'mean_io_u', 'dice_coef']

hydra_results.append(-dice)

hydra_results.append(acc.result().numpy())

hydra_results.append(iou)

hydra_results.append(dice)

results.append(hydra_results)

hydra_results = None

iou = None

dice = None

acc = None

for model in models:

    results.append(model.evaluate(x_test, y_test, batch_size = 16))



modelos = []

accuracy = []

iou = []

dice = []

x=0

for result in results:

    if x == 0:

        modelos.append("Hydra")

    else:

        modelos.append("Cabeça_" + str(x))

    accuracy.append(result[1])

    iou.append(result[2])

    dice.append(result[3])

    x+=1

corpo_result = model.evaluate(x_test, y_test, batch_size = 16)

modelos.append("Corpo")

accuracy.append(corpo_result[1])

iou.append(corpo_result[2])

dice.append(corpo_result[3])

print(dice)

corpo_result = None

## Create results DataFrame

sub = pd.DataFrame()

sub['Models'] = modelos

sub['Accuracy'] = accuracy

sub['IoU'] = iou

sub['Dice Coefficient'] = dice

sub.to_csv('results.csv', index=False)
#x_test = None

#y_test = None

#datagen = None

predictions_h_t = predictions_new

#predictions_bool = ensemble_model.predict(X_test)





# Perform a sanity check on some random training samples

ix = random.randint(0, len(predictions_h_t))

imshow(x_test[ix])

plt.show()

#imshow(np.squeeze(Y_train[ix]))

#plt.show()

imshow(np.squeeze(predictions_h_t[ix]))

plt.show()







x=0

for model in models:

    x+=1

    if x == 1:

        predictions = model.predict(X_test)

    else:

        predictions += (model.predict(X_test))

    model.save_weights("model_"+str(x)+".h5")



#predictions = predictions - model_5_predict

predictions_new = predictions / len(models)
x_test = None

x_train = None

y_test = None

y_train = None

preds_test = predictions_new

# Create list of upsampled test masks

preds_test_upsampled = []

for i in range(len(preds_test)):

    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 

                                       (sizes_test[i][0], sizes_test[i][1]), 

                                       mode='constant', preserve_range=True))
# Run length Encoding from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python



from skimage.morphology import label



def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)
new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))
submission_df = pd.DataFrame()

submission_df['ImageId'] = new_test_ids

submission_df['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

submission_df.to_csv('sub-dsbowl2018-1.csv', index=False)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)
len(test_ids)
#new_test_ids = []

#rles = []

#for n, id_ in enumerate(test_ids):

#    rle = list(prob_to_rles(preds_test_upsampled[n]))

#    rles.extend(rle)

#    new_test_ids.extend([id_] * len(rle))

    
len(new_test_ids)
# Create submission DataFrame#

#sub = pd.DataFrame()

#sub['ImageId'] = new_test_ids

#sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

#sub.to_csv('sub-dsbowl2018-1.csv', index=False)

new_test_ids = None

rles = None

sub = None

preds_test = None

predictions_h_t = None
#x = 0

#models.append(model)

#for model in models:

#    if results[x][1] < 0.85:

#        continue

#    x+=1

#    preds_test = model.predict(X_test)

#    # Create list of upsampled test masks

#    preds_test_upsampled = []

#    for i in range(len(preds_test)):

#        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 

#                                       (sizes_test[i][0], sizes_test[i][1]), 

#                                       mode='constant', preserve_range=True))##



#    new_test_ids = []

#    rles = []

#    for n, id_ in enumerate(test_ids):

#        rle = list(prob_to_rles(preds_test_upsampled[n]))

#        rles.extend(rle)

#        new_test_ids.extend([id_] * len(rle))

#    

#    # Create submission DataFrame

#    sub = pd.DataFrame()

#    sub['ImageId'] = new_test_ids

#    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

#    sub.to_csv('sub-dsbowl2018-1-'+str(x)+'.csv', index=False)

#    

#    new_test_ids = None

#    rles = None

#    sub = None

#    preds_test = None

#    preds_test_upsampled = None
#print(results[0])

#print(model_1.metrics_names)