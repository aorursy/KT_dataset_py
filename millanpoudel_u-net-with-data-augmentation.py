# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rand
import skimage.io
import matplotlib.pyplot as plt
from skimage import transform
import os, glob
import shutil
from tqdm import tqdm
import tensorflow as tf
from skimage.io import imread, imshow

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# print(check_output(["ls", "../input/"]).decode("utf8"))

os.getcwd()
train_path = '../input/road-segmentation-dataset/data/training/images'
train_images = [f for f in glob.glob(train_path + "*/*.png", recursive=True)]
print(len(train_images))

ground_t_path = '../input/road-segmentation-dataset/data/training/groundtruth'
train_masks= [f for f in glob.glob(ground_t_path + "*/*.png", recursive=True)]
print(len(train_masks))
#DIMENSIONS FOR IMAGES TO TRAIN AND TEST
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3 

X_train = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8) 
Y_train = np.zeros((len(train_masks), IMG_HEIGHT, IMG_WIDTH,1), dtype = np.bool)
import cv2 as cv
for i in range(len(train_images)):
    img=cv.imread(train_images[i])
    X_train[i]=cv.resize(img,  (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_AREA)
X_train.shape
#RESHAPING  MASKS to (128*128)
for i in range(len(train_masks)):
    img = cv.imread(train_masks[i])
    fin_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(fin_img,  (IMG_WIDTH, IMG_WIDTH), interpolation = cv.INTER_AREA)
    Y_train[i]=np.expand_dims(resized,axis=-1)

import random
image_x = random.randint(4, len(train_images))
imshow(np.squeeze(Y_train[image_x]))
plt.show()
imshow(np.squeeze(X_train[image_x]))
plt.show()
# augmentation images and corresponding masks with Numpy
image = []
masks = []
temp_mask=np.zeros(shape=(0,128,128,1))
for r in range(200):
    l = rand.randint(1,99)
    img=X_train[l]
    mask=Y_train[l]
    flipped_img=np.fliplr(img)
    flipped_mask=np.fliplr(mask)
    image.append(flipped_img)
    interm_arr=np.asarray(flipped_mask)
    interm_arr=np.expand_dims(interm_arr,axis=0)
    temp_mask=np.append(temp_mask, interm_arr, axis=0)

images_arr=np.asarray(image)
X_train=np.append(X_train, images_arr, axis=0)
Y_train=np.append(Y_train,temp_mask, axis=0)

print(X_train.shape)
print(Y_train.shape)
#Build the model(EXPANSION PATH)

# Step 1: Defining the input layer 
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
# Step 2: Convert the integer values of the input to floating points 
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#LAYER 1
# Step 3: Forming the first conv layer 
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(s)
# Step 4: Add dropout 
c1 = tf.keras.layers.Dropout(rate = 0.1)(c1)
# Step 5: Forming the second conv layer
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
# Step 6: Forming the max pool layer 
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

#LAYER 2
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
c2 = tf.keras.layers.Dropout(rate = 0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

#LAYER 3
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
c3 = tf.keras.layers.Dropout(rate = 0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

#LAYER 4
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
c4 = tf.keras.layers.Dropout(rate = 0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

#LAYER 5
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
c5 = tf.keras.layers.Dropout(rate = 0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)
#Build the Model(CONTRACTION PATH)

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(rate = 0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(rate = 0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(rate = 0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis = 3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(rate = 0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)

model1 = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model1.summary()
# Adding a model checkpoint 

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_new.h5', verbose = 1, save_best_only = True)

callback = [tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'),
            tf.keras.callbacks.TensorBoard(histogram_freq = 0)]

results = model1.fit(X_train, Y_train, validation_split = 0.10, batch_size = 20, epochs = 25, callbacks = callback)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
test_path = "../input/road-segmentation-dataset/data/test_set_images/"
test_images = [f for f in glob.glob(test_path + "*/*.png", recursive=True)]
print(test_images[0])
print(len(test_images))
X_test= np.zeros((len(test_images), 128,128,3), dtype = np.uint8) 
X_test.shape
for i in range(len(test_images)):
    img=cv.imread(test_images[i])
    X_test[i]=cv.resize(img,  (128,128), interpolation = cv.INTER_AREA)

X_test.shape
predict_train = model1.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose = 1)
predict_val = model1.predict(X_train[int(X_train.shape[0] * 0.9):], verbose = 1)
pred_test = model1.predict(X_test, verbose = 1)
predict_train_t = (predict_train > 0.55).astype(np.uint8)
predict_val_t = (predict_val > 0.55).astype(np.uint8)
pred_test_t = (predict_val > 0.55).astype(np.uint8)
#Perform a sanity check on some random training samples 
ix = random.randint(0, len(predict_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(predict_train_t[ix]))
plt.show()
#IOU CHECKING FOR MODEL EVALUATION

component1 = np.float32(Y_train[4])
component2 = predict_train_t[4]
overlap = np.logical_and(component1, component2) # Logical AND
union = np.logical_or(component1, component2) # Logical OR

IOU = overlap.sum()/float(union.sum())
print(IOU)
predict_train_t = (predict_train > 0.55).astype(np.uint8)
predict_val_t = (predict_val > 0.55).astype(np.uint8)
pred_test_t = (predict_val > 0.55).astype(np.uint8)
#Perform a sanity check on some random training samples 
ix = random.randint(0, len(pred_test_t))
imshow(X_test[ix])
plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()
imshow(np.squeeze(pred_test_t[ix]))
plt.show()
#IOU CHECKING FOR MODEL EVALUATION

# component1 = np.float32(Y_train[4])
# component2 = predict_train_t[4]
# overlap = np.logical_and(component1, component2) # Logical AND
# union = np.logical_or(component1, component2) # Logical OR

# IOU = overlap.sum()/float(union.sum())
# print(IOU)
