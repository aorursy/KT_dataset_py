import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys



import warnings

warnings.filterwarnings("ignore")



import random

random.seed = 42

np.random.seed = 42



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_path = "../input/stage1_train/"

test_path = "../input/stage1_test/"
train_ids = next(os.walk(train_path))[1]

test_ids = next(os.walk(test_path))[1]
print("total subfolders in stage1_train",len(train_ids))

print("total subfolders in stage1_test ",len(test_ids))
import cv2

import matplotlib.pyplot as plt

import numpy as np

def openCVdemo(path):

    img = cv2.imread(path,0)

    # Otsu's thresholding after Gaussian filtering

    blur = cv2.GaussianBlur(img,(5,5),0)

    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Plot Here

    plt.figure(figsize=(15,5))

    images = [blur, 0, th3]

    titles = ['Original Image (X_train)','Gaussian filtered Image (OpenCV)',"Segmened Image (OpenCV)"]

    plt.subplot(1,3,1),plt.imshow(img,'gray')

    plt.title(titles[0]), plt.xticks([]), plt.yticks([])

    plt.subplot(1,3,2),plt.imshow(images[0],'gray')

    plt.title(titles[1]), plt.xticks([]), plt.yticks([])

    plt.subplot(1,3,3),plt.imshow(images[2],'gray')

    plt.title(titles[2]), plt.xticks([]), plt.yticks([])
n = random.randint(0,len(train_ids))

path = train_path+train_ids[n]+"/images/"+train_ids[n]+".png"

openCVdemo(path)
IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow

from skimage.transform import resize

from skimage.morphology import label


print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = train_path + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = test_path + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
def plotTrainData(a,b):

    for i in range(5):

        ix = random.randint(0, len(train_ids))

        plt.subplot(1,2,1)

        plt.title("X_train")

        imshow(a[ix])

        plt.axis('off')

        plt.subplot(1,2,2)

        plt.title("Y_train")

        imshow(np.squeeze(b[ix]))

        plt.axis('off')

        plt.show()

plotTrainData(X_train,Y_train)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)

print('\nx_train',x_train.shape)

print('x_test',x_test.shape)

print('y_train',y_train.shape)

print('y_test',y_test.shape)
import tensorflow as tf

# Define IoU metric

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)
from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
# Build U-Net model

inputs = Input((IMG_WIDTH , IMG_HEIGHT , IMG_CHANNELS))

s = Lambda(lambda x:x/255)(inputs)

#Layer1

c1 = Conv2D(16,(3,3),activation = "elu",kernel_initializer = "he_normal",padding="same")(s)

c1 = Dropout(0.1)(c1)

c1 = Conv2D(16,(3,3),activation = "elu",kernel_initializer = "he_normal", padding="same")(c1)

p1 = MaxPooling2D((2,2))(c1)



#Layer2

c2 = Conv2D(32,(3,3),activation = "elu",kernel_initializer = "he_normal",padding="same")(p1)

c2 = Dropout(0.1)(c2)

c2 = Conv2D(32,(3,3),activation = "elu",kernel_initializer = "he_normal", padding="same")(c2)

p2 = MaxPooling2D((2,2))(c2)



#Layer3

c3 = Conv2D(64,(3,3),activation = "elu",kernel_initializer = "he_normal",padding="same")(p2)

c3 = Dropout(0.2)(c3)

c3 = Conv2D(64,(3,3),activation = "elu",kernel_initializer = "he_normal", padding="same")(c3)

p3 = MaxPooling2D((2,2))(c3)



#layer4

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



#layer5

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

model.summary()
# Fit model

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 

                    callbacks=[earlystopper, checkpointer])
# Predict on train, val and test

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(X_test, verbose=1)



# Threshold predictions

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Create list of upsampled test masks

preds_test_upsampled = []

for i in range(len(preds_test)):

    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 

                                       (sizes_test[i][0], sizes_test[i][1]), 

                                       mode='constant', preserve_range=True))
# Perform a sanity check on some random training samples

ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()
# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))

imshow(X_train[int(X_train.shape[0]*0.9):][ix])

plt.show()

imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))

plt.show()

imshow(np.squeeze(preds_val_t[ix]))

plt.show()
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
new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))
# Create submission DataFrame

sub = pd.DataFrame()

sub['ImageId'] = new_test_ids

sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

sub.to_csv('sub-dsbowl2018-1.csv', index=False)