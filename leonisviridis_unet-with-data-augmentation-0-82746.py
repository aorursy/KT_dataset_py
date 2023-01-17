#RLE functions from https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode/code



from skimage.io import imread, imshow, imread_collection, concatenate_images

import matplotlib.pyplot as plt

import os

import sys

import random

import warnings



import numpy as np

from numpy import fliplr, flipud

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label



from keras.models import Model, load_model

from keras.optimizers import Adam

from keras.layers import Input, BatchNormalization

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K



import tensorflow as tf





def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

 

def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T

IMG_HEIGHT = 256

IMG_WIDTH = 1600

IMG_CHANNELS = 3

DEFECT_CLASSES = 4

SCALE_FACTOR = 2

SAMPLE_SIZE = 940



train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

IMAGE_PATH = r"../input/severstal-steel-defect-detection/train_images/"

TEST_IMAGE_PATH = r"../input/severstal-steel-defect-detection/test_images/"
train.head(20)
#import images

from skimage import color

from skimage import io



#import masks

class_indices = [[],[],[],[]]



for ind in range(train.shape[0]):

    if train['EncodedPixels'][ind] != '':

        class_indices[train['ClassId'][ind]-1].append(ind) #get this index right

    

    

Y_train = np.zeros((SAMPLE_SIZE*4, int(IMG_HEIGHT/SCALE_FACTOR), int(IMG_WIDTH/SCALE_FACTOR), int(DEFECT_CLASSES)), dtype=np.bool)





#get a random sample from each class

sample_set = []

for defect_type in range(4):

    partial_sample_set = np.random.permutation(int(SAMPLE_SIZE/4))

    for sample_index in partial_sample_set:

        sample_set.append(class_indices[defect_type][sample_index])



n=0        

for ind in range(len(sample_set)):

    img = rle2mask(train['EncodedPixels'][sample_set[ind]])

    img = resize(img, (IMG_HEIGHT/SCALE_FACTOR, IMG_WIDTH/SCALE_FACTOR), mode='constant', preserve_range=True)

    Y_train[n,:,:,(train['ClassId'][sample_set[ind]]-1)] = img

    Y_train[n+1,:,:,(train['ClassId'][sample_set[ind]]-1)] = fliplr(img)

    Y_train[n+2,:,:,(train['ClassId'][sample_set[ind]]-1)] = flipud(img)

    Y_train[n+3,:,:,(train['ClassId'][sample_set[ind]]-1)] = fliplr(flipud(img))

    n+=4

len(class_indices[3])
X_train = np.zeros((SAMPLE_SIZE*4, int(IMG_HEIGHT/SCALE_FACTOR), int(IMG_WIDTH/SCALE_FACTOR), int(IMG_CHANNELS)), dtype=np.uint8)

n = 0

for ind in range(len(sample_set)):

    #img = color.rgb2gray(io.imread(IMAGE_PATH + train['ImageId'][ind]))[:,:]

    img = imread(IMAGE_PATH + train['ImageId'][sample_set[ind]])

    img = resize(img, (IMG_HEIGHT/SCALE_FACTOR, IMG_WIDTH/SCALE_FACTOR, IMG_CHANNELS), mode='constant', preserve_range=True)

    X_train[n,:,:,:] = img

    X_train[n+1,:,:,:] = fliplr(img)

    X_train[n+2,:,:,:] = flipud(img)

    X_train[n+3,:,:,:] = fliplr(flipud(img))

    n += 4
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
inputs = Input((int(IMG_HEIGHT/SCALE_FACTOR), int(IMG_WIDTH/SCALE_FACTOR), IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)

c1 = Dropout(0.2) (c1)

c1 = BatchNormalization()(c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.3) (c2)

c2 = BatchNormalization()(c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.3) (c3)

c3 = BatchNormalization()(c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

c3 = BatchNormalization()(c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.4) (c4)

c4 = BatchNormalization()(c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

c4 = BatchNormalization()(c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.5) (c5)

c5 = BatchNormalization()(c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

u6 = BatchNormalization()(u6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.4) (c6)

c6 = BatchNormalization()(c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

c6 = BatchNormalization()(c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

u7 = BatchNormalization()(u7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.3) (c7)

c7 = BatchNormalization()(c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

u8 = BatchNormalization()(u8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.3) (c8)

c8 = BatchNormalization()(c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

c8 = BatchNormalization()(c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

u9 = BatchNormalization()(u9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.2) (c9)

c9 = BatchNormalization()(c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

c9 = BatchNormalization()(c9)



outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])

adamcustom = Adam(lr=0.0003)

model.compile(optimizer=adamcustom, loss=['binary_crossentropy'], metrics=[dice_coef])

model.summary()
# Fit model

earlystopper = EarlyStopping(patience=8, verbose=1)

#checkpointer = ModelCheckpoint('../input/severstal-steel-defect-detection/severstal_unetmodel', verbose=1, save_best_only=True)

results = model.fit(X_train, Y_train, validation_split=0.1, shuffle=True, batch_size=30, epochs=100,

                    callbacks=[earlystopper])
#Get test data

testfiles = next(os.walk(TEST_IMAGE_PATH))[2]

X_test = np.zeros((len(testfiles), int(IMG_HEIGHT/SCALE_FACTOR), int(IMG_WIDTH/SCALE_FACTOR), int(IMG_CHANNELS)), dtype=np.uint8)



n = 0

for file in testfiles:

  

    #img = color.rgb2gray(io.imread(TEST_IMAGE_PATH + file))[:,:]

    img = imread(TEST_IMAGE_PATH + file)

    img = resize(img, (IMG_HEIGHT/SCALE_FACTOR, IMG_WIDTH/SCALE_FACTOR, IMG_CHANNELS), mode='constant', preserve_range=True)

    X_test[n,:,:,:] = img

    n += 1
#predict results

#model = load_model('severstal_unetmodel')

#create a new dataframe for the predictions



# submission_list = []

# for n in range(len(testfiles)):

#     preds_test = model.predict(X_test[n:n+1,:,:,:])



    # Threshold predictions

#     preds_test_t = (preds_test > 0.5).astype(np.uint8)

#     #find which mask has the most non-zero data

#     for m in range(4):

#         img = preds_test_t[0,:,:,m]

#         #resize mask to original size

#         img = resize(img, (256, 1600), mode='constant', preserve_range=True)

#         #encode results and put in dataframe

#         encoded_entry = mask2rle(img)

#         row = [testfiles[n] + '_' + str(m+1), encoded_entry]

#         submission_list.append(row)

#     if n % 100 == 0:

#         print('Saving entry', n)

    

# #create submission file

# submission_data = pd.DataFrame(submission_list, columns=['ImageId_ClassId','EncodedPixels'])

# submission_data.to_csv('./submission.csv', index=False)



submission_list = []

for n in range(len(testfiles)):

    preds_test = model.predict(X_test[n:n+1,:,:,:])



    # Threshold predictions

    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    #find which mask has the most non-zero data

    for m in range(4):

        img = preds_test_t[0,:,:,m]

        #resize mask to original size

        img = resize(img, (256, 1600), mode='constant', preserve_range=True)

        #encode results and put in dataframe

        encoded_entry = mask2rle(img)

        row = [testfiles[n] + '_' + str(m+1), encoded_entry]

        submission_list.append(row)

    if n % 1000 == 0:

        print('Saving entry', n)

    

#create submission file

submission_data = pd.DataFrame(submission_list, columns=['ImageId_ClassId','EncodedPixels'])

submission_data = submission_data.fillna('')

submission_data.to_csv('./submission.csv', index=False)
