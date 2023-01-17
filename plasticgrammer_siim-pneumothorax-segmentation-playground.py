import gc

import glob

import os, sys

import json

import pprint

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from joblib import Parallel, delayed

from tqdm import tqdm_notebook as tqdm

from PIL import Image

from skimage.transform import resize



sns.set_style('darkgrid')

sns.set_palette('bone')



print(os.listdir("../input"))
sys.path.insert(0, '/kaggle/input/siim-acr-pneumothorax-segmentation')

from mask_functions import rle2mask, mask2rle
path = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax'



rle = pd.read_csv(f'{path}/train-rle.csv')

rle.columns = ['ImageId','EncodedPixels']

print(rle.shape)

rle.head(10)
rle['ImageId'].nunique()
counts = rle['EncodedPixels'].map(lambda x: 'none' if x != ' -1' else 'pneumothorax').value_counts()

counts.plot.barh(figsize=(10,3))

counts
train_fns = sorted(glob.glob(f'{path}/dicom-images-train/*/*/*.dcm'))

test_fns = sorted(glob.glob(f'{path}/dicom-images-test/*/*/*.dcm'))



print('The training set contains {} files.'.format(len(train_fns)))

print('The test set contains {} files.'.format(len(test_fns)))
unique_id = rle['ImageId'].unique()

file_id = list(map(lambda x: x.split('/')[-1][:-4], train_fns))

print('only ImageId data:',len([f for f in unique_id if f not in file_id]))

print('only image files:', len([f for f in file_id if f not in unique_id]))

del unique_id, file_id
import pydicom as dicom



d = dicom.read_file(train_fns[0])

print(d)
print(d.pixel_array.shape)

print(d.pixel_array)
fig, axis = plt.subplots(2, 3, figsize=(14,9))

axis = np.ravel(axis)



for ax, n in zip(axis, [2,3,4,5,6,7]):

    d = dicom.read_file(train_fns[n])

    ax.axis('off')

    ax.imshow(d.pixel_array, cmap='bone')

    ax.imshow(rle2mask(rle.iloc[n, 1], d.Columns, d.Rows), alpha=0.3)
fig, axis = plt.subplots(1, 2, figsize=(10,4))



pa128 = resize(d.pixel_array, (128,128), mode='constant', preserve_range=True)

axis[0].imshow(d.pixel_array, cmap='bone')

axis[1].imshow(pa128, cmap='bone')
_='''

dim = 1024



train_img = {}

targetId = rle['ImageId'].values

for n in tqdm(range(len(train_fns))):

    d = dicom.read_file(train_fns[n])

    imageId = train_fns[n].split('/')[-1][:-4]

    if imageId in targetId:

        train_img[imageId] = [d.pixel_array / 256]

        #train_img[imageId] = [resize(d.pixel_array, (dim, dim), mode='constant', preserve_range=True).tolist()]

        train_img[imageId] = np.resize(train_img[imageId], (dim, dim, 1))



X_train = [train_img[i] for i in rle['ImageId']]

Y_train = [np.expand_dims(rle2mask(px, 1024, 1024).T, axis=2) if px != ' -1' else np.zeros((1024, 1024, 1)) for px in rle['EncodedPixels']]

#Y_train = [np.ravel(rle2mask(px, 1024, 1024)) if px != ' -1' else np.zeros(dim * dim) for px in rle['EncodedPixels']][:1000]

#Y_train = (rle.iloc[:,1] != ' -1').astype(int)

'''
#del train_fns

#del rle, train_img

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
import keras

from keras import optimizers

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Permute, concatenate

from keras.utils import np_utils



# 2D blocks

def conv2D_block(inputs, filters, activation, padding, batchnorm=False):

    conv = Conv2D(filters, 3, activation=activation, padding=padding)(inputs)

    if batchnorm:

        conv = BatchNormalization()(conv)

    conv = Conv2D(filters, 3, activation=activation, padding=padding)(conv) 

    if batchnorm:

        conv = BatchNormalization()(conv)

    return conv



def conv2D_maxpool_block(inputs, filters, activation, padding, batchnorm=False):

    conv = conv2D_block(inputs, filters, activation, padding)

    pool = MaxPooling2D()(conv)

    return pool, conv



def upsamp_conv2D_block(conv_prev, conv_direct, filters, activation, padding, batchnorm=False):

    up = UpSampling2D()(conv_prev)

    conc = concatenate([up, conv_direct])

    cm = conv2D_block(conc, filters, activation, padding, batchnorm)

    return cm



def build_unet2D(inp_shape=(None, None, 1)):

    inputs = Input(shape=inp_shape)

    

    # Three conv pool blocks

    p1, c1 = conv2D_maxpool_block(inputs, 16, 'relu', 'same', False)

    p2, c2 = conv2D_maxpool_block(p1, 32, 'relu', 'same', False)

    p3, c3 = conv2D_maxpool_block(p2, 64, 'relu', 'same', False)

    p4, c4 = conv2D_maxpool_block(p3, 128, 'relu', 'same', False)



    # Fourth conv -- lowest point

    c5 = conv2D_block(p4, 256, 'relu', 'same', False)



    # Three upsampling conv blocks

    cm2 = upsamp_conv2D_block(c5, c4, 128, 'relu', 'same', False)

    cm3 = upsamp_conv2D_block(cm2, c3, 64, 'relu', 'same', False)

    cm4 = upsamp_conv2D_block(cm3, c2, 32, 'relu', 'same', False)

    cm5 = upsamp_conv2D_block(cm4, c1, 16, 'relu', 'same', False)



    # Output

    predictions = Conv2D(1, 1, activation='sigmoid')(cm5)

    model = Model(inputs, predictions)



    return model



model = build_unet2D(inp_shape=(1024, 1024, 1))

model.summary()
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(1024,1024), n_channels=1,

                 shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size, *self.dim, self.n_channels))



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            X[i,] = np.expand_dims(dicom.read_file(ID).pixel_array, axis=2)

            

            stripped_id = ID.split('/')[-1][:-4]

            if self.labels is not None:

                rle = self.labels.get(stripped_id)



                if rle is None:

                    y[i,] = np.zeros((1024, 1024, 1))

                else:

                    if len(rle) == 1:

                        y[i,] = np.expand_dims(rle2mask(rle[0], self.dim[0], self.dim[1]).T, axis=2)

                    else: 

                        y[i,] = np.zeros((1024, 1024, 1))

                        for x in rle:

                            y[i,] =  y[i,] + np.expand_dims(rle2mask(x, 1024, 1024).T, axis=2)



        return X, y
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)
from collections import defaultdict



d = defaultdict(list)

for image_id, r in zip(rle['ImageId'], rle['EncodedPixels']):

    d[image_id].append(r)

annotated = {k: v for k, v in d.items() if v[0] != ' -1'}
params = {'dim': (1024, 1024),

          'batch_size': 8,

          'n_channels': 1,

          'shuffle': True}



# Generators

training_generator = DataGenerator(train_fns[0:8000], annotated, **params)

validation_generator = DataGenerator(train_fns[8000:10712], annotated, **params) 



# Compile model

optimizer = optimizers.Adam(lr = 0.001, epsilon = 0.1)

loss = dice_coef_loss

metrics= [dice_coef]

model.compile(optimizer=optimizer, loss=loss, metrics= metrics)



# Fit model

model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose=1)
#loss_and_metrics = model.evaluate(np.array(X_train), np.array(Y_train), batch_size=128)

#print(loss_and_metrics)
#del X_train, Y_train 

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
test_generator = DataGenerator(test_fns, None, **params)

pred = model.predict_generator(test_generator, verbose=1)
rles = []

for p in tqdm(pred):

    im = np.asarray(p)

    rles.append(mask2rle(im, 1024, 1024))
file_id = list(map(lambda x: x.split('/')[-1][:-4], test_fns))

submission = pd.DataFrame({

    "ImageId": file_id,

    "EncodedPixels": rles})

submission.loc[submission.EncodedPixels=='', 'EncodedPixels'] = '-1'

submission.to_csv("submission.csv", index=False, header=True)