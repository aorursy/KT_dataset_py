import cv2

import numpy as np

from glob import glob

from tqdm import tqdm

import pandas as pd

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

print(os.listdir())
seed = 42

version=117

np.random.seed(seed)

df = pd.read_csv('../input/train.csv')

df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.split('_', expand=True)

df['path'] = '../input/train_images/' + df['ImageId']



# Dataframe containing only defects

edf = df[df['EncodedPixels'].notnull()]

edf.head()
def get_binary_mask(encoded_pixels, input_shape=(256, 1600)):

    # run length encoding (rle)

    height, width = input_shape[:2]

    mask= np.zeros(width*height).astype(np.uint8)

    array = np.asarray([int(x) for x in encoded_pixels.split()])

    

    starts = array[0::2]

    lengths = array[1::2]



    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape(width, height).T



def mask_to_image(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def plot_img(path, pixels=None, title=None, figsize=(10, 10)):

    base_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if pixels is not None:

        mask = get_binary_mask(pixels, (256, 1600))

        base_img[mask==1] = 255

    fig = plt.figure(figsize=figsize)

    plt.title(title)

    plt.imshow(base_img, cmap='gray'); plt.show()

    return base_img
# Visualize images with their masks

for sample in edf.sample(5).itertuples(index=False):

    _ = plot_img(sample.path, sample.EncodedPixels, title=sample.ImageId_ClassId)
# How many defects per image

edf.groupby(['ImageId']).agg({'ClassId': 'count'})['ClassId'].value_counts().plot(kind='bar')
# How many defects by classId

edf['ClassId'].value_counts().plot(kind='bar')
!mkdir augmented_img

print(os.listdir('.'))
def upsample(ids, edf):

    if ids in ['1', '4']:

        edf2 = edf[edf['ClassId'] == ids].sample(200, random_state=seed)

    else:

        edf2 = edf[edf['ClassId'] == ids]

    

    def hflip(img):

        # Horizontal flip

        return np.flip(img, 0)



    def vflip(img):

        # Vertical flip

        return np.flip(img, 1)



    augmented_dict = {'EncodedPixels': [], 'ImageId': [], 'ClassId': [], 'path': [], 'ImageId_ClassId': []}



    def update_dict(adict, pixels, image, cid, path):

        adict['EncodedPixels'].append(pixels)

        adict['ImageId'].append(image)

        adict['ClassId'].append(cid)

        adict['path'].append(path)

        adict['ImageId_ClassId'].append(image + '_' + cid)

        return augmented_dict



    for img2 in edf2.itertuples(index=False):

        img = cv2.imread(img2.path, cv2.IMREAD_GRAYSCALE)

        mask = get_binary_mask(img2.EncodedPixels)



        himg = hflip(img)

        hmask = mask_to_image(hflip(mask))

        vimg = vflip(img)

        vmask = mask_to_image(vflip(mask))



        himg_path = 'augmented_img/h' + img2.ImageId.replace('jpg', 'png')

        vimg_path = 'augmented_img/v' + img2.ImageId.replace('jpg', 'png')



        augmented_dict = update_dict(augmented_dict, hmask, 'h'+img2.ImageId, img2.ClassId, path=himg_path)

        augmented_dict = update_dict(augmented_dict, vmask, 'v'+img2.ImageId, img2.ClassId, path=vimg_path)



        cv2.imwrite(f'{himg_path}', himg)

        cv2.imwrite(f'{vimg_path}', vimg)



    edf = pd.concat([edf, pd.DataFrame(augmented_dict)])

    return edf



edf = upsample('2', edf).reset_index(drop=True)
# Add masks to the dataframe

edf.loc[:, 'mask'] = edf['EncodedPixels'].map(get_binary_mask)



# Get a count of the number of pixels in each mask

edf.loc[:, 'pixel_count'] = edf['mask'].map(lambda x: x.ravel().sum())

edf.head()
edf.groupby(['ClassId']).agg({'pixel_count': 'sum'}).plot.pie(subplots=True, figsize=(5, 10))
# Get largest defect size

largest = edf.loc[edf['pixel_count'].idxmax()]

# Get smallest defect size

smallest = edf.loc[edf['pixel_count'].idxmin()]
smask = plot_img(smallest['path'], smallest['EncodedPixels'], figsize=(15,10), title=smallest['ImageId_ClassId'])

print(f'Smallest mask: {smask[smask == 255].shape}')
lmask = plot_img(largest['path'], largest['EncodedPixels'], figsize=(15,10), title=largest['ImageId_ClassId'])
mask = smallest

kernel = np.ones((3, 3),np.uint8)

so_mask = mask_to_image(cv2.morphologyEx(mask['mask'], cv2.MORPH_OPEN, kernel))

new_img = plot_img(mask['path'], so_mask, figsize=(15,10), title=mask['ImageId_ClassId'])
img1 = new_img[210:230, 1090:1120]

print(img1[img1 == 255].shape)

plt.imshow(img1); plt.show()
edf.groupby(['ClassId']).agg({'pixel_count': 'describe'})
plabel = edf['pixel_count'].sum()

total_pixels = 256 * 1600 * edf.shape[0]

print(f'Positive label: {plabel}, total pixels: {total_pixels} positive_pct: {plabel / total_pixels}')
def balance_df(df, sample=None):

    if sample:

        ndf = pd.DataFrame()

        for i in ['1', '2', '3', '4']:

            tdf = df[df['ClassId'] == i].sample(sample, random_state=seed)

            ndf = pd.concat([ndf, tdf])

        return ndf

    else:

        minor_edf = df[df['ClassId'].isin(['1', '2', '4'])]

        sampled_df = df[df['ClassId'] == '3'].sample(replace=False, n=5000, random_state=seed)

        

        return pd.concat([minor_edf, sampled_df])



# Some models can preform better when the distribution for ClassIds are about the same

# However I am not doing that here, it's an option

tdf = balance_df(edf).reset_index(drop=True)

tdf['ClassId'].value_counts()
train, valid = train_test_split(tdf.index, test_size=0.15, random_state=seed)

print('train:{}\tvalid:{}\ttdf:{}\tdf:{}'.format(train.shape, valid.shape, tdf.shape, edf.shape))

print(tdf.loc[train]['ClassId'].value_counts())

print(tdf.loc[valid]['ClassId'].value_counts())
#https://github.com/qubvel/segmentation_models

! pip install segmentation-models
import warnings

warnings.filterwarnings("ignore")

import keras

import tensorflow as tf

from keras import backend as K

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras.losses import binary_crossentropy

from keras.regularizers import l2, l1

from keras.layers import GaussianNoise

from keras import optimizers

from keras.initializers import he_normal, glorot_uniform

from keras import initializers

from keras.models import Sequential, Model, load_model, Input

from keras.layers import Dropout, Flatten, Dense, Conv2D, Add, BatchNormalization, Activation, MaxPooling2D, Layer, InputSpec, Conv2DTranspose, UpSampling2D, concatenate, GlobalAveragePooling2D, ZeroPadding2D

from keras.layers.core import SpatialDropout2D, Activation

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
img_height, img_width = 256, 1600

batch = 256

BATCH_SIZE = 5

EPOCHS = 15

N_CLASS = 4



strided = True

LEARNING_RATE = 1e-5
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)



def tversky_loss(y_true, y_pred, beta=0.75):

    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)

    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)



def binary_dice_coef_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)



def binary_tversky_loss(y_true, y_pred):

    return tversky_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)



def img_to_dataframe(img, depth=4):

    df = pd.DataFrame({i: img[..., i].ravel() for i in range(depth)})

    return df
gamma = 2

inverse_gamma = 1.0 / gamma

look_up_table = np.array([((i/255.0) ** inverse_gamma) * 255.0 for i in np.arange(0,256,1)]).astype("uint8")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))



def contrast_enhancement(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img[:,:,0] = clahe.apply(img[:,:,0])

    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

    return img



def gamma_correction(img):

    return cv2.LUT(img.astype('uint8'), look_up_table)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, mode='fit', batch_size=BATCH_SIZE, n_channels=1, reshape=None, n_classes=4, shuffle=True, strided=True):

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.strided = strided

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        

        # Pass a sliding window through the image to create 256 x 256

        if self.strided:

            Xn = []

            for jj in X:

                for i in range(0, 224 * 7, 224):

                    nm = jj[:, i:i+batch]

                    Xn.append(nm)

            X = np.array(Xn)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            

            # Pass a sliding window through the mask to create 256 x 256

            if self.strided:

                yn = []

                for jj in y:

                    for i in range(0, 224 * 7, 224):

                        nm = jj[:, i:i+batch]

                        yn.append(nm)

                y = np.array(yn)            

            return X, y

        else:

            return X



    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        # Initialization

        X = np.zeros((self.batch_size, *self.reshape, self.n_channels))

        

        for i, ID in enumerate(list_IDs_batch):

            path = self.df['path'].iloc[ID]

            img = self.__load_grayscale(path)

            X[i, :] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        

        y = np.zeros((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.df[self.df['ImageId'] == im_name][['ClassId', 'EncodedPixels']]

            

            masks = np.zeros((*self.reshape, 4))

            for cid, pixel in image_df.itertuples(index=False):

                masks[:, :, int(cid) - 1] = get_binary_mask(pixel)

            y[i, :] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path)

        img = contrast_enhancement(img)

        img = gamma_correction(img)

        img = img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0

        img = np.expand_dims(img, axis=-1)



        return img

salt = int(str(datetime.now().timestamp())[-1])



sample=tdf.sample(1, random_state=seed+salt).index[0]

print(f'sample:{sample} seed:{seed} salt:{salt}')

train_sample = [sample]

X=0; Y=0;

train_generator = DataGenerator(train_sample, df=tdf, reshape=(img_height, img_width), strided=strided, batch_size=1, shuffle=False, n_channels=1)



for x, y in train_generator:

    X=x; 

    Y=y;

    break

print(f'x: {X.shape}\ty: {Y.shape}')
fi = plt.figure(figsize=(20,20))

print('ORIGINAL IMAGE')

plt.imshow(cv2.imread(tdf.loc[sample]['path'], cv2.IMREAD_GRAYSCALE), cmap='gray')
print('BASE IMAGE WITH MASK')

base = plot_img(tdf.loc[sample]['path'], tdf.loc[sample]['EncodedPixels'], title=tdf.loc[sample]['ImageId'], figsize=(20, 15))
img = cv2.imread(tdf.loc[sample]['path'])

fig = plt.figure(figsize=(20, 20))

row=1; col=7;

print('BASE IMAGE STRIDES')

for idx, i in enumerate(range(0, 224 * 7, 224)):

    fig.add_subplot(row, col, idx+1)

    nm = img[:, i:i+batch]

    plt.imshow(nm)
fig = plt.figure(figsize=(20, 20))

row=1; col=7;

print('GENERATED IMAGE STRIDES')

if X.shape[-1] == 1:

    Xr = np.repeat(X, repeats=3, axis=-1)

else:

    Xr = X

for idx, i in enumerate(Xr):

    fig.add_subplot(row, col, idx+1)

    plt.imshow(i)
mask = get_binary_mask(tdf.loc[train_sample]['EncodedPixels'].values[0], input_shape=(256, 1600))

fig = plt.figure(figsize=(20, 20))

row=1; col=7;

print('BASE MASK STRIDED')

for idx, i in enumerate(range(0, 224 * 7, 224)):

    fig.add_subplot(row, col, idx+1)

    nm = mask[:, i:i+batch]

    plt.imshow(nm)
fig = plt.figure(figsize=(20, 20))

row=1; col=7;

channel = int(tdf.loc[train_sample]['ClassId'])-1

Yr = np.repeat(np.expand_dims(Y[..., channel], axis=-1), repeats=3, axis=-1)

print('GENERATED MASK STRIDED')

for idx, i in enumerate(Yr):

    fig.add_subplot(row, col, idx+1)

    plt.imshow(i * 255)
from segmentation_models import Unet

keras.backend.set_image_data_format('channels_last')



model = Unet('resnet34', input_shape=(256, 256, 1), classes=4, activation='sigmoid', encoder_weights=None)

model.compile(optimizer=Adam(LEARNING_RATE), loss=binary_tversky_loss, metrics=[dice_coef])

model.summary()
model_path = f'UNet_ResNet34_{LEARNING_RATE}_{EPOCHS}_{version}.h5'



train_generator = DataGenerator(train, df=tdf, reshape=(img_height, 1600), strided=strided, n_channels=1)

val_generator = DataGenerator(valid, df=tdf, reshape=(img_height, 1600), strided=strided, n_channels=1)



callbacks = [

    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-10, verbose=1),

    ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

]

history = model.fit_generator(train_generator, validation_data=val_generator, callbacks=callbacks, use_multiprocessing=False, epochs=EPOCHS)
hdf = pd.DataFrame(history.history)

hdf.to_csv('v{}_history.df'.format(version), index=False)

hdf[['loss', 'val_loss']].plot(grid=True, figsize=(15, 3), title='Loss Graphs')

hdf[['dice_coef', 'val_dice_coef']].plot(grid=True, figsize=(15, 3), title='Val Graphs')
model = load_model(model_path, custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss, 'binary_tversky_loss': binary_tversky_loss})
def test_model(model, img, df):

    test_df = df[df['ImageId'] == img]

    

    if test_df.shape[0] == 1:

        test_df = test_df.iloc[0]

    else:

        test_df = test_df.iloc[1]

    

    img = cv2.imread(test_df['path'], cv2.IMREAD_GRAYSCALE)

    

    pred = np.zeros((256, 1600, 4))

    for i in range(0, 224*7, 224):

        nm = img[:, i:i+batch]

        batch_pred = model.predict(np.expand_dims(np.expand_dims(nm, axis=-1), axis=0))[0]

        pred[:, i:i+batch, :] = np.maximum(pred[:, i:i+batch, :], batch_pred)



    pred = np.where(pred > 0.4, 1, 0)

        

    y_true = get_binary_mask(test_df['EncodedPixels'], (256, 1600))

    final_mask = mask_to_image(pred[..., int(test_df['ClassId']) - 1])

    

    plot_img(test_df['path'], test_df['EncodedPixels'], title='true_' + test_df['ImageId_ClassId'])

    plot_img(test_df['path'], final_mask, title='pred_' + test_df['ImageId_ClassId'])

    

test_img = tdf.sample(10)['ImageId'].tolist()

for i in test_img: test_model(model, i, tdf)
!rm -rf augmented_img