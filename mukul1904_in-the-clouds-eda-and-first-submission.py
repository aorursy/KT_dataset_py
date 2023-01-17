!pip install segmentation-models
import numpy as np

import pandas as pd

import os

import cv2

import collections

import time 

import sys

import tqdm

from multiprocessing import  Pool

from PIL import Image

from functools import partial

import warnings

warnings.filterwarnings("ignore")



train_on_gpu = True



# Visualisation libs

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



from albumentations import (

    PadIfNeeded,

    HorizontalFlip,

    VerticalFlip,    

    CenterCrop,    

    Crop,

    Compose,

    Transpose,

    RandomRotate90,

    ElasticTransform,

    GridDistortion, 

    OpticalDistortion,

    RandomSizedCrop,

    OneOf,

    CLAHE,

    RandomBrightnessContrast,    

    RandomGamma    

)

import segmentation_models as sm



import keras

from keras import optimizers

from keras import backend as K

from keras.models import Model

from keras.losses import binary_crossentropy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
# used from notebooks mentioned in the end

def np_resize(img, input_shape):

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask = np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#Parameters

LEARNING_RATE = 2e-3

HEIGHT = 512

WIDTH = 512

CHANNELS = 3

N_CLASSES = 4

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5

BACKBONE = 'resnet50'

BATCH_SIZE=16

SEED=42
path = '../input/understanding_cloud_organization'

os.listdir(path)
train = pd.read_csv(f'{path}/train.csv')

train.shape
sample_submission = pd.read_csv(f'{path}/sample_submission.csv')

sample_submission.shape
train.info()
train.head()
def parallelize_dataframe(df, func, n_cores=4):

    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)

    df = pd.concat(pool.map(func, df_split))

    pool.close()

    pool.join()

    return df



def add_features(df):

    df['name'] = df['Image_Label'].apply(lambda x: x.split('_')[0])

    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])

    df['has_null_encoded_pxs'] = df['EncodedPixels'].isnull()

    return df



train = parallelize_dataframe(train, add_features)



train.head(10)
f, ax = plt.subplots(figsize=(14, 6))

ax = sns.countplot(y="has_null_encoded_pxs", data=train)

plt.show()
train[ train.has_null_encoded_pxs == False ].groupby('name').size().value_counts().reset_index().rename(columns={ 'index': 'Number of masks', '0': 'Image Count'})
mask_count_df = train.groupby('name').agg(np.sum).reset_index()

mask_count_df.sort_values('has_null_encoded_pxs', ascending=False, inplace=True)

train_idx, val_idx = train_test_split(mask_count_df.index, test_size=0.2, random_state=42)

print(train_idx.shape, val_idx.shape)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path=f'{path}/train_images',

                 batch_size=BATCH_SIZE, dim=(1400, 2100), n_channels=CHANNELS, reshape=(HEIGHT, WIDTH), 

                 n_classes=N_CLASSES, random_state=SEED, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        if self.reshape is None:

            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        else:

            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['name'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = cv2.imread(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        if self.reshape is None:

            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        else:

            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['name'].iloc[ID]

            image_df = self.target_df[self.target_df['name'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y
train_generator = DataGenerator(train_idx, df=mask_count_df, target_df=train)

 

valid_generator = DataGenerator(val_idx, df=mask_count_df, target_df=train)
model = sm.Unet(

           encoder_name=BACKBONE, 

           classes=N_CLASSES,

           activation='sigmoid',

           input_shape=(HEIGHT, WIDTH, CHANNELS))



model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE), loss=binary_crossentropy, metrics=[dice_coef])



model.summary()
# using it from dimitreoliveira's kernel

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)
history = model.fit_generator(generator=train_generator,

                              validation_data=valid_generator,

                              epochs=20,

                              callbacks=[es, rlrop],

                              verbose=2).history
history
fig, ax = plt.subplots(1, 1, sharex='col', figsize=(20, 12))



ax.plot(history['dice_coef'], label='Train Dice coefficient')

ax.plot(history['val_dice_coef'], label='Validation Dice coefficient')

ax.legend(loc='best')

ax.set_title('Dice coefficient')



plt.xlabel('Epochs')

plt.show()
sample_submission = parallelize_dataframe(sample_submission, add_features)

sample_submission.head()
test = pd.DataFrame(sample_submission['name'].unique(), columns=['name'])

test.head()
test_df = []



for i in range(0, test.shape[0], 500):

    batch_idx = list(range(i, min(test.shape[0], i + 500)))

    test_generator = DataGenerator(

                     batch_idx,

                     df=test,

                     target_df=sample_submission,

                     batch_size=1,

                     reshape=(HEIGHT, WIDTH),

                     dim=(350, 525),

                     n_channels=CHANNELS,

                     n_classes=N_CLASSES,

                     random_state=SEED,

                     base_path=f'{path}/test_images',

                     mode='predict',

                     shuffle=False)



    batch_pred_masks = model.predict_generator(test_generator)



    for j, b in enumerate(batch_idx):

        filename = test['name'].iloc[b]

        image_df = sample_submission[sample_submission['name'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)

        

submission_ready = pd.concat(test_df)
submission_ready.head()
submission = submission_ready[['Image_Label', 'EncodedPixels']]

submission.to_csv('submission.csv', index=False)

print('Added submission file')

submission.head()
# !pip install Kaggle

# !kaggle competitions submit -c understanding_cloud_organization -f submission.csv -m "Message"