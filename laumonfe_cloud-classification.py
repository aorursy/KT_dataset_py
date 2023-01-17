

# for data handling 

import cv2

import glob 

import math

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

from keras.preprocessing.image import save_img

from sklearn.model_selection import train_test_split



# for neural network

import tensorflow as tf

from tensorflow import keras

from keras.models import Model, Sequential # Models

#from keras.layers import LSTM, Dense, RepeatVector,TimeDistributed, Input # Layers

#from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape # Layers



# for data visualization 

from keras.callbacks import ModelCheckpoint, History

from keras_tqdm import TQDMCallback, TQDMNotebookCallback

# Set paths to data

trainPath = '../input/understanding_cloud_organization/train_images/'

testPath = '../input/understanding_cloud_organization/test_images/'



# Get the csv with labels and the images for training and testing

dataCSV = pd.read_csv('../input/understanding_cloud_organization/train.csv')

trainData= sorted(glob.glob(trainPath+ '*.jpg'))

testData = sorted(glob.glob(testPath + '*.jpg'))
# Exploring the dataset

print('There are', len(trainData), ' images in the train set and',len(testData),'images in the test set ')
# Visualizing the dataset

labels = 'Train', 'Test'

sizes = [len(trainData), len(testData)]

explode = (0, 0.1)



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Train and Test Sets')



plt.show()
# Lets look at the CSV data

dataCSV.head()
# split column

splitCSV = dataCSV["Image_Label"].str.split("_", n = 1, expand = True)

# add new columns to dataCSV

dataCSV['Image'] = splitCSV[0]

dataCSV['Label'] = splitCSV[1]



# check the result

dataCSV.head()

fish = dataCSV[dataCSV['Label'] == 'Fish'].EncodedPixels.count()

flower = dataCSV[dataCSV['Label'] == 'Flower'].EncodedPixels.count()

gravel = dataCSV[dataCSV['Label'] == 'Gravel'].EncodedPixels.count()

sugar = dataCSV[dataCSV['Label'] == 'Sugar'].EncodedPixels.count()



print('There are {} fish clouds'.format(fish))

print('There are {} flower clouds'.format(flower))

print('There are {} gravel clouds'.format(gravel))

print('There are {} sugar clouds'.format(sugar))

# plotting a pie chart

labels = 'Fish', 'Flower', 'Gravel', 'Sugar'

sizes = [fish, flower, gravel, sugar]



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Cloud Types')



plt.show()
# Run-length decoder

def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):

    '''

    Decode rle encoded mask.

    

    :param mask_rle: run-length as string formatted (start length)

    :param shape: (height, width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape, order='F')
#Plot 4 images of each class

import os

path = '../input/understanding_cloud_organization'

os.listdir(path)



# Images known to contain the classes

fish_imgs = ['0766057', '0770f53', '07c5a0d' , '07c5fc9']

flower_imgs = ['0741fda', '0745d08', '07551f3', '0761274']

gravel_imgs = ['077bd40', '080c004' , '08329b8', '0862841']

sugar_imgs = ['0770f53', '0778609', '0799206',  '079aef5']



columns = 4

rows = 4

fig, ax = plt.subplots(rows, columns, figsize=(18, 13))

ax[0, 0].set_title('Fish', fontsize=20)

ax[0, 1].set_title('Flower', fontsize=20)

ax[0, 2].set_title('Gravel', fontsize=20)

ax[0, 3].set_title('Sugar', fontsize=20)

for i in range(len(fish_imgs)):

    fish_img = plt.imread(f"{path}/train_images/{fish_imgs[i]}.jpg")

    ax[i, 0].imshow(fish_img)

    image_label = f'{fish_imgs[i]}.jpg_Fish'

    mask_rle = dataCSV.loc[dataCSV['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    mask = rle_decode(mask_rle)

    ax[i, 0].imshow(mask, alpha=0.5, cmap='gray')

    

    flower_img = plt.imread(f"{path}/train_images/{flower_imgs[i]}.jpg")

    ax[i, 1].imshow(flower_img)

    image_label = f'{flower_imgs[i]}.jpg_Flower'

    mask_rle = dataCSV.loc[dataCSV['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    mask = rle_decode(mask_rle)

    ax[i, 1].imshow(mask, alpha=0.5, cmap='gray')

    

    gravel_img = plt.imread(f"{path}/train_images/{gravel_imgs[i]}.jpg")

    ax[i, 2].imshow(gravel_img)

    image_label = f'{gravel_imgs[i]}.jpg_Gravel'

    mask_rle = dataCSV.loc[dataCSV['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    mask = rle_decode(mask_rle)

    ax[i, 2].imshow(mask, alpha=0.5, cmap='gray')

    

    sugar_img = plt.imread(f"{path}/train_images/{sugar_imgs[i]}.jpg")

    ax[i, 3].imshow(sugar_img)

    image_label = f'{sugar_imgs[i]}.jpg_Sugar'

    mask_rle = dataCSV.loc[dataCSV['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    mask = rle_decode(mask_rle)

    ax[i, 3].imshow(mask, alpha=0.5, cmap='gray')

plt.show()
train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



print(train_df.shape)

train_df.head()
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head()
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/train_images',

                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,

                 augment=False, n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.augment = augment

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()

        np.random.seed(self.random_state)



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

            

            if self.augment:

                X, y = self.__augment_batch(X, y)

            

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

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)

            

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

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img

    

    def __random_transform(self, img, masks):

        composition = albu.Compose([

            albu.HorizontalFlip(),

            albu.VerticalFlip(),

            albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15)

        ])

        

        composed = composition(image=img, mask=masks)

        aug_img = composed['image']

        aug_masks = composed['mask']

        

        return aug_img, aug_masks

    

    def __augment_batch(self, img_batch, masks_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ], masks_batch[i, ] = self.__random_transform(

                img_batch[i, ], masks_batch[i, ])

        

        return img_batch, masks_batch
def unet(input_shape):

    """

    This is the old model. Best LB is ~0.5

    """

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2), padding='same') (c1)



    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2), padding='same') (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2), padding='same') (c3)



    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

    p4 = MaxPooling2D((2, 2), padding='same') (c4)



    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

    p5 = MaxPooling2D((2, 2), padding='same') (c5)



    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)



    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

    u6 = concatenate([u6, c5])

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)



    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

    u71 = concatenate([u71, c4])

    c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)

    c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)



    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)



    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)



    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)



    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    

    return model
BATCH_SIZE = 32



train_idx, val_idx = train_test_split(

    mask_count_df.index, random_state=2019, test_size=0.2

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    reshape=(320, 480),

    augment=True,

    n_channels=3,

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE, 

    reshape=(320, 480),

    augment=False,

    n_channels=3,

    n_classes=4

)
!pip install segmentation-models --quiet
model = sm.Unet(

    'resnet34', 

    classes=4,

    input_shape=(320, 480, 3),

    activation='sigmoid'

)

model.compile(optimizer=Nadam(lr=0.0002), loss=bce_dice_loss, metrics=[dice_coef])

model.summary()