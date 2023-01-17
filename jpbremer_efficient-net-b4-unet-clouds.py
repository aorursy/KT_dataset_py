import os

import json



import albumentations as albu

import cv2

import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.losses import binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from skimage.exposure import adjust_gamma

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.layers import LeakyReLU

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization

from keras.layers import Conv2D, Concatenate, MaxPooling2D

from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras import optimizers

from keras.legacy import interfaces

from keras.utils.generic_utils import get_custom_objects



from keras.engine.topology import Input

from keras.engine.training import Model

from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose

from keras.layers.core import Activation, SpatialDropout2D

from keras.layers.merge import concatenate

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add

from keras.regularizers import l2

from keras.layers.core import Dense, Lambda

from keras.layers.merge import concatenate, add

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
!pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591
def post_process(probability, threshold, min_size):

    """

    Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored

    """

    

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((350, 525), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num



def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

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



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

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
sub_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')

sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/understanding_cloud_organization/train_images',

                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None, gamma=None,

                 augment=False, n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.gamma = gamma

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

           # print(" ids +", i, ID)

            #if(self.modi is None):

            im_name = self.df['ImageId'].loc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            # Adjust gamma

            if self.gamma is not None:

                img = adjust_gamma(img, gamma=self.gamma)

            

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

       # print(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.

        #plt.imshow(img)

        #plt.show()



        return img

    

    def __random_transform(self, img, masks):

        composition = albu.Compose([

            albu.HorizontalFlip(),

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
def H(lst, name, use_gn=False):

    if use_gn:

        norm = GroupNormalization(groups=1, name=name+'_gn')

    else:

        norm = BatchNormalization(name=name+'_bn')

    

    x = concatenate(lst)

    num_filters = int(x.shape.as_list()[-1]/2)

    

    x = Conv2D(num_filters, (2, 2), padding='same', name=name)(x)

    x = norm(x)

    x = LeakyReLU(alpha=0.1, name=name+'_activation')(x)

    

    return x



def U(x, use_gn=False):

    if use_gn:

        norm = GroupNormalization(groups=1)

    else:

        norm = BatchNormalization()

    

    num_filters = int(x.shape.as_list()[-1]/2)

    

    x = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(x)

    x = norm(x)

    x = LeakyReLU(alpha=0.1)(x)

    

    return x
import efficientnet.keras as efn 

def EfficientUNet(input_shape):

    backbone = efn.EfficientNetB4(

        weights=None,

        include_top=False,

        input_shape=input_shape

    )

    

    input = backbone.input

    x00 = backbone.input  # (256, 512, 3)

    x10 = backbone.get_layer('stem_activation').output  # (128, 256, 4)

    x20 = backbone.get_layer('block2d_add').output  # (64, 128, 32)

    x30 = backbone.get_layer('block3d_add').output  # (32, 64, 56)

    x40 = backbone.get_layer('block5f_add').output  # (16, 32, 160)

    x50 = backbone.get_layer('block7b_add').output  # (8, 16, 448)

    

    x01 = H([x00, U(x10)], 'X01')

    x11 = H([x10, U(x20)], 'X11')

    x21 = H([x20, U(x30)], 'X21')

    x31 = H([x30, U(x40)], 'X31')

    x41 = H([x40, U(x50)], 'X41')

    

    x02 = H([x00, x01, U(x11)], 'X02')

    x12 = H([x11, U(x21)], 'X12')

    x22 = H([x21, U(x31)], 'X22')

    x32 = H([x31, U(x41)], 'X32')

    

    x03 = H([x00, x01, x02, U(x12)], 'X03')

    x13 = H([x12, U(x22)], 'X13')

    x23 = H([x22, U(x32)], 'X23')

    

    x04 = H([x00, x01, x02, x03, U(x13)], 'X04')

    x14 = H([x13, U(x23)], 'X14')

    

    x05 = H([x00, x01, x02, x03, x04, U(x14)], 'X05')

    

    x_out = Concatenate(name='bridge')([x01, x02, x03, x04, x05])

    x_out = Conv2D(4, (3,3), padding="same", name='final_output', activation="sigmoid")(x_out)

    

    return Model(inputs=input, outputs=x_out)



model = EfficientUNet((320, 480 ,3))



model.summary()
model.load_weights("../input/cloudmodels/EfficientNetB4.h5")
minsizes = [20000 ,20000, 22500, 10000]

sigmoid = lambda x: 1 / (1 + np.exp(-x))
test_df = []



subsize = 500



for i in range(0, test_imgs.shape[0], subsize):

    batch_idx = list(

        range(i, min(test_imgs.shape[0], i + subsize))

    )



    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        dim=(350, 525),

        reshape=(320, 480),

        gamma=0.8,

        n_channels=3,

        base_path='../input/understanding_cloud_organization/test_images',

        target_df=sub_df,

        batch_size=1,

        n_classes=4

    )



    batch_pred_masks = model.predict_generator(

        test_generator, 

        workers=1,

        verbose=1

    )



    for j, b in enumerate(batch_idx):

        filename = test_imgs['ImageId'].iloc[b]

        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ]

        

        pred_masks = cv2.resize(pred_masks, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)



        arrt = np.array([])

        for t in range(4):

            a, num_predict = post_process(sigmoid(pred_masks[:, :, t]), 0.6, minsizes[t])

            

            

            if(arrt.shape == (0,)):

                arrt = a.reshape(350, 525, 1)

            else:

                arrt = np.append(arrt, a.reshape(350, 525, 1), axis = 2)

           

        pred_rles = build_rles(arrt, reshape=(350, 525))



        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)



sub_df = pd.concat(test_df)
sub_df = sub_df[['Image_Label', 'EncodedPixels']]

sub_df.to_csv('submission.csv', index=False)

display(sub_df.head(10))