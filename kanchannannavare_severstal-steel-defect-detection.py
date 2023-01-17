# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import json



import cv2



import keras

from keras import backend as K

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Model

from keras.layers import Input, BatchNormalization, Activation, Dropout

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import Callback, ModelCheckpoint

from keras.losses import binary_crossentropy



import matplotlib.pyplot as plt



import numpy as np

import pandas as pd



from tqdm import tqdm



from sklearn.model_selection import train_test_split



import multiprocessing
train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train['ImageId'] = train['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train['ClassId'] = train['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train['hasMask'] = ~ train['EncodedPixels'].isna()



print(train.shape)

train.head()
mask_count = train.groupby('ImageId').agg(np.sum).reset_index()



mask_count.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count.shape)



mask_count.head()
sub = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')



sub['ImageId'] = sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0])



test_imgs = pd.DataFrame(sub['ImageId'].unique(), columns=['ImageId'])
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



def rle2mask(mask_rle, shape=(256,1600)):

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
def build_masks(rles, input_shape):

    depth = len(rles)

    height, width = input_shape

    masks = np.zeros((height, width, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, (width, height))

    

    return masks



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
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

test_imgs_folder = '../input/severstal-steel-defect-detection/test_images/'

train_imgs_folder = '../input/severstal-steel-defect-detection/train_images/'
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/severstal-steel-defect-detection/train_images',

                 batch_size=16, dim=(256, 1600), n_channels=1,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

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

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_grayscale(img_path)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#         img = cv2.resize(img, self.dim)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img

    

#     def __init__(self, images_list=None, folder_imgs=train_imgs_folder, 

#                  batch_size=32, shuffle=True, augmentation=None,

#                  resized_height=224, resized_width=224, num_channels=3):

#         self.batch_size = batch_size

#         self.shuffle = shuffle

#         self.augmentation = augmentation

#         if images_list is None:

#             self.images_list = os.listdir(folder_imgs)

#         else:

#             self.images_list = deepcopy(images_list)

#         self.folder_imgs = folder_imgs

#         self.len = len(self.images_list) // self.batch_size

#         self.resized_height = resized_height

#         self.resized_width = resized_width

#         self.num_channels = num_channels

#         self.num_classes = 4

#         self.is_test = not 'train' in folder_imgs

#         if not shuffle and not self.is_test:

#             self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len*self.batch_size]]



#     def __len__(self):

#         return self.len

    

#     def on_epoch_start(self):

#         if self.shuffle:

#             random.shuffle(self.images_list)



#     def __getitem__(self, idx):

#         current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]

#         X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))

#         y = np.empty((self.batch_size, self.num_classes))



#         for i, image_name in enumerate(current_batch):

#             path = os.path.join(self.folder_imgs, image_name)

#             img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)

#             if not self.augmentation is None:

#                 augmented = self.augmentation(image=img)

#                 img = augmented['image']

#             X[i, :, :, :] = img/255.0

#             if not self.is_test:

#                 y[i, :] = img_2_ohe_vector[image_name]

#         return X, y



#     def get_labels(self):

#         if self.shuffle:

#             images_current = self.images_list[:self.len*self.batch_size]

#             labels = [img_2_ohe_vector[img] for img in images_current]

#         else:

#             labels = self.labels

#         return np.array(labels)
BATCH_SIZE = 16



train_idx, val_idx = train_test_split(

    mask_count.index, random_state=2019, test_size=0.15

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count,

    target_df=train,

    batch_size=BATCH_SIZE, 

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    df=mask_count,

    target_df=train,

    batch_size=BATCH_SIZE, 

    n_classes=4

)
from keras.applications.resnet50 import ResNet50



def build_model(input_shape):

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)



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

    

        # Load ImageNet weights

#     if encoder == 'vgg19':

#         pretrained_model = VGG19(include_top=False)

#     else:

# #         pretrained_model = VGG16(include_top=False)

    

#     pretrained_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

#     for layer, pretrained_layer in zip(

#             model.layers[2:], pretrained_model.layers[2:]):

#         layer.set_weights(pretrained_layer.get_weights())

#     imagenet_weights = pretrained_model.layers[1].get_weights()

#     init_bias = imagenet_weights[1]

#     init_kernel = np.average(imagenet_weights[0], axis=2)

#     init_kernel = np.reshape(

#         init_kernel,

#         (init_kernel.shape[0],

#             init_kernel.shape[1],

#             1,

#             init_kernel.shape[2]))

#     init_kernel = np.dstack([init_kernel] * img_input.shape.as_list()[-1])  # input image is grayscale

#     model.layers[1].set_weights([init_kernel, init_bias])

    

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model
def resnet50(input_shape):

    inputs = Input(input_shape)

    

    model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model
# https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):

    """Function to add 2 convolutional layers with the parameters passed to it"""

    # first layer

    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\

              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    

    # second layer

    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\

              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)

    if batchnorm:

        x = BatchNormalization()(x)

    x = Activation('relu')(x)

    

    return x

  

def get_unet(input_shape, n_filters = 16, dropout = 0.1, batchnorm = True):

    # Contracting Path

    inputs = Input(input_shape)

    

    c1 = conv2d_block(inputs, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    p1 = MaxPooling2D((2, 2))(c1)

    p1 = Dropout(dropout)(p1)

    

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    p2 = MaxPooling2D((2, 2))(c2)

    p2 = Dropout(dropout)(p2)

    

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    p3 = MaxPooling2D((2, 2))(c3)

    p3 = Dropout(dropout)(p3)

    

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    p4 = MaxPooling2D((2, 2))(c4)

    p4 = Dropout(dropout)(p4)

    

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    

    # Expansive Path

    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)

    u6 = concatenate([u6, c4])

    u6 = Dropout(dropout)(u6)

    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)

    u7 = concatenate([u7, c3])

    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)

    u8 = concatenate([u8, c2])

    u8 = Dropout(dropout)(u8)

    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)

    u9 = concatenate([u9, c1])

    u9 = Dropout(dropout)(u9)

    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model
model = build_model((256, 1600, 1))

# model = resnet50((224, 224, 3))

# model = get_unet((256, 1600, 1))

model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_dice_coef', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    train_generator,

    validation_data=val_generator,

    callbacks=[checkpoint],

#     use_multiprocessing=True,

#     workers=multiprocessing.cpu_count(),

#     workers=1,

    epochs=8

)
model.load_weights('model.h5')

test_df = []



for i in range(0, test_imgs.shape[0], 500):

    batch_idx = list(

        range(i, min(test_imgs.shape[0], i + 500))

    )

    

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/severstal-steel-defect-detection/test_images',

        target_df=sub,

        batch_size=1,

        n_classes=4

    )

    

    batch_pred_masks = model.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )

    

    for j, b in tqdm(enumerate(batch_idx)):

        filename = test_imgs['ImageId'].iloc[b]

        image_df = sub[sub['ImageId'] == filename].copy()

        

        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks)

        

        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)
test_df = pd.concat(test_df)

test_df.drop(columns='ImageId', inplace=True)

test_df.to_csv('submission.csv', index=False)

test_df.head()