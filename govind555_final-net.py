# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend as K

#K.image_data_format('tf')

# from keras.models import Model

# from keras.layers import Input, ZeroPadding2D, Concatenate, Add, Flatten

# from keras.layers.core import Dropout, Activation 

# from keras.layers.convolutional import UpSampling2D, Conv2D

# from keras.layers.pooling import AveragePooling2D, MaxPooling2D

# from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split
train_dir = '../input/severstal-steel-defect-detection/' 

train_image_dir = os.path.join(train_dir, 'train_images') 

train_df = pd.read_csv(os.path.join(train_dir, 'train.csv')).fillna(-1)

train_df.head()
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

# lets create a dict with class id and encoded pixels and group all the defaults per image

train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)

grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)

train_df.head()
# from https://www.kaggle.com/robertkag/rle-to-mask-converter

def rle_to_mask(rle_string,height,width):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters: 

    rleString (str): Description of arg1 

    height (int): height of the mask

    width (int): width of the mask 



    Returns: 

    numpy.array: numpy array of the mask

    '''

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]

        rlePairs = np.array(rleNumbers).reshape(-1,2)

        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img



def mask_to_rle(mask):

    '''

    Convert a mask into RLE

    

    Parameters: 

    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background



    Returns: 

    sring: run length encoding 

    '''

    pixels= mask.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)    
# network configuration parameters

# original image is 1600x256, so we will resize it

img_w = 800 # resized weidth

img_h = 256 # resized height

batch_size = 10

epochs = 5

# batch size for training unet

k_size = 3 # kernel size 3x3

val_size = .20 # split
from keras.preprocessing.image import ImageDataGenerator



img_datagen = ImageDataGenerator(

    vertical_flip=True)

    #horizontal_flip=True)



mask_datagen = ImageDataGenerator(

    vertical_flip=True)

    #horizontal_flip=True)





class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_ids, labels, image_dir, mode='train', batch_size=32,

                 img_h=256, img_w=512, shuffle=True):

        

        self.mode = mode

        self.list_ids = list_ids

        self.labels = labels

        self.image_dir = image_dir

        self.batch_size = batch_size

        self.img_h = img_h

        self.img_w = img_w

        self.shuffle = shuffle

        self.on_epoch_end()

    

    def __len__(self):

        'denotes the number of batches per epoch'

        return int(np.floor(len(self.list_ids)) / self.batch_size)

    

    def __getitem__(self, index):

        'generate one batch of data'

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get list of IDs

        list_ids_temp = [self.list_ids[k] for k in indexes]

        # generate data

        X, y = self.__data_generation(list_ids_temp)

        # return data 

        return X, y

    

    def on_epoch_end(self):

        'update ended after each epoch'

        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    def __data_generation(self, list_ids_temp):

        'generate data containing batch_size samples'

        X = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

        

        for idx, id in enumerate(list_ids_temp):

            file_path =  os.path.join(self.image_dir, id)

            image = cv2.imread(file_path, 0)

            image_resized = cv2.resize(image, (self.img_w, self.img_h))

            image_resized = np.array(image_resized, dtype=np.float64)

            # standardization of the image

            image_resized -= image_resized.mean()

            image_resized /= image_resized.std()

            

            mask = np.empty((img_h, img_w, 4))

            

            for idm, image_class in enumerate(['1','2','3','4']):

                rle = self.labels.get(id + '_' + image_class)

                # if there is no mask create empty mask

                if rle is None:

                    class_mask = np.zeros((1600, 256))

                else:

                    class_mask = rle_to_mask(rle, width=1600, height=256)

             

                class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))

                mask[...,idm] = class_mask_resized

            

            seed = 10

#             if self.mode == 'train':

#                 if np.random.randn() > 0:

#                     image_resized = img_datagen.random_transform(image_resized, seed = seed) 

#                     mask = mask_datagen.random_transform(mask, seed = seed)

#                     #params = datagen.get_random_transform(image_resized.shape, seed = seed) 

#                     #image_resized = datagen.apply_transform(image_resized, params)

#                     #mask = datagen.apply_transform(mask, params)

            

            X[idx,] = np.expand_dims(image_resized, axis=2)

            y[idx,] = mask

        

        # normalize Y

        y = (y > 0).astype(int)

            

        return X, y
# create a dict of all the masks

masks = {}

for index, row in train_df[train_df['EncodedPixels']!=-1].iterrows():

    masks[row['ImageId_ClassId']] = row['EncodedPixels']
train_image_ids = train_df['ImageId'].unique()

X_train, X_val = train_test_split(train_image_ids, test_size=val_size, random_state=42)
params = {'img_h': img_h,

          'img_w': img_w,

          'image_dir': train_image_dir,

          'batch_size': batch_size,

          'shuffle': True}



# Get Generators

training_generator = DataGenerator(X_train, masks, mode='train', **params)

validation_generator = DataGenerator(X_val, masks, mode='validation', **params)



# training_generator = DataGenerator(X_train, masks, **params)

# validation_generator = DataGenerator(X_val, masks, **params)



x, y = training_generator.__getitem__(0)

print(x.shape, y.shape)
import keras

from keras.layers import Input, Conv2D, Add, Flatten, UpSampling2D, Concatenate, Activation,BatchNormalization

def resnet_block(inputs,  filters):

    x = conv_block(inputs, filters)

    x = conv_block(x, filters)

    x = Add()([inputs, x])

    return x



def downsample(input, filters, kernel=3, stride=2):    

    x = conv_block(input,filters, stride=2)

    return x



def upsample_conv(input, filters, rate=2):    

    # downsample

    x = BatchNormalization()(input)

    x = Activation('relu')(x)    

    x = UpSampling2D((rate, rate))(x)

    #x = keras.layers.Conv2D(filters, 3, strides=(1,1), padding='same', 

    #                        use_bias=False,kernel_initializer='he_uniform')(x)

    return x



def concat_conv2x2(b,e2,e4,filters):

    x = Concatenate(axis=-1)([b, e2])

    y = Concatenate(axis=-1)([x, e4])

    out = conv_block(y, filters, stride = 1, kernel = 3)

    return out



def concat_conv(a,b,filters):

    x = Concatenate()([a, b])

    out = conv_block(x, filters, stride = 1, kernel = 1)

    return out



def upsample_concat(input, y, filters):

    x = upsample_conv(input, filters, rate=2)

    #return layers.Concatenate()([x, y])

    #return layers.Add()([x, y])

    return concat_conv(x,y,filters)



def upsample_downsample_add(b, e4, e2, filters, rate=4):

    #print('This before b is', b.shape)

    #print('Filters', filters)

    b = upsample_conv(b,filters)

    e2 = conv_block(e2, filters, stride = 4, kernel = 5)

    #print('This upscaled b is', b.shape)

    #print('This e2 is', e2.shape)

    #print('This e4 is', e4.shape)

    

    #add =  layers.Add()([b, y])

    #return layers.Concatenate()([add, x])

    #return layers.Add()([add, x])

    return concat_conv2x2(b,e2,e4,filters)

    



def upsample_upsample_add(d2, e2, e4, filters, rate=2):  

    #print('This d2 before is', d2.shape)

    #print('This e2 before is', e2.shape)

    d2 = upsample_conv(d2, filters)

    e4 = upsample_conv(e4, filters, rate=4)

    #print('This d2 is', d2.shape)

    #print('This e2 is', e2.shape)

    #print('This e4 is', e4.shape)

    #add =  layers.Add()([x, y])

    #return layers.Concatenate()([add, input])

    #return layers.Add()([add, input])

    return concat_conv2x2(d2, e2, e4,filters)

    





def conv_block(input, filters, stride = 1, kernel = 3): 

    x = BatchNormalization()(input)

    x = Activation('relu')(x)    

    x = Conv2D(filters, kernel, strides = stride, padding='same', use_bias=False, 

                      kernel_initializer='he_normal')(x)

    return x



def first_block(input, filters, stride = 1, kernel=3):

    #print(input.shape)

    #x = Conv2D(filters=filters, kernel_size=kernel_size , padding='same', strides=stride)(inputs)

    x = Conv2D(filters, kernel, strides=stride, padding='same', use_bias=False,kernel_initializer='he_normal')(input)

   # x = layers.Conv2D(filters, kernel, strides = (stride, stride), padding='same', use_bias=False, 

   #                   kernel_initializer='he_normal')(input)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    return x

    

    
import keras

def ResUNet(img_h, img_w):

    f = [16, 32, 64, 128, 256, 512]

    inputs = Input((img_h, img_w, 1))

    

    ## Encoder

    e0 = inputs

    e1_ = first_block(e0, f[0])

    e1 = resnet_block(e1_, f[0])

    

    e2_ = downsample(e1_, f[1])

    e2 = resnet_block(e2_, f[1])

    #print('Shape of e2 is', e2)

    

    e3_ = downsample(e2, f[2])

    e3 = resnet_block(e3_, f[2])

    

    e4_ = downsample(e3, f[3])

    e4 = resnet_block(e4_, f[3])

    

    e5 = downsample(e4, f[4])

    e5_ = resnet_block(e5, f[4])

    

    

    ## Bridge

    b0 = conv_block(e5_, f[5])

    b1 = conv_block(b0, f[5])

    #b1 = b0

    

    ## Decoder

    u1 = upsample_downsample_add(b1, e4, e2, f[4])

    d1 = resnet_block(u1, f[4])

    

    u2 = upsample_concat(d1, e3, f[3])

    d2 = resnet_block(u2, f[3])

    

    u3 = upsample_upsample_add(d2, e2, e4, f[2])

    d3 = resnet_block(u3, f[2])

    

    u4 = upsample_concat(d3, e1,f[1])

    d4 = resnet_block(u4, f[1])

    

    outputs = BatchNormalization()(d4)

    outputs = Conv2D(filters = 4, strides = (1, 1), kernel_size = 3, padding="same", activation='sigmoid')(outputs)

    

    #outputs = Activation('sigmoid')(outputs)

    model = Model(inputs, outputs)

    return model
from keras.models import Model

model = ResUNet(img_h=img_h, img_w=img_w)
#model.summary()
import tensorflow as tf

import tensorflow

from keras.layers import Flatten

def tversky(y_true, y_pred, smooth=1e-6):

    y_true_pos = tf.keras.layers.Flatten()(y_true)

    y_pred_pos = Flatten()(y_pred)

    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)

    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))

    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)

    alpha = 0.7

    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)



def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true,y_pred)



def focal_tversky_loss(y_true,y_pred):

    pt_1 = tversky(y_true, y_pred)

    gamma = 0.75

    return tf.keras.backend.pow((1-pt_1), gamma)
from keras.callbacks import ModelCheckpoint

adam = keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)

model.compile(optimizer='adam', loss=focal_tversky_loss, metrics=[tversky])

#filepath = "/kaggle/output/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='val_tversky', verbose=1, period=1, save_best_only=False, mode='max')
model_path = '../input/pretrainedmodel/bestmodel2.hdf5'

load_pretrained_model = True

if load_pretrained_model:

    try:

        model.load_weights(model_path)

        print('Model loaded successfully')

    except OSError:

        print('Error in model loading')
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,)
model.save_weights("/kaggle/working/bestmodel3.hdf5")
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,)
model.save_weights("/kaggle/working/bestmodel4.hdf5")
!ls /kaggle/working
history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, verbose=1,)
model.save_weights("/kaggle/working/bestmodel3.hdf5")
# return tensor in the right shape for prediction 

def get_test_tensor(img_dir, img_h, img_w, channels=1):



    X = np.empty((1, img_h, img_w, channels))

    # Store sample

    image = cv2.imread(img_dir, 0)

    image_resized = cv2.resize(image, (img_w, img_h))

    image_resized = np.array(image_resized, dtype=np.float64)

    # normalize image

    image_resized -= image_resized.mean()

    image_resized /= image_resized.std()

    

    X[0,] = np.expand_dims(image_resized, axis=2)



    return X
from skimage import morphology



def remove_small_regions(img, size):

    """Morphologically removes small (less than size) connected regions of 0s or 1s."""

    img = morphology.remove_small_objects(img, size)

    img = morphology.remove_small_holes(img, size)

    return img
import glob

# get all files using glob

test_files = [f for f in glob.glob('../input/severstal-steel-defect-detection/test_images/' + "*.jpg", recursive=True)]

submission = []



# a function to apply all the processing steps necessery to each of the individual masks

def process_pred_mask(pred_mask):

    

    pred_mask = cv2.resize(pred_mask.astype('float32'),(1600, 256))

    pred_mask = (pred_mask > .5).astype(int)

    pred_mask = remove_small_regions(pred_mask, 0.02 * np.prod(512)) * 255

    pred_mask = mask_to_rle(pred_mask)

    

    return pred_mask



# loop over all the test images

for f in test_files:

    # get test tensor, output is in shape: (1, 256, 512, 3)

    test = get_test_tensor(f, img_h, img_w) 

    # get prediction, output is in shape: (1, 256, 512, 4)

    pred_masks = model.predict(test) 

    # get a list of masks with shape: 256, 512

    pred_masks = [pred_masks[0][...,i] for i in range(0,4)]

    # apply all the processing steps to each of the mask

    pred_masks = [process_pred_mask(pred_mask) for pred_mask in pred_masks]

    # get our image id

    id = f.split('/')[-1]

    # create ImageId_ClassId and get the EncodedPixels for the class ID, and append to our submissions list

    [submission.append((id+'_%s' % (k+1), pred_mask)) for k, pred_mask in enumerate(pred_masks)]
# convert to a csv

submission_df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])

# check out some predictions and see if RLE looks ok

submission_df[ submission_df['EncodedPixels'] != ''].head()
# take a look at our submission 

submission_df.head()
# write it out

submission_df.to_csv('./submission.csv', index=False)