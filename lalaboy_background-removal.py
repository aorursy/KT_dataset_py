import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

import keras.backend as K

import os

import cv2
df = pd.read_csv('/kaggle/input/carvana-image-masking-challenge/train_masks.csv')
df.head(2)
df2 = pd.read_csv('/kaggle/input/carvana-image-masking-challenge/metadata.csv')
df2.head(2)
from PIL import Image



x = []

y = []

rows, columns = 256, 256

channels = 3

num_imgs = 200

for i,row in df.iterrows():

    if i >= num_imgs:

        break

    img_id = str(row['img'].split('.')[0])

    img = cv2.imread('../input/carvana-image-masking-challenge/train/{}.jpg'.format(img_id))

    img = cv2.resize(img, (rows, columns))

    img = np.asarray(img).astype('float32')

    img /= 255.0

    

    file = '../input/carvana-image-masking-challenge/train_masks/' + img_id + '_mask.gif'

    new_file = file.split('/')[-1][:-4] + '.jpg'

    Image.open(file).convert('RGB').save(new_file)

    mask = cv2.imread(new_file, 0)

    mask = cv2.resize(mask, (rows, columns))

    mask = np.asarray(mask).astype('float32')

    mask /= 255.0

    

    x.append(img)

    y.append(mask)
for i in (0, 8, 100, 108):

    plt.figure(figsize = (10, 6))

    plt.subplot(121)

    plt.imshow(x[i])



    plt.subplot(122)

    plt.imshow(y[i])



    plt.show()
x = np.array(x)

y = np.array(y)

y = y.reshape(num_imgs, 256, 256, 1)

x.shape, y.shape
from keras.losses import binary_crossentropy



def dice_coef(y_true, y_pred, smooth = 1):

    intersection = K.sum(y_true * y_pred)

    return (2 * intersection + smooth) / ((K.sum(y_true) + K.sum(y_pred)) + smooth)

def bce_dice_loss(y_true, y_pred):

    return .5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
#def dice_coef(y_true, y_pred, smooth=1):

#  intersection = K.sum(y_true * y_pred, axis=[1,2,3])

#  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])

#  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

#  return dice
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model 

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten,concatenate, Dense

from keras.callbacks import ModelCheckpoint

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
x = np.array(x)

y = np.array(y)

y = y.reshape(num_imgs, 256, 256, 1)

x.shape, y.shape
from keras.losses import binary_crossentropy



def dice_coef(y_true, y_pred, smooth = 1):

    intersection = K.sum(y_true * y_pred)

    return (2 * intersection + smooth) / ((K.sum(y_true) + K.sum(y_pred)) + smooth)

def bce_dice_loss(y_true, y_pred):

    return .5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
def unet(input_size = (rows, columns, 3)):

    input_ = Input(input_size)

    conv0 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_)

    conv0 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)

    pool0 = MaxPooling2D(pool_size = (2, 2))(conv0)

    

    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)

    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)

    

    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    

    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2, 2))(conv5))

    merge6 = concatenate([conv4, up6], axis = 3)

    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

                                                                                                  

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    

    up10 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))



    conv10 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)

    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)



    model = Model(input = input_, outputs = conv11)

    

    model.compile(optimizer = Adam(lr = 0.0001), loss = bce_dice_loss, metrics = [dice_coef])

    

    return model
model = unet()

model.summary()
# Fit the model to the training set and compute dice coefficient at each validation set

model_save = ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')



model_run = model.fit(x, y, epochs = 10,  callbacks=[model_save])



model.save("saved_model.h5")
pd.DataFrame(model_run.history)[['dice_coef']].plot()
y_pred= model.predict(x)
plt.imshow(y_pred[0].reshape(256, 256)) 
plt.imshow(y[0].reshape(256, 256))
x = []

y = []

for directory, _, filenames in os.walk('../input/carvana-image-masking-challenge/train'):

    for filename in filenames[:50]:

        file = os.path.join(directory, filename)

        img = cv2.imread(file, cv2.IMREAD_COLOR)

        if img is not None:

            img = np.asarray(img) / 255

            x.append(img)
for directory, _, filenames in os.walk('../input/carvana-image-masking-challenge/train_masks'):

    for filename in filenames[:50]:

        file = os.path.join(directory, filename)

        img = cv2.imread(file, cv2.IMREAD_COLOR)

        if img is not None:

            img = np.asarray(img) / 255

            y.append(img)
from PIL import Image

%matplotlib inline



image = cv2.imread('../input/carvana-image-masking-challenge/train/00087a6bd4dc_01.jpg')

mask = Image.open('../input/carvana-image-masking-challenge/train_masks/00087a6bd4dc_01_mask.gif')



image2 = cv2.imread('../input/carvana-image-masking-challenge/train/02159e548029_04.jpg')

mask2 = Image.open('../input/carvana-image-masking-challenge/train_masks/02159e548029_04_mask.gif')



image3 = cv2.imread('../input/carvana-image-masking-challenge/train/0495dcf27283_12.jpg')

mask3 = Image.open('../input/carvana-image-masking-challenge/train_masks/0495dcf27283_12_mask.gif')



plt.figure(figsize = (10, 6))

plt.subplot(231)

plt.imshow(image)



plt.subplot(234)

plt.imshow(mask)



plt.subplot(232)

plt.imshow(image2)





plt.subplot(235)

plt.imshow(mask2)



plt.subplot(233)

plt.imshow(image3)



plt.subplot(236)

plt.imshow(mask3)



plt.show()
datagen = ImageDataGenerator(

    rotation_range = 180,

    shear_range = .2,

    zoom_range = .2,

    horizontal_flip = True,

    rescale = 1/255,

    fill_mode = 'nearest')



test_datagen = ImageDataGenerator(rescale = 1/255)

if K.image_data_format() == 'channels_first':

    input_shape = (3, 150, 150)

else:

    input_shape = (150, 150, 3)

    

train = datagen.flow_from_directory('intel-image-classification/seg_train/seg_train/',

                                    target_size = (150, 150),

                                    batch_size = 64,

                                    class_mode = 'categorical')

test = test_datagen.flow_from_directory('intel-image-classification/seg_test/seg_test',

                                        target_size = (150, 150),

                                        batch_size = 64,

                                        class_mode = 'categorical')
from keras.models import Model, load_model

from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, Add, concatenate

from keras.models import model_from_json

from keras.regularizers import l2



class Model_Tiramisu():

    

    def __init__(self):

        

        self.model = self.build_model()

    def fit_model(self, x, y, epochs = 10,validation_split = .2, workers = 6): 

        self.model.fit(x, y, epochs =epochs,

                      validation_split =validation_split ,

                      workers = 6)

        

    def build_model(self):

        

        layer_per_block = [3, 3, 3, 3, 3, 4, 12, 10, 7, 5, 4]

        

        model = self.build_graph(layer_per_block, n_pool=5, growth_rate=16)    



        model.compile(loss = [bce_dice_loss], optimizer = 'adam', metrics = [dice_coef])



        return model



    def denseBlock(self, t, nb_layers):

        for _ in range(nb_layers):

            tmp = t

            t = BatchNormalization(axis = -1,

                                    gamma_regularizer = l2(0.0001),

                                    beta_regularizer = l2(0.0001))(t)



            t = Activation('relu')(t)

            t = Conv2D(16, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', data_format = 'channels_last')(t)

            t = Dropout(0.2)(t)

            t = concatenate([t, tmp])

        return t



    def transitionDown(self, t, nb_features):

        t = BatchNormalization(axis = -1,

                               gamma_regularizer = l2(0.0001),

                               beta_regularizer = l2(0.0001))(t)

        t = Activation('relu')(t)

        t = Conv2D(nb_features,

                   kernel_size = (1, 1),

                   padding = 'same',

                   kernel_initializer = 'he_uniform',

                   data_format='channels_last')(t)

        t = Dropout(0.2)(t)

        t = MaxPooling2D(pool_size = (2, 2),

                         strides = 2,

                         padding = 'same',

                         data_format = 'channels_last')(t)

        

        return t



    def build_graph(self, layer_per_block, n_pool = 3, growth_rate = 16):

        input_layer = Input(shape = (256, 256, 3))

        t = Conv2D(48, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(input_layer)



        #dense block

        nb_features = 48

        skip_connections = []

        for i in range(n_pool):

            t = self.denseBlock(t, layer_per_block[i])

            skip_connections.append(t)

            nb_features += growth_rate * layer_per_block[i]

            t = self.transitionDown(t, nb_features)



        t = self.denseBlock(t, layer_per_block[n_pool]) # bottle neck



        skip_connections = skip_connections[::-1] #subvert the array



        for i in range(n_pool):

            keep_nb_features = growth_rate * layer_per_block[n_pool + i]

            t = Conv2DTranspose(keep_nb_features, strides=2, kernel_size=(3, 3), padding='same', data_format='channels_last')(t) # transition Up

            t = concatenate([t, skip_connections[i]])



            t = self.denseBlock(t, layer_per_block[n_pool + i + 1])



        t = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)

        output_layer = Activation('softmax')(t)



        return Model(inputs = input_layer, outputs = output_layer)
x.shape,y.shape 
model = Model_Tiramisu()

# Fit the model to the training set and compute dice coefficient at each validation set

model_save = ModelCheckpoint('best_model.hdf5',

                             save_best_only = True,

                             monitor = 'bce_dice_loss',

                             mode = 'min')



model_run = model.fit_model(x, y,

                      epochs = 10,

                      validation_split = .2,

                      #callbacks=[model_save],

                      workers = 6)



model.save("saved_model.h5")