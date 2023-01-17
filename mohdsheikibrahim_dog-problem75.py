# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.applications.vgg19 import VGG19

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import load_img

#from keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import preprocess_input

from keras.preprocessing.image import img_to_array

import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

import sys

import bcolz

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/labels.csv')

df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(10)

df_test.head(10)
%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from mpl_toolkits.axes_grid1 import ImageGrid
train_files = glob('../input/dog-breed-identification/train/*.jpg')

test_files = glob('../input/dog-breed-identification/test/*.jpg')
plt.imshow(plt.imread(train_files[100]))
targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)
!ls ../input/keras-pretrained-models/
im_size = 400
y_train = []

y_val = []

x_train_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)

x_val_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
i = 0 

for f, breed in tqdm(df_train.values):

    # load an image from file

    image = load_img('../input/dog-breed-identification/train/{}.jpg'.format(f), target_size=(im_size, im_size))

    image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    label = one_hot_labels[i]

    if random.randint(1,101) < 80: 

        x_train_raw.append(image)

        y_train.append(label)

    else:

        x_val_raw.append(image)

        y_val.append(label)

    i += 1
y_train_raw = np.array(y_train, np.uint8)

y_val_raw = np.array(y_val, np.uint8)

del(y_train,y_val)

import gc

gc.collect()
print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_val_raw.shape)

print(y_val_raw.shape)
def plotImages( images_arr, n_images=4):

    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.set_xticks(())

        ax.set_yticks(())

    plt.tight_layout()

plotImages(x_train_raw[0:16,]/255.)
num_class = y_train_raw.shape[1]
# Create the base pre-trained model

#base_model = VGG19(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))

base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))

base_model.summary()
len(base_model.layers)
layers_to_remove = 0

if layers_to_remove >0:

    for i in range(0,layers_to_remove):

        base_model.layers.pop()

    base_model.summary()
fine_tuning_layers = 0

layers_to_freeze = len(base_model.layers) - fine_tuning_layers

print(layers_to_freeze)



for layer in base_model.layers[0:layers_to_freeze]:

    layer.trainable = False
# Add a new top layer

x = base_model.layers[layers_to_freeze-1+fine_tuning_layers].output

x = BatchNormalization()(x)



#x = MaxPooling2D((4, 4), strides=(4, 4), padding='same')(x)

#x = Conv2D(128, (2, 2), padding='same')(x)

#x = Dropout(0.01)(x)

#x = BatchNormalization()(x)



#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

#x = Conv2D(128, (2, 2), padding='same')(x)

#x = Dropout(0.01)(x)

#x = BatchNormalization()(x)

#x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

#x = Conv2D(128, (2, 2), padding='same')(x)

#x = Dropout(0.01)(x)

#x = BatchNormalization()(x)

#x = Flatten()(x)



x = GlobalAveragePooling2D()(x)



x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.2)(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.15)(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.1)(x)



predictions = Dense(num_class, activation='softmax')(x)



# This is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

model.summary()
batch_size = 32

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1),checkpoint]
K.get_value(model.optimizer.lr)
K.set_value(model.optimizer.lr, 0.001)

import copy
datagen = ImageDataGenerator()

temp = np.zeros((16,im_size,im_size,3),dtype=np.float32)

image_to_test = 10

for i in range(16):

    if random.randint(1,101) < 50: 

        flip_horizontal = True

    else:

        flip_horizontal = False

    if random.randint(1,101) < 50: 

        flip_vertical = True

    else:

        flip_vertical = False

    tx = im_size*random.randint(1,15)/100.0

    ty = im_size*random.randint(1,15)/100.0

    shear = random.randint(1,10)/100.0

    zx = random.randint(80,120)/100.0

    zy = random.randint(80,120)/100.0

    brightness = random.randint(1,2)/100.0

    channel_shift_intensity = random.randint(1,10)/100.0

    

    temp[i] = datagen.apply_transform(x_train_raw[image_to_test],{

        'tx':tx,

        'ty':ty,

        'shear':shear,

        'zx':zx,

        'zy':zy,

        'flip_horizontal':flip_horizontal,

        'flip_vertical':flip_vertical,

        #'brightness':brightness,

        #'channel_shift_intensity':channel_shift_intensity

        })

plotImages(temp[0:16,]/255.0)
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, data, labels, im_size = 224, batch_size=32, shuffle=True, data_augment = False, test = False):

        'Initialization'

        self.batch_size = batch_size

        self.list_IDs = np.arange(0,data.shape[0])

        self.shuffle = shuffle

        if self.shuffle == True:

            np.random.shuffle(self.list_IDs)        

        self.data = data

        self.data_augment = data_augment

        self.test = test

        if self.test == False:

            self.labels = labels

        self.on_epoch_end()





    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]



        # Generate data

        if self.test == False:

            X, y = self.__data_generation(indexes)

            return preprocess_input(X), y

        else:

            X = self.__data_generation(indexes)

            return preprocess_input(X)



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        if self.shuffle == True:

            np.random.shuffle(self.list_IDs)        



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.zeros((list_IDs_temp.shape[0],im_size,im_size,3), dtype=np.float32)

        if self.test == False:

            y = np.zeros((list_IDs_temp.shape[0],self.labels.shape[1]), dtype=np.uint8)



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            

            if self.data_augment == True:

                if random.randint(1,101) < 50: 

                    flip_horizontal = True

                else:

                    flip_horizontal = False

                if random.randint(1,101) < 50: 

                    flip_vertical = True

                else:

                    flip_vertical = False

                tx = im_size*random.randint(1,2)/100.0

                ty = im_size*random.randint(1,2)/100.0

                shear = random.randint(1,10)/100.0

                zx = random.randint(80,120)/100.0

                zy = random.randint(80,120)/100.0

                brightness = random.randint(1,2)/100.0

                channel_shift_intensity = random.randint(1,10)/100.0

                

                X[i,] = datagen.apply_transform(self.data[ID,],{

                                                'tx':tx,

                                                'ty':ty,

                                                'shear':shear,

                                                'zx':zx,

                                                'zy':zy,

                                                'flip_horizontal':flip_horizontal,

                                                'flip_vertical':flip_vertical,

                                                #'brightness':brightness,

                                                #'channel_shift_intensity':channel_shift_intensity

                                                }

                                            )

            else:

                # Store sample

                X[i,] = self.data[ID,]



            # Store class

            if self.test == False:

                y[i,] = self.labels[ID,]



        if self.test == False:

            return X, y

        else:

            return X
# Parameters

params_trn = {

          'im_size': im_size,

          'batch_size': batch_size,

          'shuffle': True,

          'data_augment' : True,

          'test' : False

         }

params_val = {

          'im_size': im_size,

          'batch_size': batch_size,

          'shuffle': True,

          'data_augment' : False,

          'test' : False

         }



# Generators

training_generator = DataGenerator(x_train_raw, y_train_raw, **params_trn)

validation_generator = DataGenerator(x_val_raw, y_val_raw, **params_val)
sys.stdout.write('running 100 epochs with early stopping with patience 7..')
K.set_value(model.optimizer.lr, 0.001)

model.fit_generator(generator=training_generator,validation_data=validation_generator,

          steps_per_epoch=  x_train_raw.shape[0]//batch_size,

          epochs=50,

          verbose=1,shuffle=True,callbacks=callbacks_list)
del(x_train_raw,x_val_raw)

gc.collect()
x_test_raw = bcolz.zeros((0,im_size,im_size,3),np.float32)
for f in tqdm(df_test['id'].values):

    # load an image from file

    image = load_img('../input/dog-breed-identification/test/{}.jpg'.format(f), target_size=(im_size, im_size))

    image = img_to_array(image)

    # prepare the image for the VGG model

    #image = preprocess_input(image)

    x_test_raw.append(image)
print(x_test_raw.shape)
model.load_weights("weights.best.hdf5")
params_test = {

          'im_size': im_size,

          'batch_size': 1,

          'shuffle': False,

          'data_augment' : False,

          'test' : True

         }

test_generator = DataGenerator(x_test_raw, None, **params_test)
x_test_raw.shape[0]
preds = model.predict_generator(test_generator, steps = x_test_raw.shape[0], verbose=1)
sub = pd.DataFrame(preds)

# Set column names to those generated by the one-hot encoding earlier

col_names = one_hot.columns.values

sub.columns = col_names

# Insert the column id from the sample_submission at the start of the data frame

sub.insert(0, 'id', df_test['id'])

sub.head(5)
%pwd
sub.to_csv("My first submission.csv",index =False)