# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from imgaug import augmenters as iaa

from tqdm import tqdm

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
data_path = '../input'



train = os.path.join(data_path, r'train')

test = os.path.join(data_path, r'test')



train_images = sorted(os.listdir(train))

test_images = sorted(os.listdir(test))

print("Total number of images in the training set: ", len(train_images))

print("Total number of samples in the test set: ", len(test_images))
filenames = os.listdir("../input/train/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})



train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)

valid_df = valid_df.reset_index(drop=True)
train_df.head()
WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 256
class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=False,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/train/'+sample)

            img = cv2.resize(img, (SIZE, SIZE))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_images



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/train/'+sample)

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_images
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,

                         UpSampling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
# Lets' define our autoencoder now

def create_model(input_shape):

    input_tensor = Input(shape=input_shape)

    

    # encoder 

    encoded = Conv2D(8, (2,2), activation='relu', strides=(2,2))(input_tensor) # 128x128

    encoded = Conv2D(16, (2,2), activation='relu', strides=(2,2))(encoded) # 64x64

    encoded = Conv2D(32, (2,2), activation='relu', strides=(2,2))(encoded) # 32x32

    encoded = Conv2D(64, (2,2), activation='relu', strides=(2,2))(encoded) # 16x16

    

    # coded

    code = Conv2D(128, (2,2), activation='relu', strides=(2,2))(encoded) # 8x8

    

    # decoder

    decoded = Conv2D(128, (2,2), activation='relu', padding='same')(code) # 8x8

    decoded = UpSampling2D((2,2))(decoded) # 16x16

    decoded = Conv2D(64, (2,2), activation='relu', padding='same')(decoded) # 16x16

    decoded = UpSampling2D((2,2))(decoded) # 32x32

    decoded = Conv2D(32, (2,2), activation='relu', padding='same')(decoded) # 32x32

    decoded = UpSampling2D((2,2))(decoded) # 64x64

    decoded = Conv2D(16, (2,2), activation='relu', padding='same')(decoded) # 64x64

    decoded = UpSampling2D((2,2))(decoded) # 128x128

    decoded = Conv2D(8, (2,2), activation='relu', padding='same')(decoded) # 128x128

    decoded = UpSampling2D((2,2))(decoded) # 256x256

    

    output_tensor = Conv2D(3, (1,1), activation='sigmoid')(decoded)

    

    #model

    autoencoder = Model(inputs=input_tensor, outputs=output_tensor)

    return autoencoder
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 50; batch_size = 16

checkpoint = ModelCheckpoint('../working/auto_encoder.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=5)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



train_generator = My_Generator(train_df['filename'], train_df['category'], batch_size, is_train=True)

valid_generator = My_Generator(valid_df['filename'], valid_df['category'], batch_size, is_train=False)



model = create_model(input_shape=(SIZE,SIZE,3))
model.summary()
model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(1e-3))



callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]



model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_df)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_df)) / float(batch_size)),

    epochs=epochs,

    workers=WORKERS, use_multiprocessing=False,

    verbose=1,

    callbacks=callbacks_list)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg





for file_name in tqdm(test_images[:1]):

    path = os.path.join('../input/test/', file_name)

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    decoded_image = model.predict((image[np.newaxis])/255)[0]

    

    ax = plt.subplot(1, 2, 1)

    plt.imshow(image)



    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    ax = plt.subplot(1, 2, 2)

    plt.imshow(decoded_image)



    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    plt.show()    

    