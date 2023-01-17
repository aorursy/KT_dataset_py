import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import os

WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 224

NUM_CLASSES = 7
import pandas as pd

filenames = os.listdir("../input/final-thermal/data/")

categories = []

for filename in filenames:

#     print(filename.split('_')[2])

    category = filename.split('_')[2]

    if category == 'anger':

        categories.append(0)

    elif category == 'disgust':

        categories.append(1)

    elif category == 'fear':

        categories.append(2)

    elif category == 'happy':

        categories.append(3)

    elif category == 'neutral':

        categories.append(4)

    elif category == 'sadness':

        categories.append(5)

    else:

        categories.append(6)

        

thermal_df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



filenames = os.listdir("../input/final-visible/data/")

categories = []

for filename in filenames:

#     print(filename.split('_')[2])

    category = filename.split('_')[2]

    if category == 'anger':

        categories.append(0)

    elif category == 'disgust':

        categories.append(1)

    elif category == 'fear':

        categories.append(2)

    elif category == 'happy':

        categories.append(3)

    elif category == 'neutral':

        categories.append(4)

    elif category == 'sadness':

        categories.append(5)

    else:

        categories.append(6)



visible_df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
y_thermal = thermal_df['category']

y_visible = visible_df['category']
train_t_df, test_t_df = train_test_split(thermal_df, test_size=0.2, random_state=42, stratify = y_thermal)

train_t_df = train_t_df.reset_index(drop=True)

test_t_df = test_t_df.reset_index(drop=True)

train_v_df, test_v_df = train_test_split(visible_df, test_size=0.2, random_state=42, stratify = y_visible)

train_v_df = train_v_df.reset_index(drop=True)

test_v_df = test_v_df.reset_index(drop=True)
x_t = train_t_df['filename']

y_t = train_t_df['category']

x_v = train_v_df['filename']

y_v = train_v_df['category']

x_t, y_t = shuffle(x_t, y_t, random_state=8)

x_v, y_v = shuffle(x_v, y_v, random_state=12)
y_t = to_categorical(y_t, num_classes=NUM_CLASSES)



train_x, valid_x, train_y, valid_y = train_test_split(x_t, y_t, test_size=0.16,

                                                      stratify=y_t, random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
# https://github.com/aleju/imgaug

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([

    sometimes(

        iaa.OneOf([

            iaa.Add((-10, 10), per_channel=0.5),

            iaa.Multiply((0.9, 1.1), per_channel=0.5),

            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)

        ])

    ),

    iaa.Fliplr(0.5),

    # iaa.Crop(percent=(0, 0.1)),

    # iaa.Flipud(0.5)

],random_order=True)
class My_Generator_T(Sequence):



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

            img = cv2.imread('../input/final-thermal/data/'+sample)

#             print('../input/data/Data'+sample)

            img = cv2.resize(img, (SIZE, SIZE))

#             print(img.shape)

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/final-thermal/data/'+sample)

#             print(img)

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
class My_Generator_V(Sequence):



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

            img = cv2.imread('../input/final-visible/data/'+sample)

#             print('../input/data/Data'+sample)

            img = cv2.resize(img, (SIZE, SIZE))

#             print(img.shape)

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/final-visible/data/'+sample)

#             print(img)

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
function = "softmax"

def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = ResNet50(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)

#     x = Dropout(0.5)(x)

#     x = Dense(1024, activation='relu')(x)

    x = Dropout(0.3)(x)

    final_output = Dense(n_out, activation=function, name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 80; batch_size = 32

checkpoint = ModelCheckpoint('../working/Resnet50-thermal.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log_t.csv',

                       separator=',',

                       append=True)

# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]



train_generator = My_Generator_T(train_x, train_y, 128, is_train=True)

train_mixup = My_Generator_T(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator_T(valid_x, valid_y, batch_size, is_train=False)



thermal = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
thermal.summary()
import numpy as np

# warm up model

for layer in thermal.layers:

    layer.trainable = False



for i in range(-3,0):

    thermal.layers[i].trainable = True



thermal.compile(

    loss='categorical_crossentropy',

    # loss='binary_crossentropy',

    optimizer=Adam(1e-3))



thermal.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=2,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1)
# train all layers

for layer in thermal.layers:

    layer.trainable = True



callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]

thermal.compile(loss='categorical_crossentropy',

            # loss=kappa_loss,

            # loss='binary_crossentropy',

            optimizer=Adam(lr=2e-3),

#             optimizer=AdamAccumulate(lr=1e-4, accum_iters=2),

            metrics=['accuracy'])



thermal.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    workers=1, use_multiprocessing=False,

    callbacks=callbacks_list)
# submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

thermal.load_weights('../working/Resnet50-thermal.h5')

# model.load_weights('../working/Resnet50_bestqwk.h5')

predicted = []

prob = []


for sample in test_t_df['filename']:

    path = os.path.join("../input/final-thermal/data/"+sample)

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    score_predict = thermal.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    # label_predict = score_predict.astype(int).sum() - 1

    prob.append(score_predict)

    predicted.append(str(label_predict))
test_t_df['predict'] = predicted

test_t_df['prob'] = prob
y_v = to_categorical(y_v, num_classes=NUM_CLASSES)



train_x, valid_x, train_y, valid_y = train_test_split(x_v, y_v, test_size=0.16,

                                                      stratify=y_v, random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 80; batch_size = 32

checkpoint = ModelCheckpoint('../working/Resnet50-visible.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log_v.csv',

                       separator=',',

                       append=True)

# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]



train_generator = My_Generator_V(train_x, train_y, 128, is_train=True)

train_mixup = My_Generator_V(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator_V(valid_x, valid_y, batch_size, is_train=False)



visible = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
visible.summary()
import numpy as np

# warm up model

for layer in visible.layers:

    layer.trainable = False



for i in range(-3,0):

    visible.layers[i].trainable = True



visible.compile(

    loss='categorical_crossentropy',

    # loss='binary_crossentropy',

    optimizer=Adam(1e-3))



visible.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=2,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1)
# train all layers

for layer in visible.layers:

    layer.trainable = True



callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]

visible.compile(loss='categorical_crossentropy',

            # loss=kappa_loss,

            # loss='binary_crossentropy',

            optimizer=Adam(lr=2e-3),

#             optimizer=AdamAccumulate(lr=1e-4, accum_iters=2),

            metrics=['accuracy'])



visible.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    workers=1, use_multiprocessing=False,

    callbacks=callbacks_list)
# submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

visible.load_weights('../working/Resnet50-visible.h5')

# model.load_weights('../working/Resnet50_bestqwk.h5')

predicted = []

prob = []


for sample in test_v_df['filename']:

    path = os.path.join("../input/final-visible/data/"+sample)

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    score_predict = visible.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    # label_predict = score_predict.astype(int).sum() - 1

    prob.append(score_predict)

    predicted.append(str(label_predict))
test_v_df['predict'] = predicted

test_v_df['prob'] = prob


from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

alpha = f1_score(test_t_df['category'].astype(int), test_t_df['predict'].astype(int), average='macro')

beta = f1_score(test_v_df['category'].astype(int), test_v_df['predict'].astype(int), average='macro')

print(alpha,beta)