# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Training"))
data_path = "../input/fruits/fruits-360_dataset/fruits-360/"



train = os.path.join(data_path, r'Training')



train_images = sorted(os.listdir(train))

print("Total number of images in the training set: ", len(train_images))
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



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 128

NUM_CLASSES = 120
foldernames = os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Training")

categories = []

files = []

i = 0

for folder in foldernames:

    filenames = os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Training/" + folder);

    for file in filenames:

        files.append("../input/fruits/fruits-360_dataset/fruits-360/Training/" + folder + "/" + file)

        categories.append(i)

    i = i + 1

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})
df.head()
y = df['category']
df['category'].value_counts()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify = y)

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)
x = train_df['filename']

y = train_df['category']



x, y = shuffle(x, y, random_state=8)

y.hist()
x.shape
y = to_categorical(y, num_classes=NUM_CLASSES)



train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.16,

                                                      stratify=y, random_state=8)

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

            img = cv2.imread(sample)

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

            img = cv2.imread(sample)

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



epochs = 80; batch_size = 64

checkpoint = ModelCheckpoint('../working/Resnet50-visible.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)

# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]



train_generator = My_Generator(train_x, train_y, batch_size, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
# warm up model

for layer in model.layers:

    layer.trainable = False



for i in range(-3,0):

    model.layers[i].trainable = True



model.compile(

    loss='categorical_crossentropy',

    # loss='binary_crossentropy',

    optimizer=Adam(1e-3))



model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=2,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1)
# train all layers

for layer in model.layers:

    layer.trainable = True



callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]

model.compile(loss='categorical_crossentropy',

            # loss=kappa_loss,

            # loss='binary_crossentropy',

            optimizer=Adam(lr=1e-4),

#             optimizer=AdamAccumulate(lr=1e-4, accum_iters=2),

            metrics=['accuracy'])



model.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    workers=1, use_multiprocessing=False,

    callbacks=callbacks_list)
# submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

model.load_weights('../working/Resnet50-visible.h5')

# model.load_weights('../working/Resnet50_bestqwk.h5')

predicted = []
foldernames = os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Test")

categories = []

files = []

i = 0

for folder in foldernames:

    filenames = os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Test/" + folder);

    for file in filenames:

        files.append("../input/fruits/fruits-360_dataset/fruits-360/Test/" + folder + "/" + file)

        categories.append(i)

    i = i + 1

        

        

df = pd.DataFrame({

    'filename': files,

    'category': categories

})
test_df = df
for sample in test_df['filename']:

    path = os.path.join(sample)

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    score_predict = model.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    # label_predict = score_predict.astype(int).sum() - 1

    predicted.append(str(label_predict))
test_df['predict'] = predicted
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

print(accuracy_score(test_df['category'].astype(int), test_df['predict'].astype(int)))

print(f1_score(test_df['category'].astype(int), test_df['predict'].astype(int), average='macro'))
import seaborn as sn

print('Confusion Matrix')

cm = confusion_matrix(test_df['category'].astype(int), test_df['predict'].astype(int))

print(cm)

sn.set(font_scale=1.4)#for label size

sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size