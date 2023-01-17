!wget -q https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/foodc/v0.1/train_images.zip

!wget -q https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/foodc/v0.1/test_images.zip

!wget -q https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/foodc/v0.1/train.csv

!wget -q https://s3.eu-central-1.wasabisys.com/aicrowd-practice-challenges/public/foodc/v0.1/test.csv
!mkdir data

!mkdir data/test

!mkdir data/train

!unzip train_images -d data/train

!unzip test_images -d data/test
! pip install split-folders
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import cv2

import shutil

import PIL

import seaborn as sns

import tensorflow as tf

import split_folders

from cv2 import CascadeClassifier

from keras import utils as np_utils

from sklearn.metrics import confusion_matrix,classification_report

from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img

from keras.layers import Dense, Activation, Dropout, Flatten,GlobalAveragePooling2D

from keras.preprocessing import image

from keras.optimizers import Adam,SGD

from keras.models import Sequential,load_model,Model

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from sklearn.model_selection import train_test_split

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ReduceLROnPlateau
from __future__ import print_function

import pandas as pd

import shutil

import os

import sys



labels = pd.read_csv('train.csv')



# Create `train_sep` directory

train_dir = 'data/train/train_images/'

train_sep_dir = 'data/train_sep/'



if not os.path.exists(train_sep_dir):

    os.mkdir(train_sep_dir)



for filename, class_name in labels.values:

    # Create subdirectory with `class_name`

    if not os.path.exists(train_sep_dir + class_name):

        os.mkdir(train_sep_dir + class_name)



    src_path = train_dir + filename 

    dst_path = train_sep_dir + class_name + '/' + filename

    try:

        shutil.copy(src_path, dst_path)

    except IOError as e:

        print('Unable to copy file {} to {}'

              .format(src_path, dst_path))

    except:

        print('When try copy file {} to {}, unexpected error: {}'

              .format(src_path, dst_path, sys.exc_info()))

# src: https://www.kaggle.com/c/dog-breed-identification/discussion/48908
labels = pd.read_csv('train.csv')

all_labels = []



for filename,class_name in labels.values:

  if class_name not in all_labels:

    all_labels.append(class_name)
TRAIN_DIR = 'data/final_data/train/'

VAL_DIR = 'data/final_data/val/'

img_height = 256

img_width = 256

img_channels = 3

batch_size = 32

epochs = 25
split_folders.ratio("data/train_sep/", output="data/final_data", seed=1337, ratio=(.8, .2))
train_datagen = ImageDataGenerator(

    rescale = 1./255,

    zoom_range = 0.2,

    shear_range = 0.1,

    fill_mode = 'reflect',

    width_shift_range = 0.1,

    height_shift_range = 0.1

)



train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(img_height, img_width),

    color_mode="rgb",

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=True,

    seed=42

)
sum = 0

for val in all_labels:

    count = len(os.listdir(TRAIN_DIR + val))

    sum = sum + count

    print(val,count)

    plt.bar(val,count)

train_size = sum    
val_datagen = ImageDataGenerator(

    rescale = 1./255,

)



val_generator = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(img_height, img_width),

    color_mode="rgb",

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=False

)
for val in all_labels:

    count = len(os.listdir(VAL_DIR + val))

    sum = sum + count

    print(val,count)

    plt.bar(val,count)

val_size = sum    
labels = []

for val in val_generator.class_indices:

    labels.append(val)
from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras import regularizers
# base_model = ResNet50(include_top=False,weights=None,input_shape=(img_height,img_width,img_channels))

# res_model = base_model.output

# res_model = GlobalAveragePooling2D()(res_model)

# res_model = Dropout(0.5)(res_model)

# predictions = Dense(61,activation='softmax')(res_model)

# model = Model(inputs = base_model.input,outputs = predictions)
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128,activation='relu')(x)

x = Dropout(0.2)(x)



predictions = Dense(61,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer= 'Adam')
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',

                                            patience=2,

                                            factor=0.1,

                                            # min_lr = 0.0000001,

                                            verbose = 1)
path_model='model_filter.h5' 

# fit the model

hist = model.fit_generator(

    train_generator,

    epochs = epochs,

    validation_data = val_generator,

    steps_per_epoch = train_size//batch_size,

    validation_steps = val_size//batch_size,

    callbacks = [

        ModelCheckpoint(filepath=path_model),

        learning_rate_reduction,

    ]

)
model.save('model.h5')
score = model.evaluate_generator(train_generator)

print(score[1]*100)
score = model.evaluate_generator(val_generator)

print(score[1]*100)
predict = []

TEST_DIR = "data/test/test_images/"



for img in os.listdir(TEST_DIR):

    img = TEST_DIR + img

    img = image.load_img(img, target_size=(img_width, img_height),grayscale="False")

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    val = model.predict(img)

    val = np.argmax(val)

    predict.append(labels[val])

#     print(labels[val])
# Create Submission file        

df = pd.DataFrame(predict,columns=['ClassName'])

df.to_csv('submission.csv',index=False)