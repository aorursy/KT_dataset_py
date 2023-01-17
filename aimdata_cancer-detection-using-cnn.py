# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from PIL import Image
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, BatchNormalization

IMG_SIZE = (96, 96)
IN_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 64
# CROP_TEST = 36
# IN_SHAPE_TEST = (CROP_TEST,CROP_TEST,3)

df_data = pd.read_csv('../input/train_labels.csv')
train_dir = "../input/train/"
test_dir = "../input/test"
    
#Taken from https://www.kaggle.com/byrachonok/cancer-detection-show-data
fig, ax = plt.subplots(1,3, figsize=(20,5))
for i, idx in enumerate(df_data[df_data['label'] == 0]['id'][:3]):
    path = os.path.join('/kaggle/input/train/', idx)
    ax[i].imshow(Image.open(path+'.tif'))
    pf = Polygon(((32, 32), (64, 32), (64, 64), (32, 64)),
            fc=(0.0, 0.0, 0.0, 0.0), 
            ec=(0.0, 0.9, 0.0 ,0.9), lw=4, linestyle='--')
    ax[i].add_patch(pf)
fig, ax = plt.subplots(1,3, figsize=(20,5))
for i, idx in enumerate(df_data[df_data['label'] == 1]['id'][:3]):
    path = os.path.join('/kaggle/input/train/', idx)
    ax[i].imshow(Image.open(path+'.tif'))
    pt = Polygon(((32, 32), (64, 32), (64, 64), (32, 64)),
            fc=(0.0, 0.0, 0.0, 0.0), 
            ec=(0.9, 0.0, 0.0 ,0.9), lw=4, linestyle='--')
    ax[i].add_patch(pt)
train, valid = train_test_split(df_data,test_size=0.15)

train_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   horizontal_flip=True, vertical_flip=True,
                                   rotation_range=90, shear_range=0.05, zoom_range=0.1 )

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
    batch_size=BATCH_SIZE,
    seed=2018,
    shuffle=True,
    class_mode='binary',
    target_size=IMG_SIZE)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe = valid,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
    batch_size=BATCH_SIZE,
    seed=2018,
    shuffle=False,
    class_mode='binary',
    target_size=IMG_SIZE
)
# conv_base = ResNet50(
#     weights='imagenet',
#     include_top=False,
#     input_shape=IN_SHAPE
# )

# VGG model without the last classifier layers (include_top = False)
conv_base = VGG16(include_top = False,
                    input_shape = IN_SHAPE,
                    weights='imagenet')
    
# Freeze the layers 
for layer in conv_base.layers[:-12]:
    layer.trainable = False
    
# Check the trainable status of the individual layers
for layer in conv_base.layers:
    print(layer, layer.trainable)
model = Sequential()
# model.add(Cropping2D(cropping=((CROP_TEST,CROP_TEST), (48,48)), input_shape=IN_SHAPE))
model.add(conv_base)
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.6))
model.add(BatchNormalization())
model.add(Dense(1, activation = "sigmoid"))

conv_base.summary()
# conv_base.Trainable=True

# set_trainable=False
# for layer in conv_base.layers:
#     if layer.name == 'res5a_branch2a':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
model.compile(optimizers.Adam(0.001), loss = "binary_crossentropy", metrics=["accuracy"])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
learning_rate_decay = ReduceLROnPlateau(monitor='acc', patience=2, verbose=1, factor=0.3, min_lr=1e-5)

history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, 
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50,
                   callbacks=[learning_rate_decay, earlystopper])
from glob import glob
from skimage.io import imread

base_test_dir = '../input/test/'
test_files = glob(os.path.join(base_test_dir,'*.tif'))
submission = pd.DataFrame()
file_batch = 5000
max_idx = len(test_files)
for idx in range(0, max_idx, file_batch):
    print("Indexes: %i - %i"%(idx, idx+file_batch))
    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})
    test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])
    test_df['image'] = test_df['path'].map(imread)
    K_test = np.stack(test_df["image"].values)
    K_test = (K_test - K_test.mean()) / K_test.std()
    predictions = model.predict(K_test)
    test_df['label'] = predictions
    submission = pd.concat([submission, test_df[["id", "label"]]])
submission.head()
submission.to_csv("submission.csv", index = False, header = True)
# Save the last model
#model.save('../input/model.h5')
