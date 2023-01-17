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
train_dir = "../input/train/"
test_dir = "../input/test"

df_train = pd.read_csv('../input/train_labels.csv')
df_train.head()
# taking 10000 sample so our model run fast (experimentation)

from sklearn.model_selection import train_test_split

# df = df_train.sample(n=10000, random_state=2018)
df = df_train # using full dataset
train, valid = train_test_split(df,test_size=0.2)
# minimal preprocessing for experimentation. Will add more suitable preprocessing in the future

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x)
# use flow_from_dataframe method to build train and valid generator
# Only shuffle the train generator as we want valid generator to have the same structure as test

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
#     subset='training',
    batch_size=32,
    seed=2018,
    shuffle=True,
    class_mode='binary',
    target_size=(96,96))

valid_generator = test_datagen.flow_from_dataframe(
    dataframe = valid,
    directory='../input/train/',
    x_col='id',
    y_col='label',
    has_ext=False,
#     subset='validation',
    batch_size=32,
    seed=2018,
    shuffle=False,
    class_mode='binary',
    target_size=(96,96)
)
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras import layers

IMG_SIZE = (96, 96)
IN_SHAPE = (*IMG_SIZE, 3)

dropout_dense=0.5

conv_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=IN_SHAPE
)
       
model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(dropout_dense))
model.add(layers.Dense(1, activation = "sigmoid"))

conv_base.summary()
# freeze layer. unfreeze start at layer 5. if freeze everything, val acc will be really bad

conv_base.Trainable=True

set_trainable=False
for layer in conv_base.layers:
    if layer.name == 'res5a_branch2a':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    
from keras import optimizers

# conv_base.trainable = False
model.compile(optimizers.Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
# from keras.models import Sequential
# from keras import layers

# kernel_size=(3,3)
# pool_size=(2,2)
# first_filter=32
# second_filter=64
# third_filter=128

# dropout_conv=0.3

# model = Sequential()
# model.add(layers.Conv2D(first_filter, kernel_size, activation='relu', input_shape= (96,96,3)))
# model.add(layers.Conv2D(first_filter, kernel_size, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPool2D(pool_size=pool_size))
# model.add(layers.Dropout(dropout_conv))

# model.add(layers.Conv2D(second_filter, kernel_size, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(second_filter, kernel_size, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation("relu"))
# model.add(layers.MaxPool2D(pool_size = pool_size))
# model.add(layers.Dropout(dropout_conv))

# model.add(layers.Conv2D(third_filter, kernel_size, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation("relu"))
# model.add(layers.Conv2D(third_filter, kernel_size, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation("relu"))
# model.add(layers.MaxPool2D(pool_size = pool_size))
# model.add(layers.Dropout(dropout_conv))

# #model.add(GlobalAveragePooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(256, use_bias=False))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation("relu"))
# model.add(layers.Dropout(dropout_dense))
# model.add(layers.Dense(1, activation = "sigmoid"))

# # Compile the model
# model.compile(optimizers.Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, 
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=13,
                   callbacks=[reducel, earlystopper])
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
