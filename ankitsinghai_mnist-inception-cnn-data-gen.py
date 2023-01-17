# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten,Activation,Dropout
from keras.optimizers import SGD, Adam
from keras.models import Sequential
import keras
import keras.utils
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load the data in Data Frame
train_df = pd.read_csv('../input/train.csv')
# Initialise input data for CNN
input_data = np.zeros((train_df.shape[0],28,28))

# Getting Label from data Frame to np.array
Y_train = np.array(train_df['label'])
Y_train = keras.utils.to_categorical(Y_train, 10)

# method to convert an DataFrame row to matrix of n_rows
def conv_df_row_to_matrix(n_rows,arr):
    return arr.reshape(n_rows,-1)

train_np = train_df.drop('label',axis=1)
train_np = train_np / 255
for i in range(train_df.shape[0]):
    input_data[i] = conv_df_row_to_matrix(28, np.array(train_np.iloc[i]))
X_train = input_data.reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],1)
print(X_train.shape)
print(Y_train.shape)
input_img = Input(shape=(28, 28, 1))

# Layer 1
layer1_tower_0 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
layer1_tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
layer1_tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(layer1_tower_1)

layer1_tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
layer1_tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(layer1_tower_2)

layer1_tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
layer1_tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(layer1_tower_3)

layer1_output = keras.layers.concatenate([layer1_tower_1, layer1_tower_2, layer1_tower_3, layer1_tower_0], axis=1)

# Layer 2
layer2_tower_0 = Conv2D(32, (1, 1), padding='same', activation='relu')(layer1_output)
layer2_tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(layer1_output)
layer2_tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(layer2_tower_1)

layer2_tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(layer1_output)
layer2_tower_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(layer2_tower_2)

layer2_tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer1_output)
layer2_tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(layer2_tower_3)

layer2_output = keras.layers.concatenate([layer2_tower_1, layer2_tower_2, layer2_tower_3, layer2_tower_0], axis=1)

# Layer 3
layer3_tower_0 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer2_output)
layer3_tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer2_output)
layer3_tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(layer3_tower_1)

layer3_tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer2_output)
layer3_tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(layer3_tower_2)

layer3_tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer2_output)
layer3_tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer3_tower_3)

layer3_output = keras.layers.concatenate([layer3_tower_1, layer3_tower_2, layer3_tower_3, layer3_tower_0], axis=1)


# Flatten & Dense
layer3_output = Flatten()(layer3_output)
layer4_output = Dense(784,activation='relu')(layer3_output)
layer4_ouput = Dropout(0.25)(layer4_output)
output = Dense(10,activation='softmax')(layer4_output)

inception_Model_datagen = Model(inputs=input_img,outputs=output)
inception_Model = Model(inputs=input_img,outputs=output)
# checkpoint
filepath="model_acc_inception_datagen.best.hdf5"
checkpoint_acc = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
# Early Stopping
earlyStopping_acc = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
# LRU Decay
lr_reduction = ReduceLROnPlateau(monitor='val_acc',
                                 patience=3, 
                                 verbose=0, 
                                 factor=0.25, 
                                 min_lr=0.00001)
callbacks_list = [checkpoint_acc,earlyStopping_acc,lr_reduction]

inception_Model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
dataGen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2,
                             shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)
dataGen.fit(X_train)

train_generator = dataGen.flow(X_train, Y_train, batch_size=64, shuffle=True, 
                               seed=2, save_to_dir=None, subset='training')

validation_generator = dataGen.flow(X_train, Y_train, batch_size=64, shuffle=True, 
                               seed=2, save_to_dir=None, subset='validation')


history_inception_datagen = inception_Model.fit_generator(train_generator,
                                                steps_per_epoch = 600,
                                                epochs=30,
                                                validation_data = validation_generator,
                                                validation_steps = 150,
                                                callbacks = callbacks_list)
