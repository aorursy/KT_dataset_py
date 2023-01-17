# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Add

from keras.layers import BatchNormalization, Activation

from keras.models import Model

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print("It worked!")
train=train.as_matrix()

test=test.as_matrix()
#y is lable y is data in this case

y=train[:,0:1]

#our model expects one-hot-vector

y=to_categorical(y)



print(y.shape)

x=train[:,1:]

#last dimension is a dummy stand in for red/green/blue channel

x=x.reshape(42000,28,28,1)
test=test.reshape(28000,28,28,1)

test.shape
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)



print("that went well!")
# Input

inputs = Input(shape=(28, 28, 1))



bn0 = BatchNormalization(scale=True)(inputs)



# Initial Stage

conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(bn0)

conv1 = Conv2D(32, kernel_size=(7,7), padding='same', activation='relu', kernel_initializer='uniform')(conv1)

bn1 = BatchNormalization(scale=True)(conv1)

max_pool1 = MaxPooling2D(pool_size=(2,2))(bn1)



# First

conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool1)

conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)

conv2 = Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv2)

bn2 = BatchNormalization(scale=True)(conv2)



# First Residual

res_conv1 = Conv2D(32, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool1)

res_bn1 = BatchNormalization(scale=True)(res_conv1)



# First Add

add1 = Add()([res_bn1, bn2])



# First Acvtivation & MaxPooling

act1 = Activation('relu')(add1)

max_pool2 = MaxPooling2D(pool_size=(2,2))(act1)



# Second

conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(max_pool2)

conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)

conv3 = Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='uniform')(conv3)

bn3 = BatchNormalization(scale=True)(conv3)



# Second Residual

res_conv2 = Conv2D(64, kernel_size=(3,3), padding='same', kernel_initializer='uniform')(max_pool2)

res_bn2 = BatchNormalization(scale=True)(res_conv2)



# Second Add

add2 = Add()([res_bn2, bn3])



# Second Acvtivation & MaxPooling

act2 = Activation('relu')(add2)

max_pool3 = MaxPooling2D(pool_size=(2,2))(act2)



# Flattern the data

flatten = Flatten()(max_pool3)



# Fully Connected Layer

dense1 = Dense(128, activation='relu')(flatten)

do = Dropout(0.25)(dense1)



dense2 = Dense(10, activation='softmax')(do)



model = Model(inputs=[inputs], outputs=[dense2])



# Parameters for training

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(X_train,y_train, validation_split=0.2, epochs=5)

datagen = ImageDataGenerator(rotation_range=25,

                             width_shift_range=0.1,

                             height_shift_range=0.1)



model.fit(X_train, y_train)
# Checkpoint to save the best model

checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)



# Reduce learning rate

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)



batch_size = 32

epochs = 40



history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                              epochs=epochs, validation_data=(X_val,y_val),

                              verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks=[checkpointer, reduce_lr])