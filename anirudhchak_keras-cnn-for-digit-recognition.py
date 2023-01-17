# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



seed = 4529

np.random.seed(seed)
base_dir = os.path.join("..", "input")

train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))

test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))



len(train_df)
%load_ext tensorboard.notebook

%tensorboard --logdir logs
x = train_df.drop(['label'], axis=1).values

y = train_df['label'].values

test_x = test_df.values
x = x.reshape(-1, 28, 28, 1)

x = x / 255.0



test_x = test_x.reshape(-1, 28, 28, 1)

test_x = test_x / 255.0



# one-hot encoding

y = to_categorical(y, num_classes=10)



x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.1, random_state=seed)
model = Sequential([

    Conv2D(128, (3,3), activation="relu", input_shape=(28, 28, 1)),

    BatchNormalization(),

    Conv2D(128, (3,3), activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.2),

    Conv2D(64, (3,3), activation="relu"),

    BatchNormalization(),

    Conv2D(64, (3,3), activation="relu"),

    BatchNormalization(),

    MaxPooling2D(2,2),

    Dropout(0.2),

    Flatten(),

    Dense(units=256, activation='relu'),

    Dropout(0.4),

    Dense(units=256, activation='relu'),

    Dropout(0.4),

    Dense(units=10, activation='softmax')

])



model.compile(optimizer=Adam(lr=0.001), 

                 loss='categorical_crossentropy',

                 metrics=['acc'])

model.summary()
ckpt_path = 'mnist.hdf5'



earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, restore_best_weights=False)

reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

modelckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

tb = TensorBoard()



callbacks = [earlystop, reducelr, modelckpt_cb, tb]
batch_size = 128

epochs = 30



# Using Data augmentation

datagen = ImageDataGenerator(rotation_range=15,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             zoom_range=0.15, 

                             shear_range=0.15)

datagen.fit(x_train)



# Without data augmentation

# history = model.fit(x_train, 

#               y_train, 

#               batch_size=batch_size, 

#               validation_data = (x_val, y_val),

#               epochs=epochs, 

#               callbacks=callbacks)



history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                                 validation_data=(x_val, y_val),

                                 steps_per_epoch = x_train.shape[0] // batch_size,

                                 epochs=epochs, callbacks=callbacks)

# Training plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('loss')

plt.xlabel('epoch')

plt.show()



plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")

plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")

plt.legend(loc='best')

plt.title('accuracy')

plt.xlabel('epoch')

plt.show()
pred = model.predict(test_x)

pred = np.argmax(pred, axis=1)

pred = pd.Series(pred, name="Label")

test_df = pd.concat([pd.Series(range(1,28001), name = "ImageId"), pred],axis = 1)

test_df.to_csv('mnist-submission.csv', index = False)