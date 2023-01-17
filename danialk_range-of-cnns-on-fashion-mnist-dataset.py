import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import pandas as pd

np.random.seed(12345)

%matplotlib inline
batch_size = 512

train_data = np.array(pd.read_csv('../input/fashion-mnist_train.csv'))

test_data = np.array(pd.read_csv('../input/fashion-mnist_test.csv'))
test_data.shape
train_data[:, 0]
X_train_orig = train_data[:, 1:785]

y_train_orig = train_data[:, 0]

X_test = train_data[:, 1:785]

y_test = train_data[:, 0]
X_train_orig = X_train_orig.astype('float32')

X_test = X_test.astype('float32')

X_train_orig /= 255

X_test /= 255
print(X_train_orig.shape)

print(y_train_orig.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=12345)
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
plt.imshow(X_train[2, :].reshape((28, 28)))
model = Sequential([

    Dense(512, input_shape=(784,), activation='relu'),

    Dense(128, activation = 'relu'),

    Dense(10, activation='softmax')

])
model.summary()
model.compile(optimizer=Adam(lr=0.001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, y_train,

                    batch_size=batch_size,

                    epochs=20,

                    verbose=1,

                    validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
img_rows = 28

img_cols = 28

input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
cnn1 = Sequential([

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')

])
cnn1.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
cnn1.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
cnn1.optimizer.lr = 0.0001
cnn1.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
score = cnn1.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)

batches = gen.flow(X_train, y_train, batch_size=batch_size)

val_batches = gen.flow(X_val, y_val, batch_size=batch_size)
cnn1.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)
score = cnn1.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn2 = Sequential([

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),



    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),



    Conv2D(128, kernel_size=(3, 3), activation='relu'),

    Dropout(0.2),



    Flatten(),



    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(10, activation='softmax')

])
cnn2.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
cnn2.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
cnn2.optimizer.lr = 0.0001
cnn2.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
score = cnn2.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn2.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)
score = cnn2.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)

def norm_input(x): return (x-mean_px)/std_px
cnn3 = Sequential([

    Lambda(norm_input, input_shape=(28,28, 1)),

    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    BatchNormalization(),



    Conv2D(32, kernel_size=(3, 3), activation='relu'),    

    BatchNormalization(),

    Dropout(0.25),



    Conv2D(64, kernel_size=(3, 3), activation='relu'),    

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),

    

    

    Conv2D(128, kernel_size=(3, 3), activation='relu'),    

    BatchNormalization(),

    Dropout(0.25),



    Flatten(),



    Dense(512, activation='relu'),

    BatchNormalization(),

    Dropout(0.5),

    Dense(128, activation='relu'),

    BatchNormalization(),

    Dropout(0.5),

    Dense(10, activation='softmax')

])
cnn3.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
cnn3.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
cnn3.optimizer.lr = 0.0001
cnn3.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
score = cnn3.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn3.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)
score = cnn3.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn4 = Sequential([

    Lambda(norm_input, input_shape=(28,28, 1)),

    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),

    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

    MaxPooling2D(pool_size=(2, 2)),

    

    

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    

    MaxPooling2D(pool_size=(2, 2)),

    

    

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    

    MaxPooling2D(pool_size=(2, 2)),

    

    

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    MaxPooling2D(pool_size=(2, 2)),



    Flatten(),



    Dense(512, activation='relu'),

    Dropout(0.5),

    Dense(512, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])
cnn4.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
cnn4.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
cnn4.optimizer.lr = 0.0001
cnn4.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
score = cnn4.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn4.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)
score = cnn4.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn5 = Sequential([

    Lambda(norm_input, input_shape=(28,28, 1)),

    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),

    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

    BatchNormalization(),

    Dropout(0.25),

    

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),    

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),

    

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),    

    BatchNormalization(),

    Dropout(0.25),

    

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),

    MaxPooling2D(pool_size=(2, 2)),



    Flatten(),



    Dense(512, activation='relu'),

    BatchNormalization(),

    Dropout(0.5),

    Dense(512, activation='relu'),

    BatchNormalization(),

    Dropout(0.5),

    Dense(10, activation='softmax')

])
cnn5.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
cnn5.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
cnn5.optimizer.lr = 0.0001
cnn5.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=10,

          verbose=1,

          validation_data=(X_val, y_val))
score = cnn5.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
cnn5.fit_generator(batches, steps_per_epoch=48000//batch_size, epochs=50, 

                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)
score = cnn5.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])