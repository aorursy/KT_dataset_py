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
%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization 

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
PATH = '../input/'
df_train = pd.read_csv(f'{PATH}train.csv')

df_test = pd.read_csv(f'{PATH}test.csv')
X_train = df_train.drop('label', axis=1).values

X_test = df_test.values



Y_train = df_train.label



print(f'Number of training Examples, {X_train.shape[0]}')

print(f'Number of test Examples, {X_test.shape[0]}')



print(f'Number of classes, {np.unique(Y_train)}')
sns.countplot(Y_train);
Y_train.shape
X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)
print('Integer Valued Labels')

print(Y_train[:10])



Y_train = np_utils.to_categorical(Y_train, 10)



print('One Hot Labels')

print(Y_train[:10])
fig = plt.figure(figsize=(15, 15))

for i in range(6):

    ax = fig.add_subplot(1, 6, i + 1, xticks=[], yticks=[])

    ax.imshow(X_train[i][:, :, 0], cmap='gray')

    ax.set_title(str(np.argmax(Y_train[i])))
def visualize_input(img, ax):

    ax.imshow(img[:, :, 0], cmap='gray')

    width, height = img.shape[0], img.shape[1]

    thres = img.max() / 2.5

    for w in range(width):

        for h in range(height):

            ax.annotate(str(round(img[w, h, 0], 2)), xy=(h, w),

                  horizontalalignment='center',

                  verticalalignment='center',

                  color='white' if img[w][h] < thres else 'black')

            

fig = plt.figure(figsize = (12, 12))

ax = fig.add_subplot(111)

visualize_input(X_train[3], ax)
X_train = X_train.astype('float32') / 255

X_test = X_test.astype('float32') / 255
# print('Integer Valued Labels')

# print(Y_train[:10])



# Y_train = np_utils.to_categorical(Y_train, 10)



# print('One Hot Labels')

# print(Y_train[:10])
X_train, X_valid = X_train[7000:], X_train[:7000]

Y_train, Y_valid = Y_train[7000:], Y_train[:7000]
print(f'Training Data shape, {X_train.shape}')

print(f'Validation Data shape, {X_valid.shape}')

print(f'Testing Data shape, {X_test.shape}')

print(f'Output Label shape, {Y_train.shape}, {Y_valid.shape}')


model = Sequential()

model.add(Flatten(input_shape=X_train.shape[1:]))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5',

                               verbose=1, save_best_only=True)

hist = model.fit(X_train, Y_train, batch_size=128, epochs=10,

                 validation_split=0.2, callbacks=[checkpointer],

                 verbose=1, shuffle=True)
model.load_weights('mnist.model.best.hdf5')
score = model.evaluate(X_valid, Y_valid, verbose=0)

accuracy = 100 * score[1]



print(f'Test Accuracy {accuracy:.4f}')
def submit_result(model, X_test):

    results = model.predict(X_test)

    results = np.argmax(results,axis = 1)

    results = pd.Series(results,name="Label")

    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

    submission.to_csv("mnist_datagen.csv",index=False)

# print('../input/digit-recognizer/__results___files/__results___14_0.png')


cnn_model = Sequential()

cnn_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',

                     activation='relu', input_shape=X_train.shape[1:]))

cnn_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',

                     activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Dropout(0.2))



cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',

                     activation='relu'))

cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',

                     activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Dropout(0.2))



cnn_model.add(Flatten())

cnn_model.add(Dense(512, activation='relu'))

cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(512, activation='relu'))

cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(10, activation='softmax'))



cnn_model.summary()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='mnist.model.best.cnn.hdf5',

                               verbose=1, save_best_only=True)

hist = cnn_model.fit(X_train, Y_train, batch_size=128, epochs=10,

                 validation_split=0.2, callbacks=[checkpointer],

                 verbose=1, shuffle=True)
cnn_model.load_weights('mnist.model.best.cnn.hdf5')
score = cnn_model.evaluate(X_valid, Y_valid, verbose=0)

accuracy = 100 * score[1]



print(f'Test Accuracy {accuracy:.4f}')
# submit_result(cnn_model, X_test)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

print(f'Shape of the training data {X_train.shape}')

print(f'Shape of the validation dat {X_val.shape}')
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False  # randomly flip images



)



datagen.fit(X_train)
batch_size=128



checkpointer = ModelCheckpoint(filepath='mnist.model.best.cnn.aug.hdf5',

                               verbose=1, save_best_only=True)



hist = cnn_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                        epochs=10, validation_data=(X_val, Y_val), callbacks=[checkpointer],

                        verbose=2, shuffle=True, steps_per_epoch=X_train.shape[0] // batch_size)
cnn_model.load_weights('mnist.model.best.cnn.aug.hdf5')
score = cnn_model.evaluate(X_valid, Y_valid, verbose=0)

accuracy = 100 * score[1]



print(f'Test Accuracy {accuracy:.4f}')
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
checkpointer = ModelCheckpoint(filepath='mnist.model.best.cnn.aug.ann.hdf5',

                               verbose=1, save_best_only=True)



hist = cnn_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                        epochs=20, validation_data=(X_val, Y_val), callbacks=[checkpointer, learning_rate_reduction],

                        verbose=2, shuffle=True, steps_per_epoch=X_train.shape[0] // batch_size)

cnn_model.load_weights('mnist.model.best.cnn.aug.ann.hdf5')
score = cnn_model.evaluate(X_valid, Y_valid, verbose=0)

accuracy = 100 * score[1]



print(f'Test Accuracy {accuracy:.4f}')
submit_result(cnn_model, X_test)
cnn_bn_model = Sequential()



cnn_bn_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

cnn_bn_model.add(BatchNormalization())



cnn_bn_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

cnn_bn_model.add(BatchNormalization())



cnn_bn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_bn_model.add(Dropout(0.25))



cnn_bn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

cnn_bn_model.add(BatchNormalization())



cnn_bn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

cnn_bn_model.add(BatchNormalization())

cnn_bn_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

cnn_bn_model.add(Dropout(0.25))



cnn_bn_model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

cnn_bn_model.add(BatchNormalization())

cnn_bn_model.add(Dropout(0.25))



cnn_bn_model.add(Flatten())

cnn_bn_model.add(Dense(256, activation = "relu"))

cnn_bn_model.add(BatchNormalization())

cnn_bn_model.add(Dropout(0.25))



cnn_bn_model.add(Dense(10, activation = "softmax"))



cnn_bn_model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

cnn_bn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='final.model.hdf5',

                               verbose=1, save_best_only=True)



hist = cnn_bn_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                        epochs=20, validation_data=(X_val, Y_val), callbacks=[checkpointer, learning_rate_reduction],

                        verbose=2, shuffle=True, steps_per_epoch=X_train.shape[0] // batch_size)

cnn_bn_model.load_weights('final.model.hdf5')
score = cnn_bn_model.evaluate(X_valid, Y_valid, verbose=0)

accuracy = 100 * score[1]



print(f'Test Accuracy {accuracy:.4f}')
submit_result(cnn_bn_model, X_test)