# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
from time import time

import os

print(os.listdir("../input"))



import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns; sns.set()

#from IPython.display import SVG

#from keras.utils.vis_utils import model_to_dot



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

#from sklearn.metrics import confusion_matrix



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
name = "version_01"



# read data (only the first time)

read_image = True



# number of epochs to train:

epochs = 4



# Pixel size for image resizeing:

pixel_x = 32

pixel_y = 32



# batch size

batch_size = 32
t = time()

if read_image:

    # set directories for images

    p_pa = os.path.join(os.getcwd(), '../input/cell_images/cell_images/Parasitized')

    p_un = os.path.join(os.getcwd(), '../input/cell_images/cell_images/Uninfected')



    d_pa = [os.path.join(p_pa, f) for f in os.listdir(p_pa) if f.endswith('.png')]

    d_un = [os.path.join(p_un, f) for f in os.listdir(p_un) if f.endswith('.png')]



    # data container:

    N = len(d_pa) + len(d_un)

    X_all = np.zeros((N, pixel_x, pixel_y, 3))

    y_all = np.zeros(N)



    # read parasitized images and set label = 1:

    counter = 0

    for file in d_pa:

        img = plt.imread(file)

        img = cv2.resize(img, (pixel_x, pixel_y))

        X_all[counter] = img

        counter += 1

    y_all[:counter] = 1



    # read uninfected images and set label = 0:

    for file in d_un:

        img = plt.imread(file)

        img = cv2.resize(img, (pixel_x, pixel_y))

        X_all[counter] = img

        counter += 1

    y_all[counter:] = 0



    # save images as numpy variables:

    np.save("X_all.npy", X_all)

    np.save("y_all.npy", y_all)



else:

    # load image numpy arrays:

    X_all = np.load("X_all.npy")

    y_all = np.load("y_all.npy")

    pixel_x = X_all.shape[1]

    pixel_y = X_all.shape[2]

    

print("Delta Time for execution: {:.2f} sec".format(time()-t))
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,5))



n_pa = 1; n_un = 26000

axes[0].imshow(X_all[n_pa])

axes[0].set_title('Label: {:d}'.format(int(y_all[n_pa])))

axes[0].grid(False)

axes[1].imshow(X_all[n_un])

axes[1].set_title('Label: {:d}'.format(int(y_all[n_un])))

axes[1].grid(False)



n_pa = 25339; n_un = 10212

npa=2; nui=10212

axes[2].imshow(X_all[n_pa])

axes[2].set_title('Label: {:d}'.format(int(y_all[n_pa])))

axes[2].grid(False)

axes[3].imshow(X_all[n_un])

axes[3].set_title('Label: {:d}'.format(int(y_all[n_un])))

axes[3].grid(False)



plt.show()
np.random.seed(seed=42)

disorder = np.arange(X_all.shape[0])

print(disorder[:10])

np.random.shuffle(disorder)

print(disorder[:10])

X_all = X_all[disorder]

y_all = y_all[disorder]
# Test set (20%) for final rating. This set is to be used only at the very end!!

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state=42)



# Traing and validation set to optimize the CNN:

# Val set (20% from total => 25% from the X, y above)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state=42)



print("X_train shape: ", X_train.shape)

print("X_val shape:   ", X_val.shape)

print("X_test shape:  ", X_test.shape)



y_train = pd.get_dummies(y_train).values

y_val= pd.get_dummies(y_val).values

y_test = pd.get_dummies(y_test).values



print("")

print("y_train shape after encoding: ", y_train.shape)

print("y_val shape after encoding:   ", y_val.shape)

print("y_test shape after encoding:  ", y_test.shape)
# normalize only on training data

X_mean = np.mean(X_train, axis=0)

X_std = np.std(X_train, axis=0)



# apply the nomalized values from the training data to all sets

X_train_norm = np.array((X_train - X_mean)/(X_std + 0.0002), dtype="float32")

X_val_norm = np.array((X_val - X_mean)/(X_std + 0.0002), dtype="float32")

X_test_norm = np.array((X_test - X_mean)/(X_std + 0.0002), dtype="float32")
datagen = ImageDataGenerator(rotation_range=20,          #Int. Degree range for random rotations

                             width_shift_range=0.2,      #float: fraction of total width, if < 1, or pixels if >= 1.

                             height_shift_range=0.2,

                             horizontal_flip=True,

                             vertical_flip=True,

                             zoom_range=0.2)             #Float or [lower, upper],



train_generator = datagen.flow(x=X_train_norm, 

                               y=y_train, 

                               batch_size=batch_size, 

                               shuffle = True)
model = Sequential()



# CNN 1

model.add(Convolution2D(16, (3, 3), padding='same', input_shape=(pixel_x, pixel_y,3)))

model.add(Convolution2D(16, (1, 1), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))



# CNN 2

model.add(Convolution2D(16, (3, 3), padding='same'))

model.add(Convolution2D(16, (1, 1), padding='same'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())

model.add(Activation('relu'))



# CNN 3

model.add(Convolution2D(24, (3, 3), padding='same'))

model.add(Convolution2D(24, (1, 1), padding='same'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())

model.add(Activation('relu'))



# CNN 4

model.add(Convolution2D(24, (3, 3), padding='same'))

model.add(Convolution2D(24, (1, 1), padding='same'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())

model.add(Activation('relu'))



# fcNN

model.add(Flatten())

model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.evaluate(X_train_norm, y_train)
#nn = 'tensorboard/malaria/' + name + '/'

#tensorboard = keras.callbacks.TensorBoard(nn, write_graph=True, histogram_freq=1)
t = time()



history = model.fit_generator(train_generator,

                              steps_per_epoch = int(X_train_norm.shape[0]/10),

                              epochs=epochs,

                              verbose=1,

                              validation_data=(X_val_norm, y_val))#

                              #,callbacks=[tensorboard])



print("\nDelta Time for execution: {:.2f} min.\n".format((time() - t)/60))
model.save(name)
# summarize history for accuracy

import seaborn as sns; sns.set()



t_acc = max(history.history['acc'])

v_acc = max(history.history['val_acc'])



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

axes[0].plot(history.history['acc'], color='royalblue', label='training')

axes[0].plot(history.history['val_acc'], color='red', label='validation')

axes[0].set_title('Model Accuracy: train:  {:0.2f}%;  val: {:0.2f}%'.format(t_acc*100, v_acc*100))

axes[0].set_xlabel('epoch')

axes[0].set_ylabel('accuracy')

axes[0].set_ylim((-0.1, 1.1))

axes[0].legend(loc="lower right", frameon=True, shadow=True)



axes[1].plot(history.history['loss'], color='royalblue', label='training')

axes[1].plot(history.history['val_loss'], color='red', label='validation')

axes[1].set_title('Model loss (name:'+name+')')

axes[1].set_xlabel('epoch')

axes[1].set_ylabel('loss')

axes[1].set_ylim((-0.1, 2.5))

axes[1].legend(loc="upper right", frameon=True, shadow=True)

plt.tight_layout()

plt.savefig(name+'.png', dpi=400, bbox_inches='tight')

plt.show()
y_pred = model.predict(X_train_norm)



print("Accuracy = {:0.2f}%".format(np.sum(y_train[:,1] == np.argmax(y_pred, axis=1)) / len(y_pred) * 100))

print(classification_report(y_train[:,1], np.argmax(y_pred, axis=1)))
y_pred = model.predict(X_val_norm)



print("Accuracy = {:0.2f}%".format(np.sum(y_val[:,1] == np.argmax(y_pred, axis=1)) / len(y_pred) * 100))

print(classification_report(y_val[:,1], np.argmax(y_pred, axis=1)))
y_pred = model.predict(X_test_norm)



print("Acc = {:0.2f}%".format(np.sum(y_test[:,1] == np.argmax(y_pred, axis=1)) / len(y_pred) * 100))

print(classification_report(y_test[:,1], np.argmax(y_pred, axis=1)))