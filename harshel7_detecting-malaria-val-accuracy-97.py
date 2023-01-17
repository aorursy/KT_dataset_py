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
import cv2

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from PIL import Image

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils
parasitized_data = os.listdir('../input/cell_images/cell_images/Parasitized/')

print(parasitized_data[:10]) #the output we get are the .png files



uninfected_data = os.listdir('../input/cell_images/cell_images/Uninfected/')

print('\n')

print(uninfected_data[:10])
plt.figure(figsize = (12,12))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Parasitized' + "/" + parasitized_data[i])

    plt.imshow(img)

    plt.title('PARASITIZED : 1')

    plt.tight_layout()

plt.show()
plt.figure(figsize = (12,12))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i+1])

    plt.imshow(img)

    plt.title('UNINFECTED : 0')

    plt.tight_layout()

plt.show()
data = []

labels = []

for img in parasitized_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Parasitized/' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(1)

    except:

        None

        

for img in uninfected_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Uninfected' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        data.append(img_array)

        labels.append(0)

    except:

        None
plt.imshow(data[0])

plt.show()
image_data = np.array(data)

labels = np.array(labels)
idx = np.arange(image_data.shape[0])

np.random.shuffle(idx)

image_data = image_data[idx]

labels = labels[idx]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 101)
y_train = np_utils.to_categorical(y_train, num_classes = 2)

y_test = np_utils.to_categorical(y_test, num_classes = 2)
print(f'SHAPE OF TRAINING IMAGE DATA : {x_train.shape}')

print(f'SHAPE OF TESTING IMAGE DATA : {x_test.shape}')

print(f'SHAPE OF TRAINING LABELS : {y_train.shape}')

print(f'SHAPE OF TESTING LABELS : {y_test.shape}')
import keras

from keras.layers import Dense, Conv2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Dropout

from keras.models import Sequential

from keras import backend as K



from keras import optimizers
def CNNbuild(height, width, classes, channels):

    model = Sequential()

    

    inputShape = (height, width, channels)

    chanDim = -1

    

    if K.image_data_format() == 'channels_first':

        inputShape = (channels, height, width)

    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = inputShape))

    model.add(MaxPooling2D(2,2))

    model.add(BatchNormalization(axis = chanDim))

    model.add(Dropout(0.2))



    model.add(Conv2D(32, (3,3), activation = 'relu'))

    model.add(MaxPooling2D(2,2))

    model.add(BatchNormalization(axis = chanDim))

    model.add(Dropout(0.2))



    model.add(Conv2D(32, (3,3), activation = 'relu'))

    model.add(MaxPooling2D(2,2))

    model.add(BatchNormalization(axis = chanDim))

    model.add(Dropout(0.2))



    model.add(Flatten())

    

    model.add(Dense(512, activation = 'relu'))

    model.add(BatchNormalization(axis = chanDim))

    model.add(Dropout(0.5))

    model.add(Dense(classes, activation = 'softmax'))

    

    return model
#instantiate the model

height = 50

width = 50

classes = 2

channels = 3

model = CNNbuild(height = height, width = width, classes = classes, channels = channels)

model.summary()
#compile the model

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
#fit the model onto the dataset

h = model.fit(x_train, y_train, epochs = 20, batch_size = 32)
plt.figure(figsize = (18,8))

plt.plot(range(20), h.history['acc'], label = 'Training Accuracy')

plt.plot(range(20), h.history['loss'], label = 'Taining Loss')

#ax1.set_xticks(np.arange(0, 31, 5))

plt.xlabel("Number of Epoch's")

plt.ylabel('Accuracy/Loss Value')

plt.title('Training Accuracy and Training Loss')

plt.legend(loc = "best")



# ax2.plot(range(20), h.history['loss'], label = 'Training Loss')

# ax2.plot(range(20), h.history['val_loss'], label = 'Validation Loss')

# #ax2.set_xticks(np.arange(0, 31, 5))

# ax2.set_xlabel("Number of Epoch's")

# ax2.set_ylabel('Loss Value')

# ax2.set_title('Training Loss vs Validation Loss')

# ax2.legend(loc = "best")
#evaluate the model on test data

predictions = model.evaluate(x_test, y_test)
print(f'LOSS : {predictions[0]}')

print(f'ACCURACY : {predictions[1]}')
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1/255.,

                                  horizontal_flip = True,

                                  width_shift_range = 0.2,

                                  height_shift_range = 0.2,

                                  fill_mode = 'nearest',

                                  zoom_range = 0.3,

                                  rotation_range = 30)

val_datagen = ImageDataGenerator(rescale = 1/255.)



train_generator = train_datagen.flow(x_train, y_train, batch_size = 64, shuffle = False)

val_generator = val_datagen.flow(x_test, y_test, batch_size = 64, shuffle = False)
#calling the same model as above

model_aug = CNNbuild(height = height, width = width, classes = classes, channels = channels)
#compile the model

optim = optimizers.Adam(lr = 0.001, decay = 0.001 / 64)

model_aug.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on the augmented dataset

h_aug = model_aug.fit_generator(train_generator, steps_per_epoch = len(x_train) // 64, epochs = 50)
#evaluate the model on augmented test data

predict = model_aug.evaluate_generator(val_generator, steps = 5)
print(f'LOSS ON TEST DATA AFTER DATA AUGMENTATION : {predict[0]}')

print(f'ACCURACY ON TEST DATA AFTER DATA AUGMENTATION : {predict[1]}')