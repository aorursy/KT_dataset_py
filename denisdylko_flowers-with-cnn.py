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
import os

import shutil

import cv2 as cv

import numpy as np

from tqdm import tqdm



def isImage( filename ):

    if(filename.lower().endswith(('png','jpg','jpeg'))):

        return True

    else:

        return False



def get_class_labels( src_folder ):

    folders_list = [ target for target in os.listdir(src_folder) if( os.path.isdir(os.path.join(src_folder, target)) and (target != 'flowers') ) ]

    print(folders_list)

    return folders_list



def read_images( src_folder, label, result_size ):    

    images = []

    labels = []

    for filename in os.listdir(src_folder):

        target = os.path.join(src_folder,filename)

        if( os.path.isfile(target) and isImage(filename) ):

            img = cv.imread(str(target), cv.IMREAD_COLOR)

            img = cv.resize(img, result_size)

            images.append(img)

            labels.append(label)    

    return images, labels



def read_data_set(root_folder, folders, result_size):

    X = []

    Y = []    

    for i in tqdm(range(len(folders))):

        folder_x, folder_y = read_images( os.path.join(root_folder, folders[i]), i, result_size )

        X.append(np.array(folder_x))

        Y.append(np.array(folder_y))

    return X, Y



class_labels = get_class_labels('/kaggle/input/flowers-recognition/flowers/')



num_classes = len(class_labels)

print(num_classes)
from tensorflow.keras.utils import to_categorical

X, Y = read_data_set( '/kaggle/input/flowers-recognition/flowers/' , class_labels, (128,128))

X_train = []

X_test = []

Y_train = []

Y_test = []

split_ratio = 0.9



for i in tqdm(range(num_classes)):

        tensor = np.array(X[i])

        split_pos = int(round(tensor.shape[0]*split_ratio)) 

        split_config = [split_pos, tensor.shape[0] - split_pos]

        X_train.extend(tensor[:split_pos])

        X_test.extend(tensor[split_pos:])



        Y_train.extend(Y[i][:split_pos])

        Y_test.extend(Y[i][split_pos:])





DO_GRAYSCALE = False

if( DO_GRAYSCALE ):

        x_train = tf.image.rgb_to_grayscale(X_train)

        x_test = tf.image.rgb_to_grayscale(X_test)

else:

        x_train = X_train

        x_test = X_test





train_x = np.array(x_train).astype('float')/255.0

train_y = to_categorical(Y_train)



test_x = np.array(x_test).astype('float')/255.0

test_y = to_categorical(Y_test)



print(train_x.shape)

print(train_y.shape)



print(test_x.shape)

print(test_y.shape)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, SpatialDropout2D



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=train_x[0].shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dense(2048, activation='relu'))

model.add(Dense(1024, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(

        optimizer='SGD',

        loss='categorical_crossentropy',

        metrics=['acc']

)



model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False

) 



datagen.fit(train_x)

batch_size = 96

model.fit(

        x = datagen.flow(train_x, train_y, batch_size=batch_size),

        steps_per_epoch = train_x.shape[0] / batch_size,

        epochs=5,

        validation_data=(test_x, test_y)

        )