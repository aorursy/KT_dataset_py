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
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv',sep=',')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv',sep=',')
train_df.head()
test_df.head()
train_data = np.array(train_df, dtype = 'float32')

test_data = np.array(test_df, dtype='float32')
x_train = train_data[:,1:]/255 #Skip 1st column as it is a label data

y_train = train_data[:,0] # 1st column is label

x_test= test_data[:,:]/255
from sklearn.model_selection import train_test_split

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 1)

print("x_train shape: " + str(x_train.shape))

print("x_validate shape: " + str(x_validate.shape))

print("x_test shape: " + str(x_test.shape))

print("y_train shape: " + str(y_train.shape))

print("y_validate shape: " + str(y_validate.shape))
height = width = 28

x_train = x_train.reshape(x_train.shape[0],height,width,1)

x_validate = x_validate.reshape(x_validate.shape[0],height,width,1)

x_test = x_test.reshape(x_test.shape[0],height,width,1)

print("x_train shape: " + str(x_train.shape))

print("x_validate shape: " + str(x_validate.shape))

print("x_test shape: " + str(x_test.shape))
from keras.preprocessing.image import ImageDataGenerator



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

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
from keras.models import Sequential

from keras.layers import Activation,Conv2D, MaxPooling2D, Dense, Dropout, Flatten



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='glorot_uniform',input_shape=(height, width, 1),name='conv0'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool0'))

model.add(Dropout(0.25))

          

model.add(Conv2D(64, kernel_size=(3, 3), name='conv1'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),name='max_pool1'))

model.add(Dropout(0.25))

          

model.add(Conv2D(128, (3, 3), activation='relu', name='conv2'))



model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu',name = 'fc'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.summary()
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.utils import plot_model



plot_model(model,to_file='model.png',show_shapes = True, show_layer_names = True)

img = mpimg.imread('model.png')

plt.figure(figsize =(40,40))

plt.imshow(img)
model.compile(loss ='sparse_categorical_crossentropy', optimizer= 'Adam',metrics =['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath = 'cnn.hdf5', verbose = 1, save_best_only = True)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size= 128), epochs = 50, verbose=1,callbacks = [checkpoint],validation_data=(x_validate,y_validate))
predicted_classes = model.predict_classes(x_test)

print(predicted_classes)
sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv',sep = ',')

sample_submission.head()
submission  = pd.DataFrame({

    "ImageId": range(1,x_test.shape[0] + 1),

    "Label": predicted_classes

})

submission.to_csv("submission.csv", index=False)

display(submission.head(3))

display(submission.tail(3))