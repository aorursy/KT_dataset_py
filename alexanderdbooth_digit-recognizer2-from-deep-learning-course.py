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
#Mostly taken from the Deep Learning kaggle exercise, Deep Learning from scratch

#https://www.kaggle.com/alexanderdbooth/exercise-deep-learning-from-scratch

from sklearn.model_selection import train_test_split

from tensorflow import keras



#image size

img_rows, img_cols = 28, 28



#ten possible digits (classes), 0,1,2,3,4,5,6,7,8,9

num_classes = 10



#training data has x and y (labels)

def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



#test data only has x

def prep_data_test(raw_test): #only has x, no labels

    x = raw_test[:,0:]

    num_images = raw_test.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x



number_file = "../input/train.csv"

number_data = np.loadtxt(number_file, skiprows=1, delimiter=',')

x, y = prep_data(number_data)
np.shape(x) #42,000 training images
#read and prep test data

number_file_test = "../input/test.csv"

number_data_test = np.loadtxt(number_file_test, skiprows=1, delimiter=',')

x_test = prep_data_test(number_data_test)
np.shape(x_test) #28,000 test images
import matplotlib.pyplot as plt



#view a couple of the training images

for i in range(1,13):

    plt.subplot(3,4,i)

    plt.imshow(x[i-1].reshape([28,28]),cmap="gray")
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



#build the model

number_model = Sequential()

number_model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1))) #activation layer



#additional learning layers

number_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))



#final prediction layers

number_model.add(Flatten())

number_model.add(Dense(100, activation='relu'))

number_model.add(Dense(num_classes, activation='softmax'))



#compile the model

number_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
#initial fit with validation

number_model.fit(x, y,

          batch_size=100,

          epochs=3,

          validation_split = 0.2)
#Get predictions

preds_test = number_model.predict(x_test)

#the model returns a list of probabilities for each number the image could be. 

#argmax is a hack to get the most likely number

realPreds = [np.argmax(x) for x in preds_test]

realPreds[0:12]
#print a few of the test numbers to look at our predictions

for i in range(1,13):

    plt.subplot(3,4,i) #sub plots have to start at 1

    plt.imshow(x_test[i-1].reshape([28,28]),cmap="gray")

    

#looks like we are missing the fourth number
#Retrain using data augmentation for fun (again, Deep Learning Kaggle Course exercise)

#https://www.kaggle.com/alexanderdbooth/exercise-data-augmentation
#have to manually get our validation data when dealing with generators

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2)
#build the model, same as above

number_model_aug = Sequential()

number_model_aug.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1))) #activation layer



#additional learning layers

number_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

number_model_aug.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))



#final prediction layers

number_model_aug.add(Flatten())

number_model_aug.add(Dense(100, activation='relu'))

number_model_aug.add(Dense(num_classes, activation='softmax'))



#compile the model

number_model_aug.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

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



datagen.fit(X_train)
number_model_aug.fit_generator(datagen.flow(X_train,Y_train),

                              epochs = 15, validation_data = (X_val,Y_val), steps_per_epoch=20)
#Get predictions

preds_test_aug = number_model_aug.predict(x_test)

#the model returns a list of probabilities for each number the image could be. 

#argmax is a hack to get the most likely number

real_preds_aug = [np.argmax(x) for x in preds_test_aug]

real_preds_aug[0:12]
#print a few of the test numbers to look at our predictions

for i in range(1,13):

    plt.subplot(3,4,i) #sub plots have to start at 1

    plt.imshow(x_test[i-1].reshape([28,28]),cmap="gray")

    

#looks like we are missing the 4th and 5th numbers
# Save test predictions to file

# no aug performed better

output = pd.DataFrame({'ImageId': range(1,28001),

                       'Label': realPreds})

output.to_csv('submission.csv', index=False)