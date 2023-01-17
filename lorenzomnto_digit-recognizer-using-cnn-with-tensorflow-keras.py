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
from sklearn.model_selection import train_test_split



from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout

from keras.preprocessing.image import ImageDataGenerator
img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



train_size = 30000

train_file = '../input/train.csv'

test_file = '../input/test.csv'

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)



random_seed = 5

x, x_val, y, y_val = train_test_split(x, y, test_size = 0.2, random_state=random_seed)
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





datagen.fit(x)
model = Sequential()



model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(img_rows,img_cols,1)))



model.add(Conv2D(32, kernel_size=5, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.fit(x,y, batch_size=64, epochs=20, validation_split=0.2) #if we didn't use any data augmentation



batch_size = 64

model.fit_generator(datagen.flow(x,y, batch_size=batch_size),

                    validation_data= (x_val, y_val),

                    epochs = 20,

                    steps_per_epoch=x.shape[0] // batch_size)
test_data = pd.read_csv(test_file)

num_images = test_data.shape[0]

test_data_asarray = test_data.values[:,:]

test_data_shaped_array = test_data_asarray.reshape(num_images, img_rows, img_cols, 1)

t = test_data_shaped_array / 255
predictions = model.predict(t)
predictions_readable = [prediction.argmax() for prediction in predictions]

submission = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions_readable})

submission.to_csv('submission.csv',index=False,header=True)