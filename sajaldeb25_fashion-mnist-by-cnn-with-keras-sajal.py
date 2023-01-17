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
from keras.utils import to_categorical

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(data_train.iloc[:, 1:])  # select from 2nd position to end

y = to_categorical(np.array(data_train.iloc[:, 0]))  # select only 1st position 



#Here we split validation data to optimiza classifier during training

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))



# as data is in 28X28, but conv2d works with 3 dimensions, 

# so we have to reshape the data with (no_of_sample, height, width, dimensions) like (60,000 , 28, 28, 1)



X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1) 

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



# Since the dataset fits easily in RAM, we might as well convert to float immediately.

# Regarding the division by 255, this is the maximum value of a byte (the input feature's type before

# the conversion to float32), so this will ensure that the input features are scaled between 0.0 and 1.0.



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
import keras

# Here we use Sequential Model for Classification

from keras.models import Sequential  



# We use the ‘add()’ function to add layers to our model. Our first 2 layers are Conv2D layers. 

# These are convolution layers that will deal with our input images, which are seen as 2-dimensional matrices.

from keras.layers import Dense, Dropout, Flatten



# A Dense layer is required to look at the output of the final convolutional neurons/filters and

# output a number (or numbers if one hot encoding is being used) as the classification.



# MaxPooling2D(): Constructs a two-dimensional pooling layer using the max-pooling algorithm.

from keras.layers import Conv2D, MaxPooling2D





from keras.layers.normalization import BatchNormalization
batch_size = 256

num_classes = 10

epochs = 50



#input image dimensions

img_rows, img_cols = 28, 28



model = Sequential()



# relu and softmax is just a activation function

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal',input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))



# Dropout is used to overcome the problem of Overﬁtting,

#  The dropout rate is set to 25%, meaning one in 4 inputs will be randomly excluded from each update cycle.

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))



# Once the pooled featured map is obtained, the next step is to flatten it. Flattening involves

# transforming the entire pooled feature map matrix into a single column which is then fed to the 

# neural network for processing.

model.add(Flatten())



model.add(Dense(128, activation='relu'))  

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))



# loss function compute the losses and optimizer function optimize the value of weight and baises

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
# after making all layer and activation function we input the images and fit it with the model

# fitting the model

model.fit(X_train, y_train,batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(X_val, y_val))
# evaluating the model

# model.evaluate(x_test, y_test)

score = model.evaluate(X_test, y_test)
# printing the loss and Accurecy 



print('Test loss:', score[0])

print('Test accuracy(in percent):', score[1]*100)


