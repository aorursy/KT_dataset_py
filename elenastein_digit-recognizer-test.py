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
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# we insert a blank label column to ease with processing afterwards

test.insert(0, 'label', 0, True)

test.head()
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data.head()
max_pixel_value = data.iloc[:,1:].max().max()

print(max_pixel_value)

num_images = data.shape[0]
from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
# the labels are in the label column, the grey scale images 28 x 28 reformed to 783 columns 

def preprocess(data):

# here keras converts the catgorical y data to hot encoded values

    y = keras.utils.np_utils.to_categorical(data.label, 10)

    num_images = data.shape[0]

    x_array = data.iloc[:,1:].values

    x_shaped = x_array.reshape(num_images, 28,28,1)

    # divide all pixel values by the maximum pixel value

    x = x_shaped/255

    return x, y



x_test = preprocess(test)

x, y = preprocess(data)



# here we are creating a neural network with two convolution layers adding strides and drop out 

# to reduce run times and create a neural network with an even distribution of weights

# since the batch size is half of the number of examples, with each step we are using half the number of examples for training the network



digit_model = Sequential()

digit_model.add(Conv2D(20,kernel_size=(3,3), strides=2, activation='relu', input_shape=(28,28,1)))

digit_model.add(Dropout(0.5))

digit_model.add(Conv2D(20, kernel_size=(3,3), strides=2, activation='relu'))

digit_model.add(Dropout(0.5))

digit_model.add(Flatten())

digit_model.add(Dense(128, activation='relu'))

digit_model.add(Dense(10, activation='softmax')) #here the 10 is the number of classes we are categorising into





digit_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

digit_model.fit(x,y, batch_size=128, epochs=2, validation_split=0.2)
y_test = digit_model.predict_classes(x_test)
import matplotlib.pyplot as plt

%matplotlib inline 

# visualize how well the predictions are

images = test.iloc[:,1:].values.reshape(test.shape[0], 28,28)

print(images.shape)

plt.subplot()

for i in range(1,6):

    plt.subplot(1,5,i)

    plt.imshow(images[i], cmap=plt.get_cmap('gray'))

    plt.title(y_test[i]);

# print the first five images with their predictions
# saving the predictions into the required format for submission

submissions=pd.DataFrame({"ImageId": list(range(1,len(y_test)+1)),

                         "Label": y_test})

submissions.to_csv("digit.csv", index=False, header=True)