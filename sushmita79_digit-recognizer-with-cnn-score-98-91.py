# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
#we check that all the samples for numbers are almost equally distributed

train_data['label'].value_counts().plot(kind='bar')

#here we have our target variable as numbers from 0-9 , so we need to convert them to categorical variables as we don't want our models to give ordering preferences to numbers ex( 2<3<4)

#so we convert each number as an example label ‘3’ should be converted to a vector [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 

y_train=train_data['label']

x_train = train_data.drop('label',axis = 1) 

y_train = to_categorical(y_train, num_classes = 10)



#lets check a sample row to view the digit representation

plt.imshow(x_train.values[7].reshape(28,28),cmap='gray')
#normalization of pixels, as models will work faster on values between 0 and 1 than ranging from 0-255

x_train=x_train/255

test_data = test_data / 255

#We use reshape as the input values are in 1-D vector with 784 pixels but we want a 2-D form. Our Convolution Layer needs data in the below format:

#we must reshape the data into (28x28x1) 3D matrices. This is because Keras wants an Extra Dimension in the end, for channels. If this had been RGB images, there would have been 3 channels,

#but as MNIST is gray scale it only uses one.

x_train=x_train.values.reshape(-1,28,28,1)

test_data=test_data.values.reshape(-1,28,28,1)
#Create our CNN as below:



# 1)Sequential API to add linear stack of layers.

# 2) We add two Convolution layers which apply filters on the input and produce feauture maps(which are more aligned to our patterns/findings).

# 3)Pooling is used for Dimensionality Reduction. What if your image is rotated or at an angle? Pooling helps to reduce the amount of Parameters and thus helps

#   to learn more Complex features of the Image.

# 4)Finally we add Flatten layer to map the input to a 1D vector. 

classifier=Sequential()

#we add 32 filters of size 3*3 and rectilinear activation function is used 

classifier.add(Convolution2D(32,3,3,input_shape=(28,28,1),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
#Lastly, we add the Output Layer. It has units equal to the number of classes to be identified.

#Here, we use 'sigmoid' function if it is Binary Classification otherwise 'softmax' activation function in case of Multi-Class Classification.

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(10, activation = "softmax"))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#results obtained after 20 complete iterations

classifier.fit(x_train, y_train, batch_size = 64, epochs = 20)

results = classifier.predict(test_data)
#view the results by reversing the categorical var

results = np.argmax(results,axis = 1)



#submit the results , accuracy obtained 98.914%

