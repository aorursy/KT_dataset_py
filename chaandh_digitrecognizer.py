# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#print training set 

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train.head()
#print test set

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.head()
# Seperate out image data and label

y_train = train["label"]

x_train = train.drop(["label"],axis=1)

y_train.value_counts()
#Normalization for faster convergence

x_train = x_train/255.0
#Reshape datasets

x_train = x_train.values.reshape(-1,28,28,1)

x_test = test.values.reshape(-1,28,28,1)

from keras.utils import to_categorical

y_train = to_categorical(y_train,num_classes=10)
#Split data to training and cross vaildation set

from sklearn.model_selection import train_test_split

X_train, X_cv, Y_train, Y_cv = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
# Visualizing the training data

plt.imshow(X_train[25][:,:,0])

plt.show()
#CNN model

#Using relu activation in the hidden layers for adding non-linearity to network

#Using softmax activation at ouput layer to give output in terms of probability

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import Flatten

model = Sequential()

model.add(Conv2D(filters=20,kernel_size=5,strides=(1,1),padding="valid",activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=20,kernel_size=5,strides=(1,1),padding="valid",activation='relu'))

model.add(Conv2D(filters=50,kernel_size=5,strides=(1,1),padding="valid",activation='relu'))

model.add(Conv2D(filters=50,kernel_size=5,strides=(1,1),padding="valid",activation='relu'))

model.add(Conv2D(filters=50,kernel_size=5,strides=(1,1),padding="valid",activation='relu'))

model.add(Conv2D(filters=50,kernel_size=5,strides=(1,1),padding="valid",activation='relu'))

model.add(Flatten())

model.add(Dense(units=500,activation='relu'))

model.add(Dense(units=10,activation='softmax'))
# Specifying optimizer and cost fuction. RMSprop is faster version of 

# gradient descent

from keras.optimizers import RMSprop

opt = RMSprop(learning_rate=0.01, rho=0.9)

model.compile(loss="categorical_crossentropy",optimizer=opt)

# Using data augmentation for giving more generalization capability to network.

#model.fit(X_train,Y_train,batch_size=32,epochs=2,validation_data=(X_cv, Y_cv))

from keras.preprocessing.image import ImageDataGenerator

dataAugmentation = ImageDataGenerator(rotation_range = 10,

                                     width_shift_range = 0.1,

                                     height_shift_range = 0.1,

                                     zoom_range = 0.1,

                                     horizontal_flip = False,

                                     vertical_flip = False)

#Fit the model to training data

model.fit_generator(dataAugmentation.flow(X_train,Y_train,batch_size=32), epochs=2, validation_data=(X_cv, Y_cv))

#Predict the output for test set

y_test = model.predict(x_test)
#Visualizing the predicted output for one sample

plt.imshow(x_test[25000][:,:,0])

plt.show()

print(y_test[25000])
# Construct labels from the prediction output and put it in .csv file

labels = np.argmax(y_test, axis=1)

imageIds = np.arange(1,28001)

output = pd.DataFrame({'ImageId':imageIds, 'Label':labels})

output.to_csv('output.csv', index=False)

print(output)

print("Your submission is successful.")