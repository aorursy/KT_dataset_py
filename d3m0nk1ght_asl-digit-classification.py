# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/Sign-language-digits-dataset"))

import numpy as np

import matplotlib.pyplot as plt

X=np.load('../input/Sign-language-digits-dataset/X.npy')

Y=np.load('../input/Sign-language-digits-dataset/Y.npy')



from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.15, random_state=0)

#Display one train and one test image.

plt.imshow(X_train[0])

print(Y_train[0])

plt.title('First train')

plt.show()

plt.imshow(X_test[0])

print(Y_test[0])

plt.title('First test')

plt.show()









# Any results you write to the current directory are saved as output.
##Data preprocessing

#Data type of images to be trained

print(type(X_train))

#Shape of train and test images

print(X_train[0].shape)

print(X_test[1].shape)

#Reshaping train and test images

print(X_train.size)

print(X_test.size)

#Reshaping train and test images

X_train=X_train.reshape(1752,64,64,1)

X_test=X_test.reshape(310,64,64,1)

print(X_train.shape)

print(X_test.shape)



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout



##Initialise the model

model=Sequential()

##Add layers to the models

#Add Conv1 layer

model.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(64,64,1)))

#model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(64,64,1)))

#Add pool1 layer

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#Conv2 layer

model.add(Conv2D(32, kernel_size=3, activation='relu'))

#model.add(Conv2D(32, kernel_size=3, activation='relu'))

#Add pool2 layer

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.25))

#Conv3 layer

model.add(Conv2D(32, kernel_size=3, activation='relu'))

#model.add(Conv2D(32, kernel_size=3, activation='relu'))

#model.add(Conv2D(64, kernel_size=3, activation='relu'))

#Add pool3 layer

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.25))

#Flatten the  output from Conv 2 layer

model.add(Flatten())

#Add a desnse layer

model.add(Dense(128, activation='relu'))

#model.add(Dropout(0.5))

#Finally add the output dense layer with softmax function activation

model.add(Dense(10,activation='softmax'))

#model.add(Dropout(0.5))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=8)

cls=model.predict_classes((X_test))

print(cls[0])


