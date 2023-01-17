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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tqdm import tqdm_notebook 

from sklearn.metrics import accuracy_score, mean_squared_error



from keras.utils.np_utils import to_categorical

from keras.models import Sequential #to initilize a NN

from keras.layers import Convolution2D #to convolution of i/p images

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense #to add layers to a NN
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 

X_test = test
X_train = X_train.values

for i in range (0,X_train.shape[0]):

    X_train[i,:] = np.array(list(map(lambda x: 1 if x > 0 else 0, X_train[i,:])))



X_test = X_test.values    

for i in range (0,X_test.shape[0]):

    X_test[i,:] = np.array(list(map(lambda x: 1 if x > 0 else 0, X_test[i,:])))

    
Y_train = to_categorical(Y_train, num_classes = 10)
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)
classifier = Sequential()

classifier.add(Convolution2D(filters = 32, kernel_size = (5,5), input_shape = (28, 28, 1), activation = 'relu'))

classifier.add(Convolution2D(filters = 32, kernel_size = (5,5), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(filters = 16, kernel_size = (3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 32, activation = 'relu'))

classifier.add(Dense(output_dim = 16, activation = 'relu'))

classifier.add(Dense(output_dim = 10, activation = 'softmax'))



#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

classifier.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
epochs = 5

batch_size = 86
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





datagen.fit(X_train)
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

#          validation_data = (X_val, Y_val), verbose = 2)



history = classifier.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] )
results = classifier.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn.csv",index=False)
results.head()

'''Y_pred_train = ffsnn.predict(X_train)

Y_pred_binarised_train = np.zeros(Y_pred_train.shape)

for i in range (Y_pred_train.shape[0]):

    Y_pred_binarised_train[i,:] = (Y_pred_train[i,:] == max(Y_pred_train[i,:])).astype("int")

Y_pred_binarised_train = Y_pred_binarised_train.ravel()    

Y_pred_val = ffsnn.predict(X_val)

Y_pred_binarised_val = np.zeros(Y_pred_val.shape)

for i in range (Y_pred_val.shape[0]):

    Y_pred_binarised_val[i,:] = (Y_pred_val[i,:] == max(Y_pred_val[i,:])).astype("int")

Y_pred_binarised_val = Y_pred_binarised_val.ravel() 

accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train.ravel())

accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val.ravel())



print("Training accuracy", round(accuracy_train, 5))

print("Validation accuracy", round(accuracy_val, 5))'''