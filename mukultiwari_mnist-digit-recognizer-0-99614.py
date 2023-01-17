# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Importing the Required Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing Keras and required libraries



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.optimizers import Adam


# Importing the dataset



training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
training_data.head()
testing_data.head()

training_data.info()
testing_data.info()


y = training_data.iloc[:, 0:1].values



X = training_data.iloc[:, 1:].values

X = X/255.0

testing_data = testing_data.values/255.0

X = X.reshape(-1, 28, 28, 1)



testing_data = testing_data.reshape(-1, 28, 28, 1)
encoder = OneHotEncoder(categorical_features=[0])

y = encoder.fit_transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17)
plt.imshow(X_train[1][:,:,0])

plt.show()
# Structuring the CNN
# Building the convolutional neural network



classifier = Sequential()



# Adding the first 2 convolution layer



classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))

classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))



# Pooling layer



classifier.add(MaxPooling2D(pool_size=(2, 2)))



classifier.add(Dropout(.1))



# Adding the second 2 convolution layer



classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(Conv2D(64, (3, 3), activation='relu'))



# 2nd Pooling layer



classifier.add(MaxPooling2D(pool_size=(2, 2)))



classifier.add(Dropout(.1))



# Adding the flattening layer



classifier.add(Flatten())



# Adding the ANN



classifier.add(Dense(units=300, activation='relu'))

classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=150, activation='relu'))

classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=10, activation='softmax'))
# Dynamic Reduction of Learning rate when accuracy reaches plateau



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.25, patience=2, min_lr=0.0001)
# Compiling the ANN



classifier.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])



# Fitting the CNN on DATASET using the keras documentation



train_datagen = ImageDataGenerator(shear_range=0.2,

                                   zoom_range=0.2,

                                   height_shift_range=.1,

                                   rotation_range=10,

                                   width_shift_range=.1)



train_datagen.fit(X_train)


# Fitting the Generator on Network



classifier.fit_generator(

        train_datagen.flow(X_train,y_train, batch_size=40),

        steps_per_epoch=X_train.shape[0] // 40,

        epochs=40,

        validation_data=(X_test,y_test),callbacks=[reduce_lr])
# Prediction



submission = classifier.predict(testing_data)



# Maximum Probability Index



submission = np.argmax(submission, axis = 1)

submission = pd.Series(submission,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)



submission.to_csv("cnn_mnist_v4.csv",index=False)