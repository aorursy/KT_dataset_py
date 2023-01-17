# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load data set

train = pd.read_csv('../input/train.csv')

train.shape
test = pd.read_csv('../input/test.csv')

test.shape
train.head()
test.head()
#drop label column from train data set and create X:

X = train.drop('label', axis=1)

#y is the label column of train set:

y = train['label']
#distribution graph:

sns.countplot(y)
#example of image representation:



img = X.iloc[0].values

img = img.reshape((28, 28))



plt.imshow(img, cmap='gray')

plt.title(X.iloc[0,0])

plt.axis('off')

plt.show()
#we change the values to 0-1 scale:

X = X / 255.0

test = test / 255.0
X = X.values.reshape(-1,28,28,1)



test = test.values.reshape(-1,28,28,1)



print(X.shape)

print(test.shape)
from keras.utils.np_utils import to_categorical #convert to one-hot-encoding



y = to_categorical(y, num_classes = 10)
#Split the train and the validation set for the fitting:

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)



print('X_train shape:', X_train.shape)

print('X_val shape:', X_test.shape)

print('y_train shape:', y_train.shape)

print('y_val shape:', y_test.shape)
#import methods:



from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical #convert to one-hot coding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



#conv -> conv -> max pool -> dropout -> conv -> conv -> max pool -> dropout -> fully connected(2 layers)



#Convolutional layers:

model.add(Conv2D(filters=32, kernel_size = (5,5), padding='Same', activation='relu', input_shape=(28,28,1)))



model.add(Conv2D(filters=32, kernel_size = (5,5), padding='Same', activation='relu'))





#max pool:

model.add(MaxPool2D(pool_size=(2,2)))

#pool size yukarıda da 2x2 kullanmıştık.



#dropout:

model.add(Dropout(0.25))



#Convolutional layers:

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))



#max pool:

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))



#dropout:

model.add(Dropout(0.25))





#fully connected(2 layers):



#flatten:

model.add(Flatten())



#1 hidden layer:

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

#1 output layer:

model.add(Dense(10, activation='softmax'))

#Define the optimizer:

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#compile the model:

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 30

batch_size = 100
datagen = ImageDataGenerator(

        featurewise_center=False, #set input mean to 0 over the dataset

        samplewise_center=False, #set each sample mean to 0

        featurewise_std_normalization=False, #divide inputs by std of the dataset

        samplewise_std_normalization=False, #divide each input by its std

        zca_whitening=False, #dimension reduction

        rotation_range=10, #randomly rotate images in the range (degrees, 0 to 180)

        zoom_range=0.1, #Randomly Zoom by 10% some training images

        width_shift_range=0.1, #randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1, #randomly shift images vertically (fraction of total height)

        horizontal_flip=False, #randomly flip images

        vertical_flip=False) #randomly flip images



datagen.fit(X_train)
#fit the model:

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_test, y_test), steps_per_epoch=X_train.shape[0]//batch_size)
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
# predictions:

predictions = model.predict(test)



# select the index with the maximum probability

predictions = np.argmax(predictions,axis = 1)



predictions = pd.Series(predictions,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), predictions],axis = 1)



submission.to_csv("submission.csv",index=False)