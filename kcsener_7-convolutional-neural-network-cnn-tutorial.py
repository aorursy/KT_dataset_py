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

train = pd.read_csv('../input/cnn-train/train.csv')

train.shape
test = pd.read_csv('../input/cnn-test/test.csv')

test.shape
train.head()
test.head()
#drop label column from train data set and create X_train:

X_train = train.drop('label', axis=1)

#y_train is the label column of train set:

y_train = train['label']
#distribution graph:

sns.countplot(y_train)
#example of image representation:



img = X_train.iloc[0].as_matrix()

img = img.reshape((28, 28))



plt.imshow(img, cmap='gray')

plt.title(X_train.iloc[0,0])

plt.axis('off')

plt.show()
#we change the values to 0-1 scale:

X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)



test = test.values.reshape(-1,28,28,1)



print(X_train.shape)

print(test.shape)
from keras.utils.np_utils import to_categorical #convert to one-hot-encoding



y_train = to_categorical(y_train, num_classes = 10)
#Split the train and the validation set for the fitting:

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

#Burada bizim main datamız X ve y değil, X_train ve y_train olduğu için X_train, X_val, Y_train, Y_val şeklinde böldük.

#yani aslında test datasına hiç dokunmadık.

print('X_train shape:', X_train.shape)

print('X_val shape:', X_val.shape)

print('y_train shape:', y_train.shape)

print('y_val shape:', y_val.shape)

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



#conv -> max pool -> dropout -> conv -> max pool -> dropout -> fully connected(2 layers)

#Convolutional layer'ı ekliyoruz:

model.add(Conv2D(filters=8, kernel_size = (5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

#filters=8 adet filtre kullanacağız dedik, 

#kernel_size filtre size'ımız, 

#padding için 'same pedding' kullanıyoruz



#max pool -> dropout -> conv -> max pool -> dropout -> fully connected(2 layers)

#max pool ekliyoruz:

model.add(MaxPool2D(pool_size=(2,2)))

#pool size yukarıda da 2x2 kullanmıştık.



#dropout -> conv -> max pool -> dropout -> fully connected(2 layers)

#drop out'ta sıra:

model.add(Dropout(0.25))

#0.25 oranla ignore et dedik.



#conv -> max pool -> dropout -> fully connected(2 layers)

#Convolutional layer'ı ekliyoruz:

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='Same', activation='relu'))



#max pool -> dropout -> fully connected(2 layers)

#max pool ekliyoruz:

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#bu sefer stride ekliyoruz.



#dropout -> fully connected(2 layers)

#dropout ekliyoruz:

model.add(Dropout(0.25))



#fully connected(2 layers):

#fully connected yaparken önce flatten yapıcaz:

model.add(Flatten())

#1 adet hidden layer ekliyoruz:

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

#1 adet de output ekliyoruz: (2 layers bir hidden bir outputtan oluşuyordu)

model.add(Dense(10, activation='softmax'))

#Define the optimizer:

optimizer = Adam(lr=0.001, beta_1= 0.9, beta_2= 0.999)
#compile the model:

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10

batch_size = 250
datagen = ImageDataGenerator(

        featurewise_center=False, #set input mean to 0 over the dataset

        samplewise_center=False, #set each sample mean to 0

        featurewise_std_normalization=False, #divide inputs by std of the dataset

        samplewise_std_normalization=False, #divide each input by its std

        zca_whitening=False, #dimension reduction

        rotation_range=0.5, #randomly rotate images in the range 5 degrees

        zoom_range=0.5, #randomly zoom image 5%

        width_shift_range=0.5, #randomly shift images horizontally 5%

        height_shift_range=0.5, #randomly shift images vertically 5%

        horizontal_flip=False, #randomly flip images

        vertical_flip=False) #randomly flip images



datagen.fit(X_train)
#fit the model:

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, y_val), steps_per_epoch=X_train.shape[0]//batch_size)
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

y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()