# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from keras.datasets import mnist

import keras.utils.np_utils as ku

import keras.models as models

import keras.layers as layers

from keras import regularizers

import numpy.random as nr



import keras

from keras.layers import Dropout

from keras.optimizers import rmsprop, Adam

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator



import warnings

warnings.simplefilter(action='ignore')



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.head()
target=train['label']

train.drop('label',axis=1, inplace=True)

plt.figure(figsize=(15,5))

sns.countplot(target, color='crimson')

plt.title('The distribution of the digits in the dataset', weight='bold', fontsize='18')

plt.xticks(weight='bold', fontsize=16)

plt.show()
train=train/255

test=test/255
Train=train.values.reshape(-1,28,28,1)

Test=test.values.reshape(-1,28,28,1)
plt.figure(figsize=(15,8))

for i in range(60):

    plt.subplot(6,10,i+1)

    plt.imshow(Train[i].reshape((28,28)),cmap='binary')

    plt.axis("off")

plt.show()
Target=ku.to_categorical(target, num_classes=10)
print("The shape of the labels before One Hot Encoding",target.shape)

print("The shape of the labels after One Hot Encoding",Target.shape)

print("We have 10 columns for the 10 digits")
print("Shape of the first image with label: '1' after OHE")

print(Target[0])
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(Train, Target, test_size=0.10, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
nn=models.Sequential()
## Add some convolutional layers to extract features = Feature map



nn.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

nn.add(layers.MaxPooling2D((2, 2)))

nn.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

nn.add(layers.MaxPooling2D((2, 2)))

nn.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

nn.add(layers.MaxPooling2D((2, 2)))

nn.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

nn.add(layers.MaxPooling2D((2, 2)))





## latten the output of the convolutional layers so that fully connected network can be applied

nn.add(layers.Flatten())



## Finally, fully connected layers to classify the digits using the extracted features

nn.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

nn.add(Dropout(0.5))

nn.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

nn.add(Dropout(0.5))

nn.add(layers.Dense(10, activation = 'softmax'))



nn.summary()
nn1=models.Sequential()
## Add some convolutional layers to extract features = Feature map

nn1=models.Sequential()

nn1.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

nn1.add(BatchNormalization())

nn1.add(layers.MaxPooling2D((2, 2)))

nn1.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = (28, 28, 1)))

nn1.add(BatchNormalization())

nn1.add(layers.MaxPooling2D((2, 2)))

nn1.add(BatchNormalization())

nn1.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))

nn1.add(BatchNormalization())

nn1.add(layers.MaxPooling2D((2, 2)))



## latten the output of the convolutional layers so that fully connected network can be applied

nn1.add(layers.Flatten())



## Finally, fully connected layers to classify the digits using the extracted features

nn1.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

nn1.add(Dropout(0.5))

nn1.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

nn1.add(Dropout(0.5))

nn1.add(layers.Dense(10, activation = 'softmax'))



nn1.summary()
nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
nn1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
filepath = 'my_model_file.hdf5' # define where the model is saved

callbacks_list = [

        keras.callbacks.EarlyStopping(

            monitor = 'val_loss', # Use accuracy to monitor the model

            patience = 5 # Stop after 5 steps with lower accuracy

        ),

        keras.callbacks.ModelCheckpoint(

            filepath = filepath, # file where the checkpoint is saved

            monitor = 'val_loss', # Don't overwrite the saved model unless val_loss is worse

            save_best_only = True)]# Only save model if it is the best
datagen = ImageDataGenerator(

        rotation_range=15,

        zoom_range = 0.15,

        width_shift_range=0.15,

        height_shift_range=0.15)

datagen.fit(x_train)
history = nn.fit_generator(datagen.flow(x_train, y_train, batch_size=128),

                              epochs = 25, validation_data = (x_test,y_test),

                              steps_per_epoch=len(x_train) / 128, 

                              callbacks=callbacks_list)
history2 = nn1.fit_generator(datagen.flow(x_train, y_train, batch_size=128),

                              epochs = 25, validation_data = (x_test,y_test),

                              steps_per_epoch=len(x_train) / 128, verbose=1,

                              callbacks=callbacks_list)
predicted = nn.predict(x_test)
predicted2 = nn1.predict(x_test)
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,10))

#First Model

ax1 = plt.subplot2grid((2,2),(0,0))

train_loss = history.history['loss']

test_loss = history.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model 1: Loss vs. Epoch',weight='bold', fontsize=18)

ax1 = plt.subplot2grid((2,2),(0,1))

train_acc = history.history['accuracy']

test_acc = history.history['val_accuracy']

x = list(range(1, len(test_acc) + 1))

plt.plot(x, test_acc, color = 'cyan', label = 'Test accuracy')

plt.plot(x, train_acc, label = 'Training accuracy')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Model 1: Accuracy vs. Epoch', weight='bold', fontsize=18)  



#Second Model



ax1 = plt.subplot2grid((2,2),(1,0))

train_loss = history2.history['loss']

test_loss = history2.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model 2: Loss vs. Epoch',weight='bold', fontsize=18)



ax1 = plt.subplot2grid((2,2),(1,1))

train_acc = history2.history['accuracy']

test_acc = history2.history['val_accuracy']

x = list(range(1, len(test_acc) + 1))

plt.plot(x, test_acc, color = 'cyan', label = 'Test accuracy')

plt.plot(x, train_acc, label = 'Training accuracy')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Model 2: Accuracy vs. Epoch', weight='bold', fontsize=18)  



plt.show()
from sklearn.metrics import confusion_matrix



y_class = np.argmax(predicted, axis = 1) 



y_check = np.argmax(y_test, axis = 1) 



cmatrix = confusion_matrix(y_check, y_class)



plt.figure(figsize=(15,8))

plt.title('Confusion matrix of the test/predicted digits ', weight='bold', fontsize=18)

sns.heatmap(cmatrix,annot=True,cmap="Reds",fmt="d",cbar=False)

#We use np.argmax with y_test and predicted values: transform them from 10D vector to 1D

class_y = np.argmax(y_test,axis = 1) 

class_num=np.argmax(predicted, axis=1)

#Detect the errors

errors = (class_num - class_y != 0)

#Localize the error images

predicted_er = predicted[errors]

y_test_er = y_test[errors]

x_test_er = x_test[errors]



                

#Plot the misclassified numbers

plt.figure(figsize=(15,9))



for i in range(30):

    plt.subplot(5,6,i+1)

    plt.imshow(x_test_er[i].reshape((28,28)),cmap='binary')

    plt.title( np.argmax(predicted_er[i]), size=13, weight='bold', color='red')

    plt.axis("off")





plt.show()
plt.figure(figsize=(15,8))



for i in range(30):

    plt.subplot(5,6,i+1)

    plt.imshow(Train[i].reshape((28,28)),cmap='binary')

    plt.axis("off")



plt.show()
final = nn.predict(Test)

final = np.argmax(final,axis = 1)

final = pd.Series(final, name="Label")
submission = pd.concat([pd.Series(range(1,len(Test)+1),name = "ImageId"),final],axis = 1)



submission.to_csv("CNN_digit_recognizer.csv", index=False)
