# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



import keras.layers.core as core

import keras.layers.convolutional as conv

import keras.models as models

import keras.utils.np_utils as kutils

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import classification_report

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.

img_rows, img_cols = 28, 28

def load_dataset(train_path,test_path):

    global train,test,trainX,trainY,nb_classes

    train = pd.read_csv(train_path).values # produces numpy array

    test  = pd.read_csv(test_path).values # produces numpy array

    print("Train Shape :",train.shape)

    trainX = train[:, 1:].reshape(train.shape[0], img_rows, img_cols, 1)

    trainX = trainX.astype(float)

    trainX /= 255.0

    trainY = kutils.to_categorical(train[:, 0])

    nb_classes = trainY.shape[1]

    print("TrainX Shape : ",trainX.shape)

    print("Trainy shape : ",trainY.shape)

    testX = test.reshape(test.shape[0], 28, 28, 1)

    testX = testX.astype(float)

    testX /= 255.0

    trainY = kutils.to_categorical(train[:, 0])

    return train,test,trainX,trainY,testX,nb_classes
def createModel(inp_shape,nClasses):

    model = models.Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inp_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(nClasses, activation='softmax'))

    

    # Define the optimizer

    optimizer1 = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    

    model.compile(optimizer=optimizer1, loss='categorical_crossentropy', metrics=['accuracy'])

 

    return model
def submission(prediction):

    np.savetxt('mnist-submission.csv', np.c_[range(1,len(prediction)+1),prediction], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
import matplotlib.pyplot as plt

def result_visualization(out):

    # Loss Curves

    plt.figure(figsize=[8,6])

    plt.plot(out.history['loss'],'r',linewidth=3.0)

    plt.plot(out.history['val_loss'],'b',linewidth=3.0)

    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

    plt.xlabel('Epochs ',fontsize=16)

    plt.ylabel('Loss',fontsize=16)

    plt.title('Loss Curves',fontsize=16)

 

    # Accuracy Curves

    plt.figure(figsize=[8,6])

    plt.plot(out.history['acc'],'r',linewidth=3.0)

    plt.plot(out.history['val_acc'],'b',linewidth=3.0)

    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

    plt.xlabel('Epochs ',fontsize=16)

    plt.ylabel('Accuracy',fontsize=16)

    plt.title('Accuracy Curves',fontsize=16)
def mnist_eda(Y_train):

    g = sns.countplot(Y_train)

    Y_train.value_counts()
def classification_report(X_test,test):

    #get the predictions for the test data

    predicted_classes = model.predict_classes(X_test)



    #get the indices to be plotted

    y_true = test.iloc[:, 0]

    correct = np.nonzero(predicted_classes==y_true)[0]

    incorrect = np.nonzero(predicted_classes!=y_true)[0]

    target_names = ["Class {}".format(i) for i in range(num_classes)]

    print(classification_report(y_true, predicted_classes, target_names=target_names))

# Main

train_path="../input/train.csv"

test_path="../input/test.csv"

#test and testX are test dataset used for evaluation, train and trainX,trainY are training datasets

train,test,trainX,trainY,testX,nb_classes=load_dataset(train_path,test_path)

# Splitting dataset for training and testing

X_train, X_test, y_train, y_test = train_test_split(trainX,trainY,test_size=0.1, random_state=21)

#Model Creation

inp_shape=(28,28,1)

model=createModel(inp_shape,nb_classes)

#Training with Image Augmentation

imgaug=False

batch_size=128 #256

nb_epochs=30

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

if imgaug==True:

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

    datagen.fit(trainX)

    out = model.fit_generator(datagen.flow(trainX,trainY, batch_size=batch_size),

                              epochs = nb_epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=batch_size // batch_size

                              , callbacks=[learning_rate_reduction] )

#Training

else:

    out=model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1,

             validation_data=(X_test, y_test))

    





#Prediction

yPred = model.predict_classes(testX)

print("Predictions : ",yPred)

#Submission of results

submission(yPred)



#Result Visualiztion

result_visualization(out)



#Classification Report

#classification_report(X_test,test)