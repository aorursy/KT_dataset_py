# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
X_train = train_data.drop(labels=['label'],axis=1)
Y_train = train_data['label']
del train_data

Y_train.value_counts().sort_index().plot.bar()
#Check for null values
X_train.isnull().any().describe()

test_data.isnull().any().describe()
#normalizing values for ease of computation
X_train = X_train/255.0
test_data = test_data/255.0
#reshaping data
X_train = X_train.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)

#convert Y_train to onehot
Y_train = keras.utils.to_categorical(Y_train,num_classes=10)
#Here we split data into two datasets
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size = 0.1)
#Visulaising a random data
plt.imshow(X_train[1729][:,:,0])
#Lets a function which returns LeNet-5 model so that we can use it on different optimisers

def my_model():
    lenet_model = keras.Sequential()
    lenet_model.add(keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='Same',activation='relu',input_shape = (28,28,1)))
    #lenet_model.add(keras.layers.AvgPool2D(pool_size=(2,2),strides=(2,2)))
    lenet_model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    lenet_model.add(keras.layers.Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
    #lenet_model.add(keras.layers.AvgPool2D(pool_size=(2,2),strides=(2,2)))
    lenet_model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    lenet_model.add(keras.layers.Flatten())
    lenet_model.add(keras.layers.Dense(units=84,activation='relu'))
    lenet_model.add(keras.layers.Dense(units=10,activation='softmax'))
    return lenet_model;
#Learning rate modification while training (change learning rate if accuracy is not increased in consecutive epochs)
learning_rate_annealer = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',patience=3,factor=0.5,verbose=1,min_lr=0.00001)
#Data Augmentation for better results
datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,horizontal_flip=False,rotation_range=10,samplewise_center=False,
                                                       samplewise_std_normalization=False,featurewise_std_normalization=False,height_shift_range=0.1,
                                                       width_shift_range=0.1,zca_whitening=False,vertical_flip=False)
datagen.fit(X_train)
#Using RMPprop optimizer
rms_optim = keras.optimizers.rmsprop(lr=0.01,rho = 0.9,decay=0.0)
cnn_rms = my_model()

cnn_rms.compile(optimizer=rms_optim,metrics=['accuracy'],loss=keras.losses.categorical_crossentropy)
epochs = 50
batch_size = 32
cnn_rms_hist = cnn_rms.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_val,Y_val),
                                     verbose=2,steps_per_epoch=X_train.shape[0]//batch_size,callbacks=[learning_rate_annealer])
optim_adam = keras.optimizers.adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
cnn_adam = my_model()
cnn_adam.compile(optimizer=optim_adam,metrics=['accuracy'],loss=keras.losses.categorical_crossentropy)
epochs = 100
batch_size = 32
cnn_adam_hist = cnn_adam.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_val,Y_val),
                                     verbose=2,steps_per_epoch=X_train.shape[0]//batch_size,callbacks=[learning_rate_annealer])
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 #confusion matrix for rms prop
Y_pred_rms = cnn_rms.predict(X_val)
Y_pred_classes_rms = np.argmax(Y_pred_rms,axis=1)
Y_true_rms = np.argmax(Y_val,axis=1)

cm_rms = confusion_matrix(Y_true_rms,Y_pred_classes_rms)

plot_confusion_matrix(cm_rms,classes=range(10))
    
 #confusion matrix for adam
Y_pred_adam = cnn_adam.predict(X_val)
Y_pred_classes_adam = np.argmax(Y_pred_adam,axis=1)
Y_true_adam = np.argmax(Y_val,axis=1)

cm_adam = confusion_matrix(Y_true_adam,Y_pred_classes_adam)

plot_confusion_matrix(cm_adam,classes=range(10))
#seems like adam is doing good...Lets get our results with adam model
predictions = cnn_adam.predict(test_data)
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions,name = 'Label')
submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),predictions],axis=1)
submission.to_csv("lenet5_digit_recogniser.csv",index=False)
