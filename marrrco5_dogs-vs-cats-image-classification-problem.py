#This kernel aims to build a CNN model to solve the Kaggle Dogs vs. Cats image classification problem
%xmode Plain

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import RMSprop
def result_visualisation(loss,acc,val_loss,val_acc):

    #Result visualisation

    epochs = np.arange(1,len(loss)+1)

    fig,ax = plt.subplots(1,2,figsize=(20,5))

    ax[0].plot(epochs,loss,label='loss')

    ax[0].plot(epochs,val_loss,label='val_loss')

    ax[0].set_title('Loss')

    ax[0].set_xlabel('Epochs')

    ax[0].legend(['loss','val_loss'])

    ax[1].plot(epochs,acc,label='acc')

    ax[1].plot(epochs,val_acc,label='val_acc')

    ax[1].set_title('Accuracy')

    ax[1].set_xlabel('Epochs')

    ax[1].legend(['acc','val_acc'])

    plt.tight_layout()

    plt.show();
def plot_val_acc_comparison(val_acc_list,smooth=False):

    legend = []

    if smooth:

        smooth_list = []

        for i,val_acc in enumerate(val_acc_list):

            smooth_list.append(smoothing(val_acc))

        data_list = smooth_list

    else:

        data_list = val_acc_list

    for i,val_acc in enumerate(data_list):

        plt.plot(val_acc)

        legend.append('val_acc'+str(i+1))

    plt.title('Validation Accuracy across all networks')

    plt.xlabel('Epochs')

    plt.legend(legend)

    plt.tight_layout()

    plt.show();
def plot_val_loss_comparison(val_loss_list,smooth=False):

    legend = []

    if smooth:

        smooth_list = []

        for i,val_loss in enumerate(val_loss_list):

            smooth_list.append(smoothing(val_loss))

        data_list = smooth_list

    else:

        data_list = val_loss_list

    for i,val_loss in enumerate(data_list):

        plt.plot(val_loss)

        legend.append('val_loss'+str(i+1))

    plt.title('Validation Loss across all networks')

    plt.xlabel('Epochs')

    plt.legend(legend)

    plt.tight_layout()

    plt.show();
def smoothing(data,factor=0.4):

    smooth = []

    for item in data:

        if smooth:

            smooth.append(factor*smooth[-1]+(1-factor)*item)

        else:

            smooth.append(item)

    return smooth
#Directories of data

train_dir = '../input/dogs-vs-cats/train/train'

test_dir = '../input/dogs-vs-cats/test1/test1'
#Create labels for the images

filenames = os.listdir(train_dir)

labels = []

for filename in filenames:

    if filename.startswith('dog'):

        labels.append(1)

    else:

        labels.append(0)
#Import the filenames and labels into a dataframe

#Only 5000 images would be used for a shorter training time

df_train = pd.DataFrame({'Filenames':filenames,'Labels':labels})

#Stratified sampling

df_train = df_train.groupby('Labels').apply(lambda x:x.sample(frac=0.2,random_state=100))

#Read data from the directory

X_filenames = df_train['Filenames'].values

y = df_train['Labels'].values

X = [image.load_img(os.path.join(train_dir,filename),target_size=(150,150)) for filename in X_filenames]

X = np.array([image.img_to_array(item) for item in X])
#Standardization of training data

X = X/255.
#Split the data into train and val set

Xtrain,Xval,ytrain,yval = train_test_split(X,y,stratify=y,test_size=0.2)
#Constrcut the CNN model

CNN1 = Sequential()

CNN1.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

CNN1.add(Conv2D(32,(3,3),activation='relu'))

CNN1.add(MaxPooling2D((2,2)))

CNN1.add(Conv2D(64,(3,3),activation='relu'))

CNN1.add(Conv2D(64,(3,3),activation='relu'))

CNN1.add(MaxPooling2D((2,2)))

CNN1.add(Flatten())

CNN1.add(Dense(512,activation='relu'))

CNN1.add(Dense(1,activation='sigmoid'))
#Define the optimizer and compile the CNN

optimizer = RMSprop(lr=1e-4)

CNN1.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])
history1 = CNN1.fit(Xtrain,ytrain,validation_data=(Xval,yval),batch_size=128,epochs=40,verbose=0)
#Extract loss and acc for visualisation

loss1 = history1.history['loss']

acc1 = history1.history['acc']

val_loss1= history1.history['val_loss']

val_acc1 = history1.history['val_acc']
#Result visualisation

result_visualisation(loss1,acc1,val_loss1,val_acc1)
#Overfitting problem is quite prominent

#Try  to introduce a dropout layer to improve the model performance
#Constrcut the CNN model with Dropout Layer

CNN2 = Sequential()

CNN2.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

CNN2.add(MaxPooling2D((2,2)))

CNN2.add(Conv2D(32,(3,3),activation='relu'))

CNN2.add(MaxPooling2D((2,2)))

CNN2.add(Conv2D(64,(3,3),activation='relu'))

CNN2.add(MaxPooling2D((2,2)))

CNN2.add(Conv2D(128,(3,3),activation='relu'))

CNN2.add(MaxPooling2D((2,2)))

CNN2.add(Flatten())

CNN2.add(Dropout(0.5))

CNN2.add(Dense(512,activation='relu'))

CNN2.add(Dense(1,activation='sigmoid'))
CNN2.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])

history2 = CNN2.fit(Xtrain,ytrain,validation_data=(Xval,yval),batch_size=128,epochs=100,verbose=0)
#Extract loss and acc for visualisation

loss2 = history2.history['loss']

acc2 = history2.history['acc']

val_loss2= history2.history['val_loss']

val_acc2 = history2.history['val_acc']
#Result visualisation

result_visualisation(loss2,acc2,val_loss2,val_acc2)
#After introducing the dropout layer, the validation accuracy has improved

#In the following, we are going to introduce data augmentation and see if this further improve the performance
#Data Augmentation

imggen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,

                            shear_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_gen = imggen.flow(Xtrain,ytrain,batch_size=128)
#Constrcut the CNN model with Dropout Layer

CNN3 = Sequential()

CNN3.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

CNN3.add(MaxPooling2D((2,2)))

CNN3.add(Conv2D(32,(3,3),activation='relu'))

CNN3.add(MaxPooling2D((2,2)))

CNN3.add(Conv2D(64,(3,3),activation='relu'))

CNN3.add(MaxPooling2D((2,2)))

CNN3.add(Conv2D(128,(3,3),activation='relu'))

CNN3.add(MaxPooling2D((2,2)))

CNN3.add(Flatten())

CNN3.add(Dropout(0.5))

CNN3.add(Dense(512,activation='relu'))

CNN3.add(Dense(1,activation='sigmoid'))
CNN3.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['acc'])

history3 = CNN3.fit_generator(train_gen,steps_per_epoch=80,validation_data=(Xval,yval),epochs=100,verbose=2)
#Extract loss and acc for visualisation

loss3 = history3.history['loss']

acc3 = history3.history['acc']

val_loss3= history3.history['val_loss']

val_acc3 = history3.history['val_acc']
#Result visualisation

result_visualisation(loss3,acc3,val_loss3,val_acc3)
#Compare the validation accuracy across all the networks

val_acc_list = [val_acc1,val_acc2,val_acc3]

plot_val_acc_comparison(val_acc_list,smooth=True)
#Compare the validation loss across all the networks

val_loss_list = [val_loss1,val_loss2,val_loss3]

plot_val_loss_comparison(val_loss_list,smooth=True)
#By introducing data augmentation, the performance of the network on the validation set has greatly improved

#If the full training dataset has been used, it is believed that the performance of the network would be even better