import numpy as np

import os

from scipy.misc import imread, imresize

import datetime

import os

from zipfile import ZipFile

import matplotlib.pyplot as plt

%matplotlib inline



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
np.random.seed(30)

import random as rn

rn.seed(30)

from keras import backend as K

import tensorflow as tf

tf.set_random_seed(30)
#Load Data



train_doc = np.random.permutation(open('train.csv').readlines())

val_doc = np.random.permutation(open('val.csv').readlines())

batch_size = 50 #experiment with the batch size
x = 30 # # x is the number of images

y = 120 # width of the image

z = 120 # height of the image
def generator(source_path, folder_list, batch_size):

    print( 'Source path = ', source_path, '; batch size =', batch_size)

    img_idx = [x for x in range(0,x)] #create a list of image numbers you want to use for a particular video

    while True:

        t = np.random.permutation(folder_list)

        num_batches = len(folder_list)//batch_size # calculate the number of batches

        for batch in range(num_batches): # we iterate over the number of batches

            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB

            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output

            for folder in range(batch_size): # iterate over the batch_size

                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder

                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in

                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    

                    #crop the images and resize them. Note that the images are of 2 different shape 

                    #and the conv3D will throw error if the inputs in a batch have different shapes

                    temp = imresize(image,(120,120))

                    temp = temp/127.5-1 #Normalize data

                    

                    batch_data[folder,idx,:,:,0] = (temp[:,:,0]) #normalise and feed in the image

                    batch_data[folder,idx,:,:,1] = (temp[:,:,1]) #normalise and feed in the image

                    batch_data[folder,idx,:,:,2] = (temp[:,:,2]) #normalise and feed in the image

                    

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do



        

        # write the code for the remaining data points which are left after full batches



        if (len(folder_list) != batch_size*num_batches):

            print("Batch: ",num_batches+1,"Index:", batch_size)

            batch_size = len(folder_list) - (batch_size*num_batches)

            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB

            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output

            for folder in range(batch_size): # iterate over the batch_size

                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder

                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in

                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)

                    

                    #crop the images and resize them. Note that the images are of 2 different shape 

                    #and the conv3D will throw error if the inputs in a batch have different shapes

                    temp = imresize(image,(120,120))

                    temp = temp/127.5-1 #Normalize data

                    

                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])

                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])

                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])

                   

                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1

            yield batch_data, batch_labels
curr_dt_time = datetime.datetime.now()

train_path = 'train'

val_path = 'val'

num_train_sequences = len(train_doc)

print('# training sequences =', num_train_sequences)

num_val_sequences = len(val_doc)

print('# validation sequences =', num_val_sequences)

num_epochs = 50 # choose the number of epochs

print ('# epochs =', num_epochs)
import keras

from keras.models import Sequential, Model

from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation

from keras.layers.convolutional import Conv3D, MaxPooling3D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers

from keras.layers import Dropout



#write your model here



model_al1 = Sequential()



model_al1.add(Conv3D(8, #number of filters 

                 kernel_size=(3,3,3), 

                 input_shape=(30, 120, 120, 3),

                 padding='same'))

model_al1.add(BatchNormalization())

model_al1.add(Activation('relu'))



model_al1.add(MaxPooling3D(pool_size=(2,2,2)))



model_al1.add(Conv3D(16, #Number of filters, 

                 kernel_size=(3,3,3), 

                 padding='same'))

model_al1.add(BatchNormalization())

model_al1.add(Activation('relu'))



model_al1.add(MaxPooling3D(pool_size=(2,2,2)))



model_al1.add(Conv3D(32, #Number of filters 

                 kernel_size=(3,3,3), 

                 padding='same'))

model_al1.add(BatchNormalization())

model_al1.add(Activation('relu'))



model_al1.add(MaxPooling3D(pool_size=(2,2,2)))



model_al1.add(Conv3D(64, #Number pf filters 

                 kernel_size=(3,3,3), 

                 padding='same'))

model_al1.add(BatchNormalization())

model_al1.add(Activation('relu'))



model_al1.add(MaxPooling3D(pool_size=(2,2,2)))



#Flatten Layers

model_al1.add(Flatten())



model_al1.add(Dense(1000, activation='relu'))

model_al1.add(Dropout(0.5))



model_al1.add(Dense(500, activation='relu'))

model_al1.add(Dropout(0.5))



#softmax layer

model_al1.add(Dense(5, activation='softmax'))



optimiser = optimizers.Adam(lr=0.001) #write your optimizer

model_al1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print (model_al1.summary())
train_generator = generator(train_path, train_doc, batch_size)

val_generator = generator(val_path, val_doc, batch_size)
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'

    

if not os.path.exists(model_name):

    os.mkdir(model_name)

        

filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)  # write the REducelronplateau code here

callbacks_list = [checkpoint, LR]
if (num_train_sequences%batch_size) == 0:

    steps_per_epoch = int(num_train_sequences/batch_size)

else:

    steps_per_epoch = (num_train_sequences//batch_size) + 1



if (num_val_sequences%batch_size) == 0:

    validation_steps = int(num_val_sequences/batch_size)

else:

    validation_steps = (num_val_sequences//batch_size) + 1
history = model_al1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 

                    callbacks=callbacks_list, validation_data=val_generator, 

                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
def plot(history):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

    axes[0].plot(history.history['loss'])   

    axes[0].plot(history.history['val_loss'])

    axes[0].legend(['loss','val_loss'])

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    





    axes[1].plot(history.history['categorical_accuracy'])   

    axes[1].plot(history.history['val_categorical_accuracy'])

    axes[1].legend(['categorical_accuracy','val_categorical_accuracy'])

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    

    plt.show()
plot(history)
batch_size = 10

num_epochs = 25

x = 30 # # x is the number of images

y = 120 # width of the image

z = 120 # height of the image

classes = 5 #5 gestures 

channel = 3 #RGB
from keras.applications import mobilenet

pretrained_mobilenet = mobilenet.MobileNet(weights='imagenet', include_top=False)

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.recurrent import LSTM
model_6 = Sequential()

model_6.add(TimeDistributed(pretrained_mobilenet,input_shape=(x,y,z,channel)))



model_6.add(TimeDistributed(BatchNormalization()))

model_6.add(TimeDistributed(MaxPooling2D((2, 2))))

model_6.add(TimeDistributed(Flatten()))



model_6.add(GRU(128))

model_6.add(Dropout(0.25))



model_6.add(Dense(128,activation='relu'))

model_6.add(Dropout(0.25))



model_6.add(Dense(5, activation='softmax'))
train_generator6 = generator(train_path, train_doc, batch_size)

val_generator6 = generator(val_path, val_doc, batch_size)



optimiser = optimizers.Adam(lr=0.0001) #write your optimizer

model_6.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print(model_6.summary())
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'

    

if not os.path.exists(model_name):

    os.mkdir(model_name)

        

filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'



checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)



LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1)  # write the REducelronplateau code here

callbacks_list = [checkpoint, LR]
if (num_train_sequences%batch_size) == 0:

    steps_per_epoch = int(num_train_sequences/batch_size)

else:

    steps_per_epoch = (num_train_sequences//batch_size) + 1



if (num_val_sequences%batch_size) == 0:

    validation_steps = int(num_val_sequences/batch_size)

else:

    validation_steps = (num_val_sequences//batch_size) + 1
history6 = model_6.fit_generator(train_generator6, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 

                    callbacks=callbacks_list, validation_data=val_generator6, 

                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history6)