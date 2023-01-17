# Import the needed Libraries

import numpy as np
import os
from scipy.misc import imread, imresize
import datetime
import os

# Supress all the warnings

import warnings
warnings.filterwarnings('ignore')
# Import the random seet and keras, tensorflow

np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(30)
## Let us take the Input in to Train and Val doc and Iniatise the Bath size as 10 first and then we train the model

train_doc = np.random.permutation(open('./Project_data/train.csv').readlines())
val_doc = np.random.permutation(open('./Project_data/val.csv').readlines())
batch_size =  40 #experiment with the batch size

## We have changed the right path where the files are stored above.
# Let we do the generators and input the images as we see that our images have two different sizes. 
x = 30 # No. of frames images
y = 120 # Width of the image
z = 120 # height

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
                    # Let us resize all the images 
                    Temp_img = imresize(image,(y,z))
                    Temp_img = Temp_img/127.5-1 
                    
                    batch_data[folder,idx,:,:,0] = (Temp_img[:,:,0])#normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = (Temp_img[:,:,1])#normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = (Temp_img[:,:,2])#normalise and feed in the image
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches.
        
# Let us see that if the folder is not equal to the batch size * num of batches
        
        if (len(folder_list) != batch_size*num_batches):
            print("Batches: ",num_batches+1,"Index:", batch_size)
            batch_size = len(folder_list) - (batch_size*num_batches)
            batch_data = np.zeros((batch_size,x,y,z,3)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    Temp_img = imresize(image,(y,z))
                    Temp_img = Temp_img/127.5-1 #Normalize data
                    
                    batch_data[folder,idx,:,:,0] = (Temp_img[:,:,0])
                    batch_data[folder,idx,:,:,1] = (Temp_img[:,:,1])
                    batch_data[folder,idx,:,:,2] = (Temp_img[:,:,2])
                   
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels

curr_dt_time = datetime.datetime.now()
train_path = './Project_data/train'
val_path = './Project_data/val'
num_train_sequences = len(train_doc)
print('# Training_Sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# Validation_Sequences =', num_val_sequences)
num_epochs = 15 # choose the number of epochs
print ('# Epochs = ', num_epochs)
# Let us import all the needed libraries of Keras.

from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation,Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

#write your model here
# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_1 = Sequential()       
model_1.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(30, 120, 120, 3),padding='same'))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))

model_1.add(Conv3D(16, (3, 3, 3), padding='same'))
model_1.add(Activation('relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_1.add(Conv3D(32, (2, 2, 2), padding='same'))
model_1.add(Activation('relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_1.add(Conv3D(64, (2, 2, 2), padding='same'))
model_1.add(Activation('relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_1.add(Conv3D(128, (2, 2, 2), padding='same'))
model_1.add(Activation('relu'))
model_1.add(BatchNormalization())
model_1.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_1.add(Flatten())

model_1.add(Dense(1000, activation='relu'))
model_1.add(Dropout(0.5))

model_1.add(Dense(500, activation='relu'))
model_1.add(Dropout(0.5))

#Softmax layer

model_1.add(Dense(5, activation='softmax'))
        
        
# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_1.summary())
# Let us train and validate the model 

train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)
# Let us see the Validate the Losses and put back the checkpoint

model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1) # write the REducelronplateau code here
callbacks_list = [checkpoint, LR]
# Let us see that the steps_per_epoch and validation steps are used by fit_generator to decide the no. of next()

if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1
# Let us fit the model

model_1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us Experiement the Different Size of images first and try check the model.

# As We know we have five classes as Volume up, Volume down, Right swipe, Left swipe, Stop
channels = 3 # RGB as 3D Convlution
clases = 5
x = 30
y = 120
z = 120 

def generator_1(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [x for x in range(0,x)] #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,channels)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,clases)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    # Let us resize all the images 
                    Temp_img = imresize(image,(y,z)) 
                    Temp_img = Temp_img.mean(axis=-1,keepdims=1)  # Let us convert to the Grey scale
                    Temp_img = Temp_img/127.5-1 
                    
                    batch_data[folder,idx] = Temp_img #normalise and feed in the image
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches.
        
# Let us see that if the folder is not equal to the batch size * num of batches
        
        if (len(folder_list) != batch_size*num_batches):
            print("Batches: ",num_batches+1,"Index:", batch_size)
            batch_size = len(folder_list) - (batch_size*num_batches)
            batch_data = np.zeros((batch_size,x,y,z,channels)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,clases)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    Temp_img = imresize(image,(y,z))
                    Temp_img = Temp_img.mean(axis=-1,keepdims=1)  # Let us convert to the Grey scale 
                    Temp_img = Temp_img/127.5-1 #Normalize data
                    
                    batch_data[folder,idx] = Temp_img
                   
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_2 = Sequential()       
model_2.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(30, 120, 120, 3),padding='same'))
model_2.add(BatchNormalization())
model_2.add(Activation('relu'))

model_2.add(Conv3D(16, (3, 3, 3), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_2.add(Conv3D(32, (2, 2, 2), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_2.add(Conv3D(64, (2, 2, 2), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_2.add(Conv3D(128, (2, 2, 2), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_2.add(Flatten())

model_2.add(Dense(1000, activation='relu'))
model_2.add(Dropout(0.5))

model_2.add(Dense(500, activation='relu'))
model_2.add(Dropout(0.5))

#Softmax layer

model_2.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_2.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_2.summary())
# Let us train and validate the model 

train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_2.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 60 # image width
z = 60 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex1 = Sequential()       
model_ex1.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex1.add(BatchNormalization())
model_ex1.add(Activation('relu'))

model_ex1.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex1.add(Activation('relu'))
model_ex1.add(BatchNormalization())
model_ex1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex1.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex1.add(Activation('relu'))
model_ex1.add(BatchNormalization())
model_ex1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex1.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex1.add(Activation('relu'))
model_ex1.add(BatchNormalization())
model_ex1.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex1.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex1.add(Activation('relu'))
model_ex1.add(BatchNormalization())
model_ex1.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex1.add(Flatten())

model_ex1.add(Dense(1000, activation='relu'))
model_ex1.add(Dropout(0.5))

model_ex1.add(Dense(500, activation='relu'))
model_ex1.add(Dropout(0.5))

#Softmax layer

model_ex1.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex1.summary())
# Let us train and validate the model 

train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 160 # image width
z = 160 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex2 = Sequential()       
model_ex2.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex2.add(BatchNormalization())
model_ex2.add(Activation('relu'))

model_ex2.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex2.add(Activation('relu'))
model_ex2.add(BatchNormalization())
model_ex2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex2.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex2.add(Activation('relu'))
model_ex2.add(BatchNormalization())
model_ex2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex2.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex2.add(Activation('relu'))
model_ex2.add(BatchNormalization())
model_ex2.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex2.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex2.add(Activation('relu'))
model_ex2.add(BatchNormalization())
model_ex2.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex2.add(Flatten())

model_ex2.add(Dense(1000, activation='relu'))
model_ex2.add(Dropout(0.5))

model_ex2.add(Dense(500, activation='relu'))
model_ex2.add(Dropout(0.5))

#Softmax layer

model_ex2.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex2.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex2.summary())
# Let us train and validate the model 

train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 140 # image width
z = 140 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex3 = Sequential()       
model_ex3.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex3.add(BatchNormalization())
model_ex3.add(Activation('relu'))

model_ex3.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex3.add(Activation('relu'))
model_ex3.add(BatchNormalization())
model_ex3.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex3.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex3.add(Activation('relu'))
model_ex3.add(BatchNormalization())
model_ex3.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex3.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex3.add(Activation('relu'))
model_ex3.add(BatchNormalization())
model_ex3.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex3.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex3.add(Activation('relu'))
model_ex3.add(BatchNormalization())
model_ex3.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex3.add(Flatten())

model_ex3.add(Dense(1000, activation='relu'))
model_ex3.add(Dropout(0.5))

model_ex3.add(Dense(500, activation='relu'))
model_ex3.add(Dropout(0.5))

#Softmax layer

model_ex3.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex3.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex2.summary())
# Let us train and validate the model 

train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex4 = Sequential()       
model_ex4.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex4.add(BatchNormalization())
model_ex4.add(Activation('relu'))

model_ex4.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex4.add(Activation('relu'))
model_ex4.add(BatchNormalization())
model_ex4.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex4.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex4.add(Activation('relu'))
model_ex4.add(BatchNormalization())
model_ex4.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex4.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex4.add(Activation('relu'))
model_ex4.add(BatchNormalization())
model_ex4.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex4.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex4.add(Activation('relu'))
model_ex4.add(BatchNormalization())
model_ex4.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex4.add(Flatten())

model_ex4.add(Dense(1000, activation='relu'))
model_ex4.add(Dropout(0.5))

model_ex4.add(Dense(500, activation='relu'))
model_ex4.add(Dropout(0.5))

#Softmax layer

model_ex4.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex4.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex4.summary())
# Let us train and validate the model 
batch_size = 20
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex4.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us train and validate the model 
batch_size = 60
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us train and validate the model 
batch_size = 80
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 20 # number of frames
y = 160 # image width
z = 160 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex7 = Sequential()       
model_ex7.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex7.add(BatchNormalization())
model_ex7.add(Activation('relu'))

model_ex7.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex7.add(Activation('relu'))
model_ex7.add(BatchNormalization())
model_ex7.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex7.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex7.add(Activation('relu'))
model_ex7.add(BatchNormalization())
model_ex7.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex7.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex7.add(Activation('relu'))
model_ex7.add(BatchNormalization())
model_ex7.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex7.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex7.add(Activation('relu'))
model_ex7.add(BatchNormalization())
model_ex7.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex7.add(Flatten())

model_ex7.add(Dense(1000, activation='relu'))
model_ex7.add(Dropout(0.5))

model_ex7.add(Dense(500, activation='relu'))
model_ex7.add(Dropout(0.5))

#Softmax layer

model_ex7.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex7.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex7.summary())
# Let us train and validate the model 
batch_size = 20
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex7.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 20 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex8 = Sequential()       
model_ex8.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex8.add(BatchNormalization())
model_ex8.add(Activation('relu'))

model_ex8.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex8.add(Activation('relu'))
model_ex8.add(BatchNormalization())
model_ex8.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex8.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex8.add(Activation('relu'))
model_ex8.add(BatchNormalization())
model_ex8.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex8.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex8.add(Activation('relu'))
model_ex8.add(BatchNormalization())
model_ex8.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex8.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex8.add(Activation('relu'))
model_ex8.add(BatchNormalization())
model_ex8.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex8.add(Flatten())

model_ex8.add(Dense(1000, activation='relu'))
model_ex8.add(Dropout(0.5))

model_ex8.add(Dense(500, activation='relu'))
model_ex8.add(Dropout(0.5))

#Softmax layer

model_ex8.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex8.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex8.summary())
# Let us train and validate the model 
batch_size = 20
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex8.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 20 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex9 = Sequential()       
model_ex9.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex9.add(BatchNormalization())
model_ex9.add(Activation('relu'))

model_ex9.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex9.add(Activation('relu'))
model_ex9.add(BatchNormalization())
model_ex9.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex9.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex9.add(Activation('relu'))
model_ex9.add(BatchNormalization())
model_ex9.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex9.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex9.add(Activation('relu'))
model_ex9.add(BatchNormalization())
model_ex9.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex9.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex9.add(Activation('relu'))
model_ex9.add(BatchNormalization())
model_ex9.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex9.add(Flatten())

model_ex9.add(Dense(1000, activation='relu'))
model_ex9.add(Dropout(0.5))

model_ex9.add(Dense(500, activation='relu'))
model_ex9.add(Dropout(0.5))

#Softmax layer

model_ex9.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex9.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex9.summary())
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex9.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex10 = Sequential()       
model_ex10.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex10.add(BatchNormalization())
model_ex10.add(Activation('relu'))

model_ex10.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex10.add(Activation('relu'))
model_ex10.add(BatchNormalization())
model_ex10.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex10.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex10.add(Activation('relu'))
model_ex10.add(BatchNormalization())
model_ex10.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex10.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex10.add(Activation('relu'))
model_ex10.add(BatchNormalization())
model_ex10.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex10.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex10.add(Activation('relu'))
model_ex10.add(BatchNormalization())
model_ex10.add(MaxPooling3D(pool_size=(2, 2, 2)))      

# Flatten layer 

model_ex10.add(Flatten())

model_ex10.add(Dense(1000, activation='relu'))
model_ex10.add(Dropout(0.5))

model_ex10.add(Dense(500, activation='relu'))
model_ex10.add(Dropout(0.5))

#Softmax layer

model_ex10.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex10.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex10.summary())
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex10.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex11 = Sequential()       
model_ex11.add(Conv3D(8,kernel_size=(2,2,2),input_shape=(x,y,z,3),padding='same'))
model_ex11.add(BatchNormalization())
model_ex11.add(Activation('relu'))

model_ex11.add(Conv3D(16, (2, 2, 2), padding='same'))
model_ex11.add(Activation('relu'))
model_ex11.add(BatchNormalization())
model_ex11.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex11.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex11.add(Activation('relu'))
model_ex11.add(BatchNormalization())
model_ex11.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex11.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex11.add(Activation('relu'))
model_ex11.add(BatchNormalization())
model_ex11.add(MaxPooling3D(pool_size=(2, 2, 2)))

# Flatten layer 

model_ex11.add(Flatten())

model_ex11.add(Dense(256, activation='relu'))
model_ex11.add(Dropout(0.5))

model_ex11.add(Dense(128, activation='relu'))
model_ex11.add(Dropout(0.5))

#Softmax layer

model_ex11.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex11.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex11.summary())
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex11.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNN network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex12 = Sequential()       
model_ex12.add(Conv3D(8,kernel_size=(3,3,3),input_shape=(x,y,z,3),padding='same'))
model_ex12.add(BatchNormalization())
model_ex12.add(Activation('relu'))

model_ex12.add(Conv3D(16, (3, 3, 3), padding='same'))
model_ex12.add(Activation('relu'))
model_ex12.add(BatchNormalization())
model_ex12.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex12.add(Conv3D(32, (2, 2, 2), padding='same'))
model_ex12.add(Activation('relu'))
model_ex12.add(BatchNormalization())
model_ex12.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex12.add(Conv3D(64, (2, 2, 2), padding='same'))
model_ex12.add(Activation('relu'))
model_ex12.add(BatchNormalization())
model_ex12.add(MaxPooling3D(pool_size=(2, 2, 2)))

model_ex12.add(Conv3D(128, (2, 2, 2), padding='same'))
model_ex12.add(Activation('relu'))
model_ex12.add(BatchNormalization())
model_ex12.add(MaxPooling3D(pool_size=(2, 2, 2))) 

# Flatten layer 

model_ex12.add(Flatten())

model_ex12.add(Dense(256, activation='relu'))
model_ex12.add(Dropout(0.5))

model_ex12.add(Dense(128, activation='relu'))
model_ex12.add(Dropout(0.5))

#Softmax layer

model_ex12.add(Dense(5, activation='softmax'))

# Let us use the Adam optimiser 

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex12.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex12.summary())
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex12.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us import all the needed libraries of Keras.

from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation,Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers


# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex13 = Sequential()   
model_ex13.add(TimeDistributed(Conv2D(16, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex13.add(TimeDistributed(BatchNormalization()))
model_ex13.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex13.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex13.add(TimeDistributed(BatchNormalization()))
model_ex13.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex13.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex13.add(TimeDistributed(BatchNormalization()))
model_ex13.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex13.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex13.add(TimeDistributed(BatchNormalization()))
model_ex13.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex13.add(TimeDistributed(Conv2D(256, (3, 3) , padding='same', activation='relu')))
model_ex13.add(TimeDistributed(BatchNormalization()))
model_ex13.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex13.add(TimeDistributed(Flatten()))

model_ex13.add(LSTM(64))
model_ex13.add(Dropout(0.25))

# Dense layer 
model_ex13.add(Dense(64,activation='relu'))
model_ex13.add(Dropout(0.25))
# Softmax layer

model_ex13.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex13.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex13.summary())
        
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex13.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment the same model and change the image size and try again
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 160 # image width
z = 160 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex14 = Sequential()   
model_ex14.add(TimeDistributed(Conv2D(16, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex14.add(TimeDistributed(BatchNormalization()))
model_ex14.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex14.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex14.add(TimeDistributed(BatchNormalization()))
model_ex14.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex14.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex14.add(TimeDistributed(BatchNormalization()))
model_ex14.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex14.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex14.add(TimeDistributed(BatchNormalization()))
model_ex14.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex14.add(TimeDistributed(Conv2D(256, (3, 3) , padding='same', activation='relu')))
model_ex14.add(TimeDistributed(BatchNormalization()))
model_ex14.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex14.add(TimeDistributed(Flatten()))

model_ex14.add(LSTM(64))
model_ex14.add(Dropout(0.25))

# Dense layer 
model_ex14.add(Dense(64,activation='relu'))
model_ex14.add(Dropout(0.25))
# Softmax layer

model_ex14.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex14.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex14.summary())
        
# Let us fit the model

model_ex14.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# Let us train and validate the model 
batch_size = 20
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex14.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

# Let us train and validate the model 
batch_size = 60
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex14.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment the same model and change the image size and try again
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex17 = Sequential()   
model_ex17.add(TimeDistributed(Conv2D(16, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex17.add(TimeDistributed(BatchNormalization()))
model_ex17.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex17.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex17.add(TimeDistributed(BatchNormalization()))
model_ex17.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex17.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex17.add(TimeDistributed(BatchNormalization()))
model_ex17.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex17.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex17.add(TimeDistributed(BatchNormalization()))
model_ex17.add(TimeDistributed(MaxPooling2D((2, 2))))
        


# Flatten layer 

model_ex17.add(TimeDistributed(Flatten()))

model_ex17.add(LSTM(64))
model_ex17.add(Dropout(0.25))

# Dense layer 
model_ex17.add(Dense(64,activation='relu'))
model_ex17.add(Dropout(0.25))
# Softmax layer

model_ex17.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex17.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex17.summary())
        

# Let us train and validate the model 
batch_size = 20
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex17.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=25, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us import all the needed libraries of Keras.

from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation,Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers


# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex18 = Sequential()   
model_ex18.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex18.add(TimeDistributed(BatchNormalization()))
model_ex18.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex18.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex18.add(TimeDistributed(BatchNormalization()))
model_ex18.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex18.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex18.add(TimeDistributed(BatchNormalization()))
model_ex18.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex18.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex18.add(TimeDistributed(BatchNormalization()))
model_ex18.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex18.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex18.add(TimeDistributed(BatchNormalization()))
model_ex18.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex18.add(TimeDistributed(Flatten()))

model_ex18.add(LSTM(64))
model_ex18.add(Dropout(0.25))

# Dense layer 
model_ex18.add(Dense(64,activation='relu'))
model_ex18.add(Dropout(0.25))
# Softmax layer

model_ex18.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex18.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex18.summary())
        

# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
# Let us fit the model

model_ex18.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=25, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex19 = Sequential()   
model_ex19.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex19.add(TimeDistributed(BatchNormalization()))
model_ex19.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex19.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex19.add(TimeDistributed(BatchNormalization()))
model_ex19.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex19.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex19.add(TimeDistributed(BatchNormalization()))
model_ex19.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex19.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex19.add(TimeDistributed(BatchNormalization()))
model_ex19.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex19.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex19.add(TimeDistributed(BatchNormalization()))
model_ex19.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex19.add(TimeDistributed(Flatten()))

model_ex19.add(LSTM(64))
model_ex19.add(Dropout(0.25))

# Dense layer 
model_ex19.add(Dense(64,activation='relu'))
model_ex19.add(Dropout(0.25))
# Softmax layer

model_ex19.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex19.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex19.summary())
        

# Let us train and validate the model 
batch_size = 30
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex19.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=25, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex20 = Sequential()   
model_ex20.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex20.add(TimeDistributed(BatchNormalization()))
model_ex20.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex20.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex20.add(TimeDistributed(BatchNormalization()))
model_ex20.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex20.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex20.add(TimeDistributed(BatchNormalization()))
model_ex20.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex20.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex20.add(TimeDistributed(BatchNormalization()))
model_ex20.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex20.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex20.add(TimeDistributed(BatchNormalization()))
model_ex20.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex20.add(TimeDistributed(Flatten()))

model_ex20.add(LSTM(64))
model_ex20.add(Dropout(0.25))

# Dense layer 
model_ex20.add(Dense(64,activation='relu'))
model_ex20.add(Dropout(0.25))
# Softmax layer

model_ex20.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex20.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex20.summary())
        

# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex20.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=25, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
from keras.models import load_model

model_ex20.save_weights('model_weights.h5')

# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex21 = Sequential()   
model_ex21.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex21.add(TimeDistributed(BatchNormalization()))
model_ex21.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex21.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex21.add(TimeDistributed(BatchNormalization()))
model_ex21.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex21.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex21.add(TimeDistributed(BatchNormalization()))
model_ex21.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex21.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex21.add(TimeDistributed(BatchNormalization()))
model_ex21.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex21.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex21.add(TimeDistributed(BatchNormalization()))
model_ex21.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex21.add(TimeDistributed(Flatten()))

model_ex21.add(LSTM(64))
model_ex21.add(Dropout(0.25))

# Dense layer 
model_ex21.add(Dense(64,activation='relu'))
model_ex21.add(Dropout(0.25))
# Softmax layer

model_ex21.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.SGD(lr=0.001) #write your optimizer
model_ex21.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex21.summary())
        

# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex21.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex23 = Sequential()   
model_ex23.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex23.add(TimeDistributed(BatchNormalization()))
model_ex23.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex23.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex23.add(TimeDistributed(BatchNormalization()))
model_ex23.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex23.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex23.add(TimeDistributed(BatchNormalization()))
model_ex23.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex23.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex23.add(TimeDistributed(BatchNormalization()))
model_ex23.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex23.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex23.add(TimeDistributed(BatchNormalization()))
model_ex23.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex23.add(TimeDistributed(Flatten()))

model_ex23.add(LSTM(64))
model_ex23.add(Dropout(0.25))

# Dense layer 
model_ex23.add(Dense(64,activation='relu'))
model_ex23.add(Dropout(0.25))
# Softmax layer

model_ex23.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex23.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex23.summary())
        

# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex23.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex24 = Sequential()   
model_ex24.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex24.add(TimeDistributed(BatchNormalization()))
model_ex24.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex24.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex24.add(TimeDistributed(BatchNormalization()))
model_ex24.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex24.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex24.add(TimeDistributed(BatchNormalization()))
model_ex24.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex24.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex24.add(TimeDistributed(BatchNormalization()))
model_ex24.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex24.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex24.add(TimeDistributed(BatchNormalization()))
model_ex24.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex24.add(TimeDistributed(Flatten()))

model_ex24.add(GRU(64))
model_ex24.add(Dropout(0.25))

# Dense layer 
model_ex24.add(Dense(64,activation='relu'))
model_ex24.add(Dropout(0.25))
# Softmax layer

model_ex24.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex24.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex24.summary())
        

# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex24.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex25 = Sequential()   

model_ex25.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex25.add(TimeDistributed(BatchNormalization()))
model_ex25.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex25.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex25.add(TimeDistributed(BatchNormalization()))
model_ex25.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex25.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex25.add(TimeDistributed(BatchNormalization()))
model_ex25.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex25.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex25.add(TimeDistributed(BatchNormalization()))
model_ex25.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex25.add(TimeDistributed(Flatten()))

model_ex25.add(GRU(64))
model_ex25.add(Dropout(0.25))

# Dense layer 
model_ex25.add(Dense(64,activation='relu'))
model_ex25.add(Dropout(0.25))
# Softmax layer

model_ex25.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex25.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex25.summary())
        
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex25.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 18 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex26 = Sequential()  
model_ex26.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex26.add(TimeDistributed(BatchNormalization()))
model_ex26.add(TimeDistributed(MaxPooling2D((2, 2))))

model_ex26.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex26.add(TimeDistributed(BatchNormalization()))
model_ex26.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex26.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex26.add(TimeDistributed(BatchNormalization()))
model_ex26.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex26.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex26.add(TimeDistributed(BatchNormalization()))
model_ex26.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex26.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex26.add(TimeDistributed(BatchNormalization()))
model_ex26.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex26.add(TimeDistributed(Flatten()))

model_ex26.add(GRU(64))
model_ex26.add(Dropout(0.25))

# Dense layer 
model_ex26.add(Dense(64,activation='relu'))
model_ex26.add(Dropout(0.25))
# Softmax layer

model_ex26.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex26.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex26.summary())
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex26.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us import the needed libraries

from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation,Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.applications import mobilenet

mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

x = 30
y = 120
z = 120


model_trs = Sequential()
model_trs.add(TimeDistributed(mobilenet_transfer,input_shape=(x,y,z,3)))
               
model_trs.add(TimeDistributed(BatchNormalization()))
model_trs.add(TimeDistributed(MaxPooling2D((2, 2))))
model_trs.add(TimeDistributed(Flatten()))

# LTSM

model_trs.add(LSTM(64))
model_trs.add(Dropout(0.25))
# Dense layer

model_trs.add(Dense(64,activation='relu'))
model_trs.add(Dropout(0.25))
        
model_trs.add(Dense(5,activation='softmax'))

# Optimisers and compiler
        
optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_trs.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_trs.summary())


# Let us train and validate the model 
batch_size = 10
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_trs.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

x = 30
y = 120
z = 120


model_trs2 = Sequential()
model_trs2.add(TimeDistributed(mobilenet_transfer,input_shape=(x,y,z,3)))
               
model_trs2.add(TimeDistributed(BatchNormalization()))
model_trs2.add(TimeDistributed(MaxPooling2D((2, 2))))
model_trs2.add(TimeDistributed(Flatten()))

# GRU

model_trs2.add(GRU(128))
model_trs2.add(Dropout(0.25))
# Dense layer

model_trs2.add(Dense(128,activation='relu'))
model_trs2.add(Dropout(0.25))
        
model_trs2.add(Dense(5,activation='softmax'))

# Optimisers and compiler
        
optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_trs2.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_trs2.summary())
# Let us train and validate the model 
batch_size = 10
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_trs2.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex29 = Sequential()   
model_ex29.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex29.add(TimeDistributed(BatchNormalization()))
model_ex29.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex29.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex29.add(TimeDistributed(BatchNormalization()))
model_ex29.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex29.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex29.add(TimeDistributed(BatchNormalization()))
model_ex29.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex29.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex29.add(TimeDistributed(BatchNormalization()))
model_ex29.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex29.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex29.add(TimeDistributed(BatchNormalization()))
model_ex29.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex29.add(TimeDistributed(Flatten()))

model_ex29.add(GRU(64))
model_ex29.add(Dropout(0.25))

# Dense layer 
model_ex29.add(Dense(64,activation='relu'))
model_ex29.add(Dropout(0.25))
# Softmax layer

model_ex29.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.adam(lr=0.001) #write your optimizer
model_ex29.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex29.summary())
        
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex29.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 16 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex30 = Sequential()   
model_ex30.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex30.add(TimeDistributed(BatchNormalization()))
model_ex30.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex30.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex30.add(TimeDistributed(BatchNormalization()))
model_ex30.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex30.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex30.add(TimeDistributed(BatchNormalization()))
model_ex30.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex30.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex30.add(TimeDistributed(BatchNormalization()))
model_ex30.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex30.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex30.add(TimeDistributed(BatchNormalization()))
model_ex30.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex30.add(TimeDistributed(Flatten()))

model_ex30.add(GRU(128))
model_ex30.add(Dropout(0.25))

# Dense layer 
model_ex30.add(Dense(128,activation='relu'))
model_ex30.add(Dropout(0.25))
# Softmax layer

model_ex30.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex30.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex30.summary())
        
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex30.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
from keras.models import load_model

model_ex30.save('model_weights.h5')
# Let us experiment different x,y,z value in the CNNLSTM network and find tune all the image size & Hyperparameters later

x = 30 # number of frames
y = 120 # image width
z = 120 # image height

# Input all the images sequencial by building the layer with dropouts and batchnormalisation

model_ex31 = Sequential()   
model_ex31.add(TimeDistributed(Conv2D(8, (3, 3),padding='same', activation='relu'),input_shape=(x,y,z,3)))
model_ex31.add(TimeDistributed(BatchNormalization()))
model_ex31.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex31.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu')))
model_ex31.add(TimeDistributed(BatchNormalization()))
model_ex31.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex31.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
model_ex31.add(TimeDistributed(BatchNormalization()))
model_ex31.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex31.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
model_ex31.add(TimeDistributed(BatchNormalization()))
model_ex31.add(TimeDistributed(MaxPooling2D((2, 2))))
        
model_ex31.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
model_ex31.add(TimeDistributed(BatchNormalization()))
model_ex31.add(TimeDistributed(MaxPooling2D((2, 2))))

# Flatten layer 

model_ex31.add(TimeDistributed(Flatten()))

model_ex31.add(GRU(128))
model_ex31.add(Dropout(0.25))

# Dense layer 
model_ex31.add(Dense(128,activation='relu'))
model_ex31.add(Dropout(0.25))
# Softmax layer

model_ex31.add(Dense(5, activation='softmax'))

# Adam optimiser

optimiser = optimizers.Adam(lr=0.001) #write your optimizer
model_ex31.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_ex31.summary())
        
# Let us train and validate the model 
batch_size = 40
train_generator = generator_1(train_path, train_doc, batch_size)
val_generator = generator_1(val_path, val_doc, batch_size)
model_ex31.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=30, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Dropout
import datetime
import os

project_folder = './'
import abc ###import abstract base class to modify the class further

class ModelBuilderMoreAugmentation(metaclass= abc.ABCMeta):
    
    def initialize_path(self,project_folder):
        self.train_doc = np.random.permutation(open(project_folder + '/' + 'train.csv').readlines())
        self.val_doc = np.random.permutation(open(project_folder + '/' + 'val.csv').readlines())
        self.train_path = project_folder + '/' + 'train'
        self.val_path =  project_folder + '/' + 'val'
        self.num_train_sequences = len(self.train_doc)
        self.num_val_sequences = len(self.val_doc)
        
    def initialize_image_properties(self,image_height=100,image_width=100):
        self.image_height=image_height
        self.image_width=image_width
        self.channels=3 #RGB
        self.num_classes=5 # Gestures
        self.total_frames=30 # number of frames
    
    #tuning hyperparmeters
    def initialize_hyperparams(self,frames_to_sample=30,batch_size=20,num_epochs=20):
        self.frames_to_sample=frames_to_sample
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        
    #  defning generator +  
    def generator(self,source_path, folder_list, augment=False):
        img_idx = np.round(np.linspace(0,self.total_frames-1,self.frames_to_sample)).astype(int)
        batch_size=self.batch_size
        while True:
            t = np.random.permutation(folder_list)
            num_batches = len(t)//batch_size
        
            for batch in range(num_batches): 
                batch_data, batch_labels= self.one_batch_data(source_path,t,batch,batch_size,img_idx,augment)
                yield batch_data, batch_labels 

            remaining_seq=len(t)%batch_size
        
            if (remaining_seq != 0):
                batch_data, batch_labels= self.one_batch_data(source_path,t,num_batches,batch_size,img_idx,augment,remaining_seq)
                yield batch_data, batch_labels 
    
    
    def one_batch_data(self,source_path,t,batch,batch_size,img_idx,augment,remaining_seq=0):
    
        seq_len = remaining_seq if remaining_seq else batch_size
    
        batch_data = np.zeros((seq_len,len(img_idx),self.image_height,self.image_width,self.channels)) 
        batch_labels = np.zeros((seq_len,self.num_classes)) 
    
        if (augment): batch_data_aug = np.zeros((seq_len,len(img_idx),self.image_height,self.image_width,self.channels))

        
        for folder in range(seq_len): 
            imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) 
            for idx,item in enumerate(img_idx): 
                image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                image_resized=imresize(image,(self.image_height,self.image_width,3))
            

                batch_data[folder,idx,:,:,0] = (image_resized[:,:,0])/255
                batch_data[folder,idx,:,:,1] = (image_resized[:,:,1])/255
                batch_data[folder,idx,:,:,2] = (image_resized[:,:,2])/255
            ##Generate batches of tensor image data with real-time data augmentation. Thereby shifting image by 0.5pixel
                if (augment):
                    shifted = cv2.warpAffine(image, 
                                             np.float32([[1, 0, np.random.randint(-30,30)],[0, 1, np.random.randint(-30,30)]]), 
                                            (image.shape[1], image.shape[0]))
                    ##convert an image from one color space to another
                    gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)

                    x0, y0 = np.argwhere(gray > 0).min(axis=0)
                    x1, y1 = np.argwhere(gray > 0).max(axis=0) 
                    
                    ##Cropping and resizing the image to fit the scale,since the image has different resolution depending upon webcam
                    cropped=shifted[x0:x1,y0:y1,:]
                    
                    image_resized=imresize(cropped,(self.image_height,self.image_width,3))
                    
                    ##Handling image rotation
                    M = cv2.getRotationMatrix2D((self.image_width//2,self.image_height//2),
                                                np.random.randint(-10,10), 1.0)
                    rotated = cv2.warpAffine(image_resized, M, (self.image_width, self.image_height))
                    batch_data_aug[folder,idx,:,:,0] = (rotated[:,:,0])/255
                    batch_data_aug[folder,idx,:,:,1] = (rotated[:,:,1])/255
                    batch_data_aug[folder,idx,:,:,2] = (rotated[:,:,2])/255
                
            
            batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            
    
        if (augment):
            batch_data=np.concatenate([batch_data,batch_data_aug])
            batch_labels=np.concatenate([batch_labels,batch_labels])

        
        return(batch_data,batch_labels)
    
    ## to be executed at end
    def train_model(self, model, augment_data=False):
        train_generator = self.generator(self.train_path, self.train_doc,augment=augment_data)
        val_generator = self.generator(self.val_path, self.val_doc)

        model_name = 'model_init' + '_' + str(datetime.datetime.now()).replace(' ','').replace(':','_') + '/'
    
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        
        filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', period=1)
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4)
        callbacks_list = [checkpoint, LR]

        if (self.num_train_sequences%self.batch_size) == 0:
            steps_per_epoch = int(self.num_train_sequences/self.batch_size)
        else:
            steps_per_epoch = (self.num_train_sequences//self.batch_size) + 1

        if (self.num_val_sequences%self.batch_size) == 0:
            validation_steps = int(self.num_val_sequences/self.batch_size)
        else:
            validation_steps = (self.num_val_sequences//self.batch_size) + 1
    ##generator exe
        history=model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=self.num_epochs, verbose=1, 
                            callbacks=callbacks_list, validation_data=val_generator, 
                            validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
        return history

        
    @abc.abstractmethod
    def define_model(self):
        pass
#CNN LSTM with GRU
class RNNCNN2(ModelBuilderMoreAugmentation):
    
    def define_model(self,lstm_cells=64,dense_neurons=64,dropout=0.25):

        model = Sequential()

        model.add(TimeDistributed(Conv2D(16, (3, 3) , padding='same', activation='relu'),
                                  input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        
        model.add(TimeDistributed(Conv2D(32, (3, 3) , padding='same', activation='relu')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        
        model.add(TimeDistributed(Conv2D(64, (3, 3) , padding='same', activation='relu')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        
        model.add(TimeDistributed(Conv2D(128, (3, 3) , padding='same', activation='relu')))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        
        ## GRU,LSTM
        model.add(TimeDistributed(Flatten()))

        model.add(GRU(lstm_cells))
        model.add(Dropout(dropout))
        
        ##Finally regularizing the model
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(Dropout(dropout))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        optimiser = optimizers.Adam(lr=0.0002) ##as observed SGD didnt perform well
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

rnn_cnn2=RNNCNN2()
rnn_cnn2.initialize_path(project_folder)
rnn_cnn2.initialize_image_properties(image_height=120,image_width=120)
rnn_cnn2.initialize_hyperparams(frames_to_sample=18,batch_size=20,num_epochs=20)
rnn_cnn2_model=rnn_cnn2.define_model(lstm_cells=128,dense_neurons=128,dropout=0.25)
rnn_cnn2_model.summary()
from scipy.misc import imread, imresize
import cv2
import matplotlib.pyplot as plt
% matplotlib inline
print("Total Params:", rnn_cnn2_model.count_params())
model_ex32=rnn_cnn2.train_model(rnn_cnn2_model,augment_data=True)
#Transfer Learning using imagenet
from keras.applications import mobilenet
#Same configuration but using prev TL,inbuild Keras model MobileNet
##MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

class RNNCNN_TL(ModelBuilderMoreAugmentation):
    
    def define_model(self,lstm_cells=64,dense_neurons=64,dropout=0.25):
        
        model = Sequential()
        model.add(TimeDistributed(mobilenet_transfer,input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
        
        
        for layer in model.layers:
            layer.trainable = False
        
        
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(lstm_cells))
        model.add(Dropout(dropout))
        
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(Dropout(dropout))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        
        optimiser = optimizers.Adam()
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
rnn_cnn_tl=RNNCNN_TL()
rnn_cnn_tl.initialize_path(project_folder)
rnn_cnn_tl.initialize_image_properties(image_height=120,image_width=120)
rnn_cnn_tl.initialize_hyperparams(frames_to_sample=16,batch_size=5,num_epochs=20)
rnn_cnn_tl_model=rnn_cnn_tl.define_model(lstm_cells=128,dense_neurons=128,dropout=0.25)
rnn_cnn_tl_model.summary()
print("Total Params:", rnn_cnn_tl_model.count_params())
model_ex33=rnn_cnn_tl.train_model(rnn_cnn_tl_model,augment_data=True)
##using image net with TL from mobilenet which has accuracy of more than .70   
from keras.applications import mobilenet

mobilenet_transfer = mobilenet.MobileNet(weights='imagenet', include_top=False)

class RNNCNN_TL2(ModelBuilderMoreAugmentation):
    
    def define_model(self,gru_cells=64,dense_neurons=64,dropout=0.25):
        
        model = Sequential()
        model.add(TimeDistributed(mobilenet_transfer,input_shape=(self.frames_to_sample,self.image_height,self.image_width,self.channels)))
 
        ##Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))

        model.add(GRU(gru_cells))
        model.add(Dropout(dropout))
        
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(Dropout(dropout))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        
        optimiser = optimizers.Adam()
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
rnn_cnn_tl2=RNNCNN_TL2()
rnn_cnn_tl2.initialize_path(project_folder)
rnn_cnn_tl2.initialize_image_properties(image_height=120,image_width=120)
rnn_cnn_tl2.initialize_hyperparams(frames_to_sample=16,batch_size=5,num_epochs=20)
rnn_cnn_tl2_model=rnn_cnn_tl2.define_model(gru_cells=128,dense_neurons=128,dropout=0.25)
rnn_cnn_tl2_model.summary()
print("Total Params:", rnn_cnn_tl2_model.count_params())
model_ex34=rnn_cnn_tl2.train_model(rnn_cnn_tl2_model,augment_data=True)
