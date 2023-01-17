import numpy as np
import pandas as pd
import os
from scipy.misc import imread, imresize
import datetime
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image

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

train_doc = np.random.permutation(open('Project_data/train.csv').readlines())
val_doc = np.random.permutation(open('Project_data/val.csv').readlines())
# batch_size = #experiment with the batch size
directory = "/mnt/disks/user/project/PROJECT/Project_data/train"
childDirectories = next(os.walk(directory))[1]
df_train = pd.DataFrame(childDirectories,columns =['Names'])
df_train = df_train["Names"].str.split("_", n = 6, expand = True)
df_train = df_train.drop(df_train.columns[[0, 1, 2, 3, 4, 5]], axis=1)
df_train[6].value_counts()

directory = "/mnt/disks/user/project/PROJECT/Project_data/val"
childDirectories = next(os.walk(directory))[1]
df_val = pd.DataFrame(childDirectories,columns =['Names'])
df_val = df_val["Names"].str.split("_", n = 6, expand = True)
df_val = df_val.drop(df_val.columns[[0, 1, 2, 3, 4, 5]], axis=1)
df_val[6].value_counts()

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
img = imread('Project_data/train/WIN_20180907_15_35_09_Pro_Right Swipe_new/WIN_20180907_15_35_09_Pro_00012.png')
plt.imshow(img)

plt.subplot(1,2,2)
img1 = imresize(img,(120,120))
plt.imshow(img1)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
img2 = imread('Project_data/train/WIN_20180926_18_07_05_Pro_Right_Swipe_new/WIN_20180926_18_07_05_Pro_00009.png')
plt.imshow(img2)

plt.subplot(1,2,2)
img_2 = imresize(img2,(120,120))
plt.imshow(img_2)
batch_size = 10
x = 30 # # x is the number of images
y = 120 # width of the image
z = 120 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB
def generator(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [x for x in range(0,x)] #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
                    temp = temp/127.5-1 #Normalize data
                    
                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])#normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])#normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])#normalise and feed in the image
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches

        if (len(folder_list) != batch_size*num_batches):
            print("Batch: ",num_batches+1,"Index:", batch_size)
            batch_size = len(folder_list) - (batch_size*num_batches)
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
                    temp = temp/127.5-1 #Normalize data
                    
                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])
                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])
                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])
                   
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels
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
curr_dt_time = datetime.datetime.now()
train_path = 'Project_data/train'
val_path = 'Project_data/val'
num_train_sequences = len(train_doc)
print('# training sequences =', num_train_sequences)
num_val_sequences = len(val_doc)
print('# validation sequences =', num_val_sequences)
num_epochs = 30 # choose the number of epochs
print ('# epochs =', num_epochs)
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

#write your model here

# Model 1
# One input and output layer
# 4 Convolutional and max pooling layers to obtain the most important/informatic features
# 2 Dense layers followed by dropout

model_1 = Sequential()

model_1.add(Conv3D(8,kernel_size=(3,3,3), input_shape=(x,y,z,channel), padding='same'))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))
model_1.add(MaxPooling3D(pool_size=(2,2,2)))


model_1.add(Conv3D(16,kernel_size=(3,3,3), padding='same'))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))
model_1.add(MaxPooling3D(pool_size=(2,2,2)))


model_1.add(Conv3D(32,kernel_size=(3,3,3), padding='same'))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))
model_1.add(MaxPooling3D(pool_size=(2,2,2)))


model_1.add(Conv3D(64,kernel_size=(3,3,3), padding='same'))
model_1.add(BatchNormalization())
model_1.add(Activation('relu'))
model_1.add(MaxPooling3D(pool_size=(2,2,2)))


#Flatten Layers
model_1.add(Flatten())
model_1.add(Dense(1000, activation='relu'))
model_1.add(Dropout(0.5))

model_1.add(Dense(500, activation='relu'))
model_1.add(Dropout(0.5))

#softmax layer
model_1.add(Dense(5, activation='softmax'))
optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print (model_1.summary())
train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)
model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)
        
filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1) # write the REducelronplateau code here
callbacks_list = [checkpoint, LR]
if (num_train_sequences%batch_size) == 0:
    steps_per_epoch = int(num_train_sequences/batch_size)
else:
    steps_per_epoch = (num_train_sequences//batch_size) + 1

if (num_val_sequences%batch_size) == 0:
    validation_steps = int(num_val_sequences/batch_size)
else:
    validation_steps = (num_val_sequences//batch_size) + 1
history1 = model_1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history1)
batch_size = 10
num_epochs = 50

train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_1.summary())
history1_1 = model_1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history1_1)
batch_size = 30
num_epochs = 50

train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history1_2 = model_1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history1_2)
batch_size = 50
num_epochs = 15

train_generator = generator(train_path, train_doc, batch_size)
val_generator = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_1.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history1_3 = model_1.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history1_3)
from keras import regularizers
# Model 2
# One input and output layer
# 4 Convolutional and max pooling layers to obtain the most important/informatic features
# 2 Dense layers followed by dropout
# Dropout after each covolutional layer and 

batch_size = 20
num_epochs = 15
x = 30 # # x is the number of images
y = 120 # width of the image
z = 120 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB

model_2 = Sequential()

model_2.add(Conv3D(8,kernel_size=(3,3,3), input_shape=(x,y,z,channel), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2,2,2)))
model_2.add(Dropout(0.25))

model_2.add(Conv3D(16,kernel_size=(3,3,3), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2,2,2)))
model_2.add(Dropout(0.25))

model_2.add(Conv3D(32,kernel_size=(3,3,3), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2,2,2)))
model_2.add(Dropout(0.25))

model_2.add(Conv3D(64,kernel_size=(3,3,3), padding='same'))
model_2.add(Activation('relu'))
model_2.add(BatchNormalization())
model_2.add(MaxPooling3D(pool_size=(2,2,2)))
model_2.add(Dropout(0.25))

#Flatten Layers
model_2.add(Flatten())
model_2.add(Dense(1000, activation='relu'))
model_2.add(Dropout(0.5))

model_2.add(Dense(500, activation='relu'))
model_2.add(Dropout(0.5))

#softmax layer
model_2.add(Dense(5, activation='softmax'))
train_generator2 = generator(train_path, train_doc, batch_size)
val_generator2 = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_2.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_2.summary())
history2 = model_2.fit_generator(train_generator2, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator2, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history2)
def generator_1(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [x for x in range(0,x)] #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
                    
                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])/255 #normalise and feed in the image
                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])/255 #normalise and feed in the image
                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])/255 #normalise and feed in the image
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels #you yield the batch_data and the batch_labels, remember what does yield do

        
        # write the code for the remaining data points which are left after full batches

        if (len(folder_list) != batch_size*num_batches):
            print("Batch: ",num_batches+1,"Index:", batch_size)
            batch_size = len(folder_list) - (batch_size*num_batches)
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
                    
                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])/255
                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])/255
                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])/255
                   
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels
from keras import regularizers
# Model 3
# One input and output layer
# 4 Convolutional and max pooling layers to obtain the most important/informatic features
# 2 Dense layers followed by dropout
# Dropout after each covolutional layer and 
# Filter size (2x2x2)

batch_size = 30
num_epochs = 30

model_3 = Sequential()

model_3.add(Conv3D(16,kernel_size=(2,2,2), input_shape=(x,y,z,channel), padding='same'))
model_3.add(Activation('relu'))
model_3.add(BatchNormalization())
model_3.add(MaxPooling3D(pool_size=(2,2,2)))

model_3.add(Conv3D(32,kernel_size=(2,2,2), padding='same'))
model_3.add(Activation('relu'))
model_3.add(BatchNormalization())
model_3.add(MaxPooling3D(pool_size=(2,2,2)))


model_3.add(Conv3D(64,kernel_size=(2,2,2), padding='same'))
model_3.add(Activation('relu'))
model_3.add(BatchNormalization())
model_3.add(MaxPooling3D(pool_size=(2,2,2)))


model_3.add(Conv3D(128,kernel_size=(2,2,2), padding='same'))
model_3.add(Activation('relu'))
model_3.add(BatchNormalization())
model_3.add(MaxPooling3D(pool_size=(2,2,2)))


#Flatten Layers
model_3.add(Flatten())
model_3.add(Dense(64, activation='relu'))
model_3.add(Dropout(0.5))

model_3.add(Dense(64, activation='relu'))
model_3.add(Dropout(0.5))

#softmax layer
model_3.add(Dense(5, activation='softmax'))

train_generator3 = generator_1(train_path, train_doc, batch_size)
val_generator3 = generator_1(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0002) #write your optimizer
model_3.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_3.summary())
history3 = model_3.fit_generator(train_generator3, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator3, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history3)
batch_size = 30
num_epochs = 35
x = 30 # # x is the number of images
y = 160 # width of the image
z = 160 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB
def generator_2(source_path, folder_list, batch_size):
    print( 'Source path = ', source_path, '; batch size =', batch_size)
    img_idx = [x for x in range(0,x)] #create a list of image numbers you want to use for a particular video
    while True:
        t = np.random.permutation(folder_list)
        num_batches = len(folder_list)//batch_size # calculate the number of batches
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
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
            batch_data = np.zeros((batch_size,x,y,z,channel)) # x is the number of images you use for each video, (y,z) is the final size of the input images and 3 is the number of channels RGB
            batch_labels = np.zeros((batch_size,classes)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                imgs = os.listdir(source_path+'/'+ t[folder + (batch*batch_size)].split(';')[0]) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = imread(source_path+'/'+ t[folder + (batch*batch_size)].strip().split(';')[0]+'/'+imgs[item]).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    temp = imresize(image,(y,z))
                    temp = temp/127.5-1 #Normalize data
                    
                    batch_data[folder,idx,:,:,0] = (temp[:,:,0])
                    batch_data[folder,idx,:,:,1] = (temp[:,:,1])
                    batch_data[folder,idx,:,:,2] = (temp[:,:,2])
                   
                batch_labels[folder, int(t[folder + (batch*batch_size)].strip().split(';')[2])] = 1
            yield batch_data, batch_labels
# Model 4


model_4 = Sequential()

model_4.add(Conv3D(16,kernel_size=(3,3,3), input_shape=(x,y,z,channel), padding='same'))
model_4.add(Activation('relu'))
model_4.add(BatchNormalization())
model_4.add(MaxPooling3D(pool_size=(2,2,2)))

model_4.add(Conv3D(32,kernel_size=(3,3,3), padding='same'))
model_4.add(Activation('relu'))
model_4.add(BatchNormalization())
model_4.add(MaxPooling3D(pool_size=(2,2,2)))


model_4.add(Conv3D(64,kernel_size=(3,3,3), padding='same'))
model_4.add(Activation('relu'))
model_4.add(BatchNormalization())
model_4.add(MaxPooling3D(pool_size=(2,2,2)))


model_4.add(Conv3D(128,kernel_size=(3,3,3), padding='same'))
model_4.add(Activation('relu'))
model_4.add(BatchNormalization())
model_4.add(MaxPooling3D(pool_size=(2,2,2)))


#Flatten Layers
model_4.add(Flatten())
model_4.add(Dense(64, activation='relu'))
model_4.add(Dropout(0.5))

model_4.add(Dense(64, activation='relu'))
model_4.add(Dropout(0.5))

#softmax layer
model_4.add(Dense(5, activation='softmax'))

train_generator4 = generator_2(train_path, train_doc, batch_size)
val_generator4 = generator_2(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0002) #write your optimizer
model_4.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_4.summary())
history4 = model_4.fit_generator(train_generator4, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator4, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history4)
batch_size = 30
num_epochs = 15
x = 30 # # x is the number of images
y = 120 # width of the image
z = 120 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM

model_5 = Sequential()
model_5.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu'),input_shape=(x,y,z,channel)))
model_5.add(TimeDistributed(BatchNormalization()))
model_5.add(TimeDistributed(MaxPooling2D((2,2))))

model_5.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
model_5.add(TimeDistributed(BatchNormalization()))
model_5.add(TimeDistributed(MaxPooling2D((2, 2))))

model_5.add(TimeDistributed(Conv2D(64, (3, 3),padding='same', activation='relu')))
model_5.add(TimeDistributed(BatchNormalization()))
model_5.add(TimeDistributed(MaxPooling2D((2, 2))))

model_5.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',	activation='relu')))
model_5.add(TimeDistributed(BatchNormalization()))
model_5.add(TimeDistributed(MaxPooling2D((2, 2))))

model_5.add(TimeDistributed(Conv2D(256, (3, 3), padding='same',	activation='relu')))
model_5.add(TimeDistributed(BatchNormalization()))
model_5.add(TimeDistributed(MaxPooling2D((2, 2))))

model_5.add(TimeDistributed(Flatten()))
model_5.add(LSTM(1024))

model_5.add(Dense(512,activation='relu'))
model_5.add(Dropout(0.25))

model_5.add(Dense(5, activation='softmax'))
model_5.add(Dropout(0.25))

train_generator5 = generator(train_path, train_doc, batch_size)
val_generator5 = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0002) #write your optimizer
model_5.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_5.summary())
history5 = model_5.fit_generator(train_generator5, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator5, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history5)
batch_size = 10
num_epochs = 20

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_5.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history5_1 = model_5.fit_generator(train_generator5, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator5, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history5_1)
batch_size = 10
num_epochs = 20
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
history6 = model_6.fit_generator(train_generator6, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator6, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history6)
batch_size = 10
num_epochs = 25
x = 30 # # x is the number of images
y = 120 # width of the image
z = 120 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB


model_61 = Sequential()
model_61.add(TimeDistributed(pretrained_mobilenet,input_shape=(x,y,z,channel)))

model_61.add(TimeDistributed(BatchNormalization()))
model_61.add(TimeDistributed(MaxPooling2D((2, 2))))
model_61.add(TimeDistributed(Flatten()))

model_61.add(GRU(256))
model_61.add(Dropout(0.3))

model_61.add(Dense(256,activation='relu'))
model_61.add(Dropout(0.3))

model_61.add(Dense(5, activation='softmax'))
train_generator61 = generator(train_path, train_doc, batch_size)
val_generator61 = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_61.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_61.summary())
history61 = model_61.fit_generator(train_generator6, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator6, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history61)
batch_size = 10
num_epochs = 20
x = 30 # # x is the number of images
y = 120 # width of the image
z = 120 # height of the image
classes = 5 #5 gestures 
channel = 3 #RGB
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM

model_7 = Sequential()
model_7.add(TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu'),input_shape=(x,y,z,channel)))
model_7.add(TimeDistributed(BatchNormalization()))
model_7.add(TimeDistributed(MaxPooling2D((2, 2))))

model_7.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
model_7.add(TimeDistributed(BatchNormalization()))
model_7.add(TimeDistributed(MaxPooling2D((2, 2))))

model_7.add(TimeDistributed(Conv2D(64, (3, 3),padding='same', activation='relu')))
model_7.add(TimeDistributed(BatchNormalization()))
model_7.add(TimeDistributed(MaxPooling2D((2, 2))))

model_7.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation='relu')))
model_7.add(TimeDistributed(BatchNormalization()))
model_7.add(TimeDistributed(MaxPooling2D((2, 2))))

model_7.add(TimeDistributed(Flatten()))
model_7.add(LSTM(1024))

model_7.add(Dense(512,activation='relu'))
model_7.add(Dropout(0.25))

model_7.add(Dense(5, activation='softmax'))
model_7.add(Dropout(0.25))

train_generator7 = generator(train_path, train_doc, batch_size)
val_generator7 = generator(val_path, val_doc, batch_size)

optimiser = optimizers.Adam(lr=0.0001) #write your optimizer
model_7.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model_7.summary())
history7 = model_7.fit_generator(train_generator5, steps_per_epoch=steps_per_epoch, epochs=num_epochs, verbose=1, 
                    callbacks=callbacks_list, validation_data=val_generator5, 
                    validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)
plot(history7)
