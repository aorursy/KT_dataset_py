import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import os

import matplotlib.pyplot as plt

import cv2

from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator

import seaborn as sns

%matplotlib inline
path_normal_train='../input/chest-xray-pneumonia/chest_xray/train/NORMAL'

j_n=os.listdir(path_normal_train)

path_pneumonia_train='../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'

j_p=os.listdir(path_pneumonia_train)
fig=plt.figure(figsize=(20,20))

o=1

for i in range(2):

    fig.add_subplot(4,2,o)

    plt.title(str(i+1)+' normal img')

    j=cv2.imread(os.path.join(path_normal_train,j_n[i]))

    plt.imshow(j)

    o+=1

    fig.add_subplot(4,2,o)

    plt.title(str(i+1)+' pneumonia img')

    j=cv2.imread(os.path.join(path_pneumonia_train,j_p[i]))

    plt.imshow(j)

    o+=1

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x=[0,1], y= [len(j_n),len(j_p)])

plt.title('no. of train cases')

plt.xlabel('0:normal images, 1:pneumonia images')

plt.ylabel('no. of images')
gen=ImageDataGenerator(rotation_range=20,width_shift_range=0.1,

                       height_shift_range=0.1,shear_range=0.15,

                       zoom_range=0.1,channel_shift_range=10,horizontal_flip=True,vertical_flip=True,

                       fill_mode='constant',cval=125)
#below function plots the  

def plots(figsize,images,a,b):

    fig=plt.figure(figsize=figsize)

    for i in range(1,len(images)+1):

        fig.add_subplot(a,b,i)

        plt.imshow(images[i-1])

    plt.show()

               

        

def create_images(path,j_p,i,pri):

    img_path=os.path.join(path,j_p[i])

    image=np.expand_dims(cv2.imread(img_path),0)

    aug_itter=gen.flow(image)#,save_to_dir=path,save_prefix='aug',save_format='jpeg')

    aug_images=[next(aug_itter)[0].astype(np.uint8) for i in range(10)]

    if pri=='print':

        plots((10,20),aug_images,5,2)





#takes-> path of directory, 

#        images_name_list, 

#        which image(i'th image),

#       'print' if want to print images else empty with ""    

create_images(path_pneumonia_train,j_p,3,'print')
train_='../input/chest-xray-pneumonia/chest_xray/train'

test_='../input/chest-xray-pneumonia/chest_xray/test'

val_='../input/chest-xray-pneumonia/chest_xray/val'



#loads train_data

train_gen=ImageDataGenerator(rescale=1./255)

train=train_gen.flow_from_directory(train_,

                                   target_size=(250,250),

                                   batch_size=10,

                                   class_mode='binary')





#loads augmented load_data

augmentation=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.1,

                       height_shift_range=0.1,shear_range=0.15,

                       zoom_range=0.1,channel_shift_range=10,horizontal_flip=True,vertical_flip=True,

                       fill_mode='constant',cval=125)

train_augmentation=augmentation.flow_from_directory(train_,

                                target_size=(250,250),

                                batch_size=10,

                                class_mode='binary')





#loads test data

test_gen=ImageDataGenerator(rescale=1./255)

test=test_gen.flow_from_directory(test_,

                                  target_size=(250,250),

                                  batch_size=10,

                                  class_mode='binary')





#loads validaiton data

val_gen=ImageDataGenerator(rescale=1./255)

val=val_gen.flow_from_directory(test_,

                                target_size=(250,250),

                                batch_size=10,

                                class_mode='binary')


model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(250,250,3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),

    tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32,activation='sigmoid'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.summary()
from keras.optimizers import Adam

opt=Adam(lr=0.0001)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc'])



#this function plots the graph between train and validation data parameters

def plot_metrics(history,epochs):

    epoch=range(1,epochs+1)

    loss=history.history['loss']

    accuracy=history.history['acc']

    vlos=history.history['val_loss']

    vacc=history.history['val_acc']

    plt.title('Loss graph')

    plt.xlabel('epochs')

    plt.ylabel('loss')

    plt.plot(epoch,loss)

    plt.plot(epoch,vlos)

    plt.show()

    plt.title('Accuracy graph')

    plt.xlabel('epochs')

    plt.ylabel('accuracy')

    plt.plot(epoch,accuracy)

    plt.plot(epoch,vacc)

    plt.show()
epochs=30

history=model.fit_generator(train_augmentation,

                            steps_per_epoch=20,

                            epochs=epochs,

                            validation_data=val,

                            validation_steps=8,

                            verbose=1)
plot_metrics(history,epochs)
history=model.fit_generator(train,

                            steps_per_epoch=20,

                            epochs=epochs,

                            validation_data=val,

                            validation_steps=8,

                            verbose=1)
plot_metrics(history,epochs)
epochs=15

history=model.fit_generator(train_augmentation,

                            steps_per_epoch=20,

                            epochs=epochs,

                            validation_data=val,

                            validation_steps=8,

                            verbose=1)
plot_metrics(history,epochs)
loss, acc = model.evaluate_generator(test, steps=3, verbose=1)
tf.keras.utils.plot_model(model, show_shapes=True)