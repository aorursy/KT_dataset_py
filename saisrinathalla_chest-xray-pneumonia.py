import pandas as pd

import numpy as np

import tensorflow as tf

import os

import seaborn as sns

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Flatten,Conv2D, MaxPool2D , Dropout , BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop,Adam

from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input
training_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/train/*/*')

validation_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/val/*/*')

total_files = training_images

total_files.extend(validation_images)

train_images, val_images = train_test_split(total_files, test_size = 0.2)

print(len(train_images))

print(len(val_images))
train_images[5]
count_normal = len([x for x in train_images if "NORMAL" in x])

print(f'Normal images count in training set: {count_normal}')



count_pneumonia = len([x for x in train_images if "PNEUMONIA" in x])

print(f'Pneumonia images count in training set: {count_pneumonia}')



count_array = []

count_array += ['positive']*count_pneumonia

count_array += ['negative']*count_normal



sns.set_style('ticks')

sns.countplot(count_array)
tf.io.gfile.makedirs('/kaggle/working/val_dataset/negative/')

tf.io.gfile.makedirs('/kaggle/working/val_dataset/positive/')

tf.io.gfile.makedirs('/kaggle/working/train_dataset/negative/')

tf.io.gfile.makedirs('/kaggle/working/train_dataset/positive/')

for ele in train_images:

    parts_of_path = ele.split('/')



    if 'PNEUMONIA' == parts_of_path[-2]:

        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/train_dataset/positive/' +  parts_of_path[-1])

    else:

        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/train_dataset/negative/' +  parts_of_path[-1])

for ele in val_images:

    parts_of_path = ele.split('/')



    if 'PNEUMONIA' == parts_of_path[-2]:

        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/val_dataset/positive/' +  parts_of_path[-1])

    else:

        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/val_dataset/negative/' +  parts_of_path[-1])

train_datagen = ImageDataGenerator(rescale = 1/255,

                                 rotation_range = 30,

                                 zoom_range = 0.2,

                                 width_shift_range = 0.1,

                                 height_shift_range = 0.1)

val_datagen = ImageDataGenerator(rescale = 1/255)

                                



train_generator = train_datagen.flow_from_directory(

    '/kaggle/working/train_dataset/',

    target_size = [150,150],

    batch_size = 128 ,

    class_mode = 'binary'

)



validation_generator = val_datagen.flow_from_directory(

    '/kaggle/working/val_dataset/',

    target_size = [150,150],

    batch_size = 128 ,

    class_mode = 'binary'

)
eval_datagen = ImageDataGenerator(rescale = 1/255)



test_generator = eval_datagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/test',

    target_size =[150,150],

    batch_size = 128, 

    class_mode = 'binary'

)
initial_bias = np.log([count_pneumonia/count_normal])

initial_bias

weight_for_0 = (1 / count_normal)*(len(train_images))/2.0 

weight_for_1 = (1 / count_pneumonia)*(len(train_images))/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
model=Sequential([Conv2D(32,

                         (3,3),

                         activation='relu',

                         input_shape=[150,150,3]),

                      

                    MaxPool2D((2,2), strides=2, padding='same'),

                      

                    Conv2D(filters=256,

                                kernel_size=(3,3),

                                activation='relu'),

                      

                    MaxPool2D((2,2), strides=2, padding='same'),



                      Conv2D(filters=512,

                                kernel_size=(3,3),

                                activation='relu'),

                      

                    MaxPool2D((2,2),strides=2,padding='same'),

                      

                    Conv2D(filters=128,

                                kernel_size=(3,3),

                                activation='relu'),

                      

                        Flatten(),

                    

                          Dense(units=256,activation='relu'),

                          Dropout(0.2),

                          Dense(1,activation='sigmoid')])        

          

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])

model.summary()
history1 = model.fit(

    train_generator,

    epochs =15,

     validation_data= validation_generator,

    class_weight = class_weight)
figure, axis = plt.subplots(1, 2, figsize=(18,5))

axis = axis.ravel()



for i,element in enumerate(['accuracy', 'loss']):

    axis[i].plot(history1.history[element])

    axis[i].plot(history1.history['val_' + element])

    axis[i].set_title('Model {}'.format(element))

    axis[i].set_xlabel('epochs')

    axis[i].set_ylabel(element)

    axis[i].legend(['train', 'val'])
eval_result1 = model.evaluate_generator(test_generator)

print(eval_result1)

print('loss rate on evaluation data :', eval_result1[0])

print('accuracy rate on evaluation data :', eval_result1[1])