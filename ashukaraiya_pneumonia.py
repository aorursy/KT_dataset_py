import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt



from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential



import os
os.getcwd()
os.listdir('../input/chest_xray/chest_xray')
img_name = 'IM-0119-0001.jpeg'

img_normal = load_img('../input/chest_xray/chest_xray/train/NORMAL/' + img_name)

plt.imshow(img_normal)

plt.show()
img_name = 'person1016_bacteria_2947.jpeg'

img_infected = load_img('../input/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)



plt.imshow(img_infected)

plt.show
num_train_normal = os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')

num_train_pneumonia = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA/')
sns.set_style('whitegrid')

sns.barplot(x = ['NORMAL', 'PNEUMONIA'], y = [len(num_train_normal),len(num_train_pneumonia)])
infected = len(num_train_pneumonia)

normal = len(num_train_normal)

total = infected + normal
print('INFECTED = ', infected)

print('NORMAL = ', normal)

print('TOTAl = ', total)
image_height = 150 

image_width = 150

batch_size = 16

num_epoch = 10

model = Sequential()



model.add(Conv2D(64,(3,3), strides = (1,1),input_shape=(image_height,image_width, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size= (2,2)))



model.add(Conv2D(64,(3,3), strides = (1,1), activation = 'relu'))

model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3), strides = (1,1),activation = 'relu'))

model.add(MaxPooling2D(pool_size= (2,2)))



model.add(Conv2D(64,(3,3), strides = (1,1), activation = 'relu'))

model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 1, activation= 'sigmoid'))





model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





model.summary()
train_data_dir = '../input/chest_xray/chest_xray/train'

validation_data_dir = '../input/chest_xray/chest_xray/val'

test_data_dir = '../input/chest_xray/chest_xray/test'







# Rescale

train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.1)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

val_datagen = ImageDataGenerator(rescale = 1./ 255)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size =(image_width, image_height), batch_size = batch_size,

                                                    class_mode = 'binary')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size =(image_width, image_height), batch_size = batch_size,

                                                    class_mode = 'binary')
val_generator = val_datagen.flow_from_directory(validation_data_dir, target_size =(image_width, image_height), batch_size = batch_size,

                                                    class_mode = 'binary')
model.fit_generator(

train_generator,

steps_per_epoch = total// batch_size,

epochs = num_epoch,

validation_data = val_generator, 

validation_steps = 17// batch_size)