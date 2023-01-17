# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf





# Any results you write to the current directory are saved as output.
print(os.listdir('../input/'))
mountain = '../input/seg_train/seg_train/mountain'

street = '../input/seg_train/seg_train/street'

glacier = '../input/seg_train/seg_train/glacier'

buildings = '../input/seg_train/seg_train/buildings'

sea = '../input/seg_train/seg_train/sea'

forest = '../input/seg_train/seg_train/forest'



mountain_files = os.listdir(mountain)

street_files = os.listdir(street)

glacier_files = os.listdir(glacier)

buildings_files = os.listdir(buildings)

sea_files = os.listdir(sea)

forest_files = os.listdir(forest)

print(os.listdir('../input/seg_train/seg_train'))

print("Mountain :",len(os.listdir(mountain)))

print("Street :",len(os.listdir(street)))

print("Glacier :",len(os.listdir(glacier)))

print("Buildings :",len(os.listdir(buildings)))

print("Sea :",len(os.listdir(sea)))

print("Forest :",len(os.listdir(forest)))

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.pyplot as mpimg



pic_index = 2



next_mountain = [os.path.join(mountain,fname) for fname in mountain_files[pic_index-2:pic_index]]

next_street = [os.path.join(street,fname) for fname in street_files[pic_index-2:pic_index]]

next_glacier = [os.path.join(glacier,fname) for fname in glacier_files[pic_index-2:pic_index]]

next_buildings = [os.path.join(buildings,fname) for fname in buildings_files[pic_index-2:pic_index]]

next_sea = [os.path.join(sea,fname) for fname in sea_files[pic_index-2:pic_index]]

next_forest = [os.path.join(forest,fname) for fname in forest_files[pic_index-2:pic_index]]





for i, img_path in enumerate(next_mountain+next_street+next_glacier+next_buildings+next_sea+next_forest):

  #print(img_path)

  img = mpimg.imread(img_path)

  plt.imshow(img)

  plt.axis('Off')

  plt.show()
import tensorflow as tf

import keras_preprocessing 

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = '../input/seg_train/seg_train'

training_datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range = 40,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    fill_mode = 'nearest'



)



VALIDATION_DIR = '../input/seg_test/seg_test'

validation_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = training_datagen.flow_from_directory(

    TRAINING_DIR,

    target_size = (150,150),

    class_mode = 'categorical'



)



validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    target_size = (150,150),

    class_mode = 'categorical'



)
model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 150x150 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(6, activation='softmax')

])

 

model.summary()

model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=100, validation_data = validation_generator, verbose = 1)



model.save("model.h5")
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs,acc,'r',label='Training Accuracy')

plt.plot(epochs,val_acc,'b',label='Validation accuracy')

plt.title('Training and Validation Accuracy')

plt.legend(loc=0)

plt.figure()



plt.show()
print(os.listdir('../working'))