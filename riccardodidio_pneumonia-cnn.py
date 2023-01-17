# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        a=1

     #   print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
try:

  # %tensorflow_version only exists in Colab.

  %tensorflow_version 2.x

  print(tf.__version__)

except Exception:

  pass
from pathlib import Path

import os

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import tensorflow as tf



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (Input, Activation, Dense, Flatten, Conv2D, 

                                     MaxPooling2D, Dropout, BatchNormalization)
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')
train_dir = data_dir / 'train'



val_dir = data_dir / 'val'



test_dir = data_dir / 'test'
Normal_dir = train_dir / 'NORMAL'

pneumonia_dir = train_dir / 'PNEUMONIA'



Normal_val_dir = val_dir / 'NORMAL'

pneumonia_val_dir = val_dir / 'PNEUMONIA'
Normal_dir_ = os.path.join(Normal_dir)

pneumonia_dir_ = os.path.join(pneumonia_dir)



Normal_img = os.listdir(Normal_dir_)



pneumonia_img = os.listdir(pneumonia_dir_)



Normal_val_dir_ = os.path.join(Normal_val_dir)

pneumonia_val_dir_ = os.path.join(pneumonia_val_dir)



Normal_val_img = os.listdir(Normal_val_dir_)



pneumonia_val_img = os.listdir(pneumonia_val_dir_)
print('normal: ', len(os.listdir(Normal_dir_)))

print('pneumonia: ', len(os.listdir(pneumonia_dir_)))
img_path = os.path.join(pneumonia_dir, pneumonia_img[1]) 

img = mpimg.imread(img_path)

plt.imshow(img)
img_path = os.path.join(Normal_dir, Normal_img[1]) 

img = mpimg.imread(img_path)

plt.imshow(img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1/255, 

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   horizontal_flip=True,

                                  )



val_datagen = ImageDataGenerator(rescale=1/255, 

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   horizontal_flip=True,

                                  )



test_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), 

                                                    batch_size=128,  class_mode='binary')



validation_generator = train_datagen.flow_from_directory(val_dir, target_size=(300, 300), 

                                                    batch_size=128,  class_mode='binary')

augmented_images = [train_generator[0][0][0] for i in range(5)]

plotImages(augmented_images)
class pneumonia (tf.keras.Model):

    

    def __init__(self):

       

        super(pneumonia, self).__init__()

        

        self.conv_1        = Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=(300, 300, 3)) 

        self.max_pool_1    = MaxPooling2D(pool_size=(2, 2))

        self.conv_2        = Conv2D(16, kernel_size=(3, 3), activation='relu') 

        self.max_pool_2    = MaxPooling2D(pool_size=(2, 2))

        self.Dropout_0     = Dropout(0.3)

        

        self.conv_3        = Conv2D(32, kernel_size=(3, 3), activation='relu') 

        self.max_pool_3    = MaxPooling2D(pool_size=(2, 2))   

        self.Dropout_1     = Dropout(0.3)

        

        self.conv_4        = Conv2D(64, kernel_size=(3, 3), activation='relu') 

        self.max_pool_4    = MaxPooling2D(pool_size=(2, 2)) 

        self.Dropout_2     = Dropout(0.3)

        

        self.conv_5        = Conv2D(64, kernel_size=(3, 3), activation='relu') 

        self.max_pool_5    = MaxPooling2D(pool_size=(2, 2))      

        self.Dropout_3     = Dropout(0.2)

        

        self.conv_6        = Conv2D(64, kernel_size=(3, 3), activation='relu') 

        self.max_pool_6    = MaxPooling2D(pool_size=(2, 2))      

        self.Dropout_4     = Dropout(0.2)

        

        self.conv_7        = Conv2D(128, kernel_size=(3, 3), activation='relu') 

        self.max_pool_7    = MaxPooling2D(pool_size=(2, 2))      

        self.Dropout_5     = Dropout(0.2)

        

        self.conv_8        = Conv2D(128, kernel_size=(3, 3), activation='relu') 

        self.max_pool_8    = MaxPooling2D(pool_size=(2, 2))      

        self.Dropout_6     = Dropout(0.2)

        

        self.conv_9        = Conv2D(128, kernel_size=(3, 3), activation='relu') 

        self.max_pool_9    = MaxPooling2D(pool_size=(2, 2))      

        self.Dropout_7     = Dropout(0.2)

  

        self.flatten       = tf.keras.layers.Flatten()

 

        self.dense_1       = tf.keras.layers.Dense(units=128, activation='relu')

        self.dense_2       = tf.keras.layers.Dense(units=64, activation='relu')

        self.dense_3       = tf.keras.layers.Dense(units=1, activation='sigmoid')

        

    def call(self, inputs, training=True):



        x = self.max_pool_1(self.conv_1(inputs))

        x = self.max_pool_2(self.conv_2(x))

        x = self.Dropout_0(x)

        

        x = self.max_pool_3(self.conv_3(x))

        

        x = self.Dropout_1(x) 

        

        x = self.max_pool_4(self.conv_4(x))

        x = self.Dropout_3(x)

        

        x = self.max_pool_5(self.conv_5(x))

        x = self.Dropout_3(x) 

        

        x = self.max_pool_6(self.conv_6(x))

        x = self.Dropout_4(x)

        

      #  x = self.max_pool_7(self.conv_7(x))

       # x = self.Dropout_5(x)

        

      #  x = self.max_pool_8(self.conv_8(x))

       # x = self.Dropout_6(x)

        

       # x = self.max_pool_9(self.conv_9(x))

        #x = self.Dropout_7(x)

        

        x = self.flatten(x)

        x = self.dense_1(x)     

        x = self.dense_2(x)

        x = self.dense_3(x)

        

        return x
model = pneumonia()
loss_object = tf.keras.losses.BinaryCrossentropy()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
model.compile(optimizer=optimizer,

              loss=loss_object,

              metrics=['acc'])
num_normal_tr = len(os.listdir(Normal_dir))

num_pneumonia_tr = len(os.listdir(pneumonia_dir))



num_normal_val = len(os.listdir(Normal_val_dir))

num_pneumonia_val = len(os.listdir(pneumonia_val_dir))



total_train = num_normal_tr + num_pneumonia_tr

total_val = num_normal_val + num_pneumonia_tr



print(total_train)

print(total_val)
history = model.fit_generator(train_generator,

                              validation_data=validation_generator,

                          ##    steps_per_epoch=total_train // 128,

                              epochs=10,

                          ##    validation_steps=total_val // 128,

                              verbose=1)
acc      = history.history[     'acc' ]

val_acc  = history.history[ 'val_acc' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]
epochs   = range(len(acc))
plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()
plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )
test_Normal_dir = test_dir / 'NORMAL'

test_pneumonia_dir = test_dir / 'PNEUMONIA'



test_Normal_dir_ = os.path.join(test_Normal_dir)

tes_pneumonia_dir_ = os.path.join(test_pneumonia_dir)



test_Normal_img = os.listdir(test_Normal_dir_)



test_pneumonia_img = os.listdir(tes_pneumonia_dir_)
from keras.preprocessing import image
img_path = os.path.join(test_Normal_dir, test_Normal_img[1]) 

img = mpimg.imread(img_path)

plt.imshow(img)