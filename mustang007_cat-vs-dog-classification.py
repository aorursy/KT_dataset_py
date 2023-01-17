# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.applications.inception_v3 import InceptionV3
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_dir = '/kaggle/input/training-data/'

train_dir = os.path.join(base_dir, 'training/training')

validation_dir = os.path.join(base_dir, 'testing/testing')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_dogs_dir
print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
pre_trained_model = InceptionV3(input_shape = (150,150,3),
                                include_top = False,
                               weights = 'imagenet')

# pre_trained_model.load_weights(weight_file)
for layer in pre_trained_model.layers:
      layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
x = layers.Flatten()(last_output)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 300x300 with 3 bytes color
#     # This is the first convolution
#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The third convolution
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     # The fourth convolution
# #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2,2),
# #     # The fifth convolution
# #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
# #     tf.keras.layers.MaxPooling2D(2,2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.summary()
# class mycallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('val_accuracy')>0.98):
#             print('\nReached 75% accuracy so cancelling training!')
#             self.model.stop_training=True
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/training-data/training/training/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/kaggle/input/training-data/testing/testing/',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


callbacks = [earlystop, learning_rate_reduction]
callbacks = mycallback()
history = model.fit(
      train_generator,
      steps_per_epoch=10,  
      epochs=100,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=10,
      callbacks = [callbacks]
)
import matplotlib.pyplot as plt
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc , label='acc')
plt.plot  ( epochs, val_acc , label='val_acc')
plt.title ('Training and validation accuracy')
plt.legend()
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
import numpy as np
File_Name = []
Category = []

from keras.preprocessing import image
# predicting images
path = '/kaggle/input/mh-cat-and-dog/Data/Test/'
for i in sorted(os.listdir(path), key=len):
#     print(i)
    File_Name.append(i)
    
    img_path = os.path.join(path+i)
    img = image.load_img(img_path, target_size=(150, 150))

    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images)

    result = (np.round(classes[0]))
    print(result)
    Category.append(result)

#     if classes[0]>0:
       

#     else:
#         print(" is a cat")

File_Name
# Category.to_excel('result.xlsx')
category = pd.DataFrame(Category)
category.rename( columns={0:'Category'}, inplace=True)
file_name = pd.DataFrame(File_Name)
file_name.rename( columns={0:'File_Name'} , inplace=True)
result = pd.concat([category, file_name], axis=1)
result
sub = pd.read_csv('/kaggle/input/mh-cat-and-dog/Data/Sample_Submission.csv')
sub
# train = train.merge(center,on='center_id')
# train.head(10)

Result = sub.merge(result, on='File_Name')
Result.to_csv('result.csv')