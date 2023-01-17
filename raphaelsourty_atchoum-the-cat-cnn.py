import numpy as np
import pandas as pd
import matplotlib
from matplotlib.pyplot import plot

import zipfile # Preprocessing

import cv2 # Preprocessing
import imageio # Preprocessing

import PIL # Preprocessing
import keras

import shutil # Preprocessing
import os # Preprocessing

from tqdm import tqdm # Progress bar
# import keras_tqdm # Progress bar 

import os
# Any results you write to the current directory are saved as output.
from keras import backend 
from keras import applications
from keras.preprocessing import image # Preprocessing
from keras.preprocessing.image import ImageDataGenerator #DataAugmentation
from keras.callbacks import * 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D
list_picture = os.listdir("../input/dogs-vs-cats-redux-kernels-edition/train/")
df = pd.DataFrame({"file" : list_picture})
df['label'] = df['file'].apply(lambda x : 0 if x.split('.')[0] == 'cat' else 1)
df['validation'] = df['label'].apply(lambda x : 1 if np.random.randint(0,11) <= 2 else 0)
df.sample(10)
print('Percentage of validation data : {}'.format(len(df[df['validation']==1])/len(df)*100))
try : 
    shutil.rmtree('data/train/cat/')
    shutil.rmtree('data/train/dog/')
    shutil.rmtree('data/validation/cat/')
    shutil.rmtree('data/validation/dog/')
except : 
    print('No folders to delete')
os.makedirs('data/train/cat/')
os.makedirs('data/train/dog/')
os.makedirs('data/validation/cat/')
os.makedirs('data/validation/dog/')
list_picture_test = os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test/")
try : 
    shutil.rmtree('reshape_test')
except : 
    print('No folder to delete')
os.makedirs('reshape_test')
img = keras.preprocessing.image.load_img('../input/dogs-vs-cats-redux-kernels-edition/train/dog.11931.jpg')
img
np.array(img)[0]
np.array(img).shape
img_preprocessed = np.array(img.convert('L').rotate(45).transpose(PIL.Image.TRANSPOSE))
matplotlib.pyplot.imshow(img_preprocessed, interpolation='nearest')
matplotlib.pyplot.show()
datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
x = keras.preprocessing.image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
try : 
    os.makedirs('example')
except: 
    print('Folder already exist')
#i = 0
for batch in datagen.flow(x, save_to_dir='example', save_prefix='preprocessed', save_format='jpg'):
    #Create 20 pictures  : 
    #i += 1
    #if i > 20:
    break  # otherwise the generator would loop indefinitely
keras.preprocessing.image.load_img('example/{}'.format(os.listdir("example")[0]))
img_width, img_height = 150, 150

if backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
input_shape
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape= input_shape, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # This converts our 3D feature maps to 1D feature vectors 3*3*128 

model.add(Dense(128, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5)) # The Dropout is aggresive but it allow to reduce overfiting.

model.add(Dense(64, use_bias=False)) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1)) # Binary classification
model.add(BatchNormalization())
model.add(Activation('sigmoid'))# Binary classification
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# optimizer = SGD(momentum=0.9, nesterov=True) could be better here 
model.summary()
train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
        'data/train',  # target directory
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
# This class will allow me to visualize results of the training
class LossHistory(Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
history = LossHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                              min_delta=0,
                              patience=15, # Maximum number of epochs without improvment of val_loss, here I disabled early stopping
                              verbose=0, 
                            mode='auto')
model.load_weights("../input/weights-2/model_weights_2.h5")
list_picture_test = [int(file.split('.')[0]) for file in os.listdir('reshape_test')]
list_picture_test.sort()
list_picture_test = ['{}.jpg'.format(file) for file in list_picture_test]
classes = []

img = keras.preprocessing.image.load_img('../input/reshape/144.jpg')
img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255 # Scaling image
model.predict_classes(img)
img = keras.preprocessing.image.load_img('../input/reshape/145.jpg')
img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255 # Scaling image
model.predict_classes(img)
img = keras.preprocessing.image.load_img('../input/reshape/146.jpg')
img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255 # Scaling image
model.predict_classes(img)
