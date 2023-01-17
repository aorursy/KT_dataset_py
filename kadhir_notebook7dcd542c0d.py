# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
base_dir = '../input/cloud-classification/data/train'
data_dir = pathlib.Path(base_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
"""roses = list(data_dir.glob('Ac/*'))
PIL.Image.open(str(roses[0]))
"""
tf.test.gpu_device_name()
batch_size = 32
img_height = 400
img_width = 400
test_path = "../input/cloud-classification/data/test"
test_dir = pathlib.Path(test_path)
train_generator = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
num_classes = 11
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 400
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)


train_generator = data_generator.flow_from_directory(directory='../input/cloud-classification/data/train',
                                                     subset='training')
validation_generator = data_generator.flow_from_directory(directory='../input/cloud-classification/data/train',
                                                   subset='validation')
my_new_model.add(keras.layers.Dropout(0.7))

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

a = my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=40,
        epochs = 5,
        validation_data=validation_generator,
        validation_steps=1, callbacks=[es_callback])
#print(a)
print(a.history.keys())
plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(a.history['accuracy'])  
plt.plot(a.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(a.history['loss'])  
plt.plot(a.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
import glob
file_list = glob.glob("../input/cloud-classification/data/test/*.jpg")
#print(file_list)
predictions = my_new_model.predict('../input/cloud-classification/data/test/1.jpg')
import numpy as np
from PIL import Image
from keras.preprocessing import image
img = image.load_img('../input/cloud-classification/data/test/1.jpg')# , target_size=(32,32))
img  = image.img_to_array(img)
img  = img.reshape((1,) + img.shape)
# img  = img/255

img_class=my_new_model.predict_classes(img) 
# this model above was already trained 
# code from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-#neural-networks-python-keras/
prediction = img_class[0]
classname = img_class[0]
print("Class: ",classname)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

"""img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(400,400,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
batch_size = 16

# this is the augmentation configuration we will use for training
"""data_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)"""

# this is the augmentation configuration we will use for testing:
# only rescaling

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
data_generator = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input, validation_split=0.2)


train_generator = data_generator.flow_from_directory(directory='../input/cloud-classification/data/train',
                                                     subset='training')
validation_generator = data_generator.flow_from_directory(directory='../input/cloud-classification/data/train',
                                                   subset='validation')
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training