#@title Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.
from __future__ import absolute_import, division, print_function, unicode_literals



import os

import numpy as np

import glob

import shutil

import matplotlib.pyplot as plt
import tensorflow as tf ##

from tensorflow.keras.models import Sequential ##

from tensorflow.keras.layers import Dense, Flatten ##

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout ##

from tensorflow.keras.preprocessing.image import ImageDataGenerator ##
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"



zip_file = tf.keras.utils.get_file(origin=_URL, 

                                   fname="flower_photos.tgz", 

                                   extract=True)



base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
for cl in classes:

  img_path = os.path.join(base_dir, cl)

  images = glob.glob(img_path + '/*.jpg')

  print("{}: {} Images".format(cl, len(images)))

  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]



  for t in train:

    if not os.path.exists(os.path.join(base_dir, 'train', cl)):

      os.makedirs(os.path.join(base_dir, 'train', cl))

    shutil.move(t, os.path.join(base_dir, 'train', cl))



  for v in val:

    if not os.path.exists(os.path.join(base_dir, 'val', cl)):

      os.makedirs(os.path.join(base_dir, 'val', cl))

    shutil.move(v, os.path.join(base_dir, 'val', cl))
train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')
batch_size = 100 ##

IMG_SHAPE = 150 ##
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)##



train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, 

                                               directory=train_dir, 

                                               shuffle=True, 

                                               target_size=(IMG_SHAPE,IMG_SHAPE))##
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

    plt.tight_layout()

    plt.show()

    

    

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)##



train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                              directory=train_dir,

                                              shuffle=True,

                                              target_size=(IMG_SHAPE,IMG_SHAPE))##
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen = ImageDataGenerator(rescale=1./255,zoom_range=0.5)



train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                              directory=train_dir,

                                              shuffle=True,

                                              target_size=(IMG_SHAPE,IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen_train = ImageDataGenerator(rescale=1./255,

                                    horizontal_flip=True,

                                    rotation_range=45,

                                    zoom_range=0.5,

                                    width_shift_range=0.15, #圖片水平及上下位置平移

                                    height_shift_range=0.15) ##





train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                    directory=train_dir,

                                                    shuffle=True,

                                                    target_size=(IMG_SHAPE,IMG_SHAPE),

                                                    class_mode='sparse') ##
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
image_gen_val = ImageDataGenerator(rescale=1./255) ##

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                    directory=val_dir,

                                                    shuffle=False, #也可以不用加

                                                    target_size=(IMG_SHAPE,IMG_SHAPE),

                                                    class_mode='sparse') ##
model = Sequential()



model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE, 3))) #彩色圖片, padding='same'=0-padding=輸入輸出size一樣, padding='valid'=不採用0-padding, (input, filter, strides, padding)

model.add(MaxPooling2D(pool_size=(2, 2))) #默認stride=pooling_size



model.add(Conv2D(32, 3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, 3, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))



model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax')) ##
model.summary()
# Compile the model

model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy']) #優化器(梯度下降法)+損失函數+評估指標
epochs = 100



history = model.fit_generator(train_data_gen,

                              steps_per_epoch=int(np.ceil(train_data_gen.n/float(batch_size))), #np.ceil=四捨五入取整數, 此在計算一次epoch等於幾次iteration

                              epochs = epochs,

                              validation_data = val_data_gen,

                              validation_steps = int(np.ceil(val_data_gen.n/float(batch_size)))

                             ) ##

acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8,8)) #figsize:指定figure的寬和高

plt.subplot(1,2,1) #row,column,位置

plt.plot(epochs_range,acc,label='Training Accuracy') #plt.plot(x, y), x=epochs數, y=正確率

plt.plot(epochs_range,val_acc,label='Validation Accuracy')

plt.legend(loc='lower right') #圖例figure的位置=右下角

plt.title('Training and Validation Accuracy')



plt.subplot(1,2,2)

plt.plot(epochs_range,loss,label='Training Loss')

plt.plot(epochs_range,val_acc,label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss') ##
