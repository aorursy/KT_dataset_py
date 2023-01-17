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
import cv2
apple = cv2.imread("../input/fruits-fresh-and-rotten-for-classification/dataset/test/freshapples/Screen Shot 2018-06-08 at 4.59.44 PM.png")
apple = cv2.cvtColor(apple,cv2.COLOR_BGR2RGB)
plt.imshow(apple)
apple.shape
rotap = cv2.imread('../input/fruits-fresh-and-rotten-for-classification/dataset/test/rottenapples/Screen Shot 2018-06-07 at 2.15.34 PM.png')
rotap = cv2.cvtColor(rotap,cv2.COLOR_BGR2RGB)
rotap.shape
plt.imshow(rotap)
from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range= 30,
                              width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              rescale =1/255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              fill_mode ='nearest')
image_gen.flow_from_directory('../input/fruits-fresh-and-rotten-for-classification/dataset/train')
image_gen.flow_from_directory('../input/fruits-fresh-and-rotten-for-classification/dataset/test')
img_shape = (150,150,3)

from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary, 0=cat , 1=dog
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16
img_shape = (150,150,3)
train_image_gen = image_gen.flow_from_directory('../input/fruits-fresh-and-rotten-for-classification/dataset/train',
                                               target_size=  (150,150),
                                                batch_size = 32,
                                                
                                               
                                              
                                               class_mode='categorical')
test_image_gen = image_gen.flow_from_directory('../input/fruits-fresh-and-rotten-for-classification/dataset/test',
                                               target_size=(150,150),
                                               batch_size=batch_size,
                                               class_mode='categorical')
train_image_gen.class_indices

test_image_gen.class_indices

import warnings
warnings.filterwarnings('ignore')
results = model.fit_generator(train_image_gen,epochs=100,
                              steps_per_epoch=150,
                              validation_data=test_image_gen,
                             validation_steps=12)
# summarize history for accuracy
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import numpy as np
from keras.preprocessing import image
import cv2


testfile = cv2.imread('../input/fruits-fresh-and-rotten-for-classification/dataset/train/rottenapples/Screen Shot 2018-06-07 at 2.15.20 PM.png')
testfile = cv2.cvtColor(testfile,cv2.COLOR_BGR2RGB)
plt.imshow(testfile)


testfile_11 = '../input/fruits-fresh-and-rotten-for-classification/dataset/train/rottenapples/Screen Shot 2018-06-07 at 2.15.20 PM.png'

testfile_1 = image.load_img(testfile_1, target_size=(150, 150))

testfile_1 = image.img_to_array(testfile_1)

testfile_1 = np.expand_dims(testfile_1, axis=0)
testfile_1 = testfile_1/255
testfile1 = np.array(testfile_1)
prediction_prob = model.predict(testfile1)

print(prediction_prob)
train_image_gen.class_indices