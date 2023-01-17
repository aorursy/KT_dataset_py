## plot the image from class C as example to plot the images 

from matplotlib import pyplot as plt
import os
import random


_, _, sign_images = next(os.walk('../input/handsignimages/Train/C/'))

### prepare a 4x4 plot (total of 16 images)
fig, ax = plt.subplots(3, 2, figsize=(10,10))

### randomly select and plot an image
for idx, img in enumerate(random.sample(sign_images, 6)):
    img_read = plt.imread('../input/handsignimages/Train/C/'+img)
    ax[int(idx/2), idx%2].imshow(img_read)
    ax[int(idx/2), idx%2].axis('off')
    ax[int(idx/2), idx%2].set_title('Train/C'+img)
plt.show()

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
 #       print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from keras.preprocessing.image import ImageDataGenerator

training_data_generator = ImageDataGenerator(rescale =1/255, validation_split=0.1)

training_set = training_data_generator.flow_from_directory('../input/handsignimages/Train', target_size =(28, 28), batch_size = 16 , class_mode ='categorical',  subset='training')

validation_set = training_data_generator.flow_from_directory('../input/handsignimages/Train', target_size =(28, 28), batch_size = 16 , class_mode ='categorical',  subset='validation')

print(training_set)
print(training_set)

from random import uniform
from keras.layers.normalization import BatchNormalization


input_size = 28
filter_size = 3
num_filter = 8
maxpool_size = 2
batch_size = 16
epochs = 10
steps_per_epoch = 24720/batch_size

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense


model = Sequential()
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))

model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=1))


model.add(Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))


model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))


model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(24,activation='softmax'))

 



METRICS = [ 'accuracy']#, 'precision','recall']


model.compile( optimizer= keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=METRICS)

#
model.summary()
history = model.fit_generator(
training_set,
steps_per_epoch= steps_per_epoch,
epochs= epochs,
validation_data=validation_set,
validation_steps= 2735  // batch_size

)

testing_data_generator = ImageDataGenerator(rescale =1/255)

testing_set = training_data_generator.flow_from_directory('../input/handsignimages/Test', 
                                                          target_size =(28, 28), 
                                                          batch_size = 16 , 
                                                          class_mode ='categorical')

score = model.evaluate_generator(testing_set, steps= len(testing_set))
for idx, metric in enumerate(model.metrics_names):
    print(metric, score[idx])
from random import uniform
from keras.layers.normalization import BatchNormalization


input_size = 28
filter_size = 3
num_filter = 8
maxpool_size = 2
batch_size = 16
epochs = 10
steps_per_epoch = 24720/batch_size

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense


model = Sequential()
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=1))
model.add(Dropout(uniform(0, 1)))

model.add(Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))
model.add(Dropout(uniform(0, 1)))  

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(24,activation='softmax'))

 



METRICS = [ 'accuracy']#, 'precision','recall']


model.compile( optimizer= keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=METRICS)

#
model.summary()

history = model.fit_generator(
training_set,
steps_per_epoch= steps_per_epoch,
epochs= epochs,
validation_data=validation_set,
validation_steps= 2735  // batch_size

)

testing_data_generator = ImageDataGenerator(rescale =1/255)

testing_set = training_data_generator.flow_from_directory('../input/handsignimages/Test', 
                                                          target_size =(28, 28), 
                                                          batch_size = 16 , 
                                                          class_mode ='categorical')

score = model.evaluate_generator(testing_set, steps= len(testing_set))
for idx, metric in enumerate(model.metrics_names):
    print(metric, score[idx])
from random import uniform
from keras.layers.normalization import BatchNormalization


input_size = 28
filter_size = 3
num_filter = 8
maxpool_size = 2
batch_size = 16
epochs = 10
steps_per_epoch = 24720/batch_size

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense


model = Sequential()
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=1))
model.add(Dropout(uniform(0, 1)))

model.add(Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(Conv2D(32, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))
model.add(Dropout(uniform(0, 1)))  

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(24,activation='softmax'))

 

METRICS = [ 'accuracy']#, 'precision','recall']


model.compile( optimizer= keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=METRICS)

#
model.summary()
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=2)

history = model.fit_generator(
training_set,
steps_per_epoch= steps_per_epoch,
epochs= epochs,
validation_data=validation_set,
validation_steps= 2735  // batch_size,
callbacks=[es]
)
#callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
testing_data_generator = ImageDataGenerator(rescale =1/255)

testing_set = training_data_generator.flow_from_directory('../input/handsignimages/Test', 
                                                          target_size =(28, 28), 
                                                          batch_size = 16 , 
                                                          class_mode ='categorical')
score = model.evaluate_generator(testing_set, steps= len(testing_set))
for idx, metric in enumerate(model.metrics_names):
    print(metric, score[idx])
input_size = 28
filter_size = 3
num_filter = 8
maxpool_size = 2
batch_size = 16
epochs = 10
steps_per_epoch = 24720/batch_size

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense


model = Sequential()
model.add(Conv2D(16, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))


model.add(Conv2D(32, (filter_size,filter_size), 
                 input_shape= (input_size,input_size,3), 
                 activation ='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))
model.add(Dropout(uniform(0, 1)))  

model.add(Conv2D(64, (filter_size,filter_size), 
                 activation='relu', 
                 padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(maxpool_size, maxpool_size),strides=2))
model.add(Dropout(uniform(0, 1)))  

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(24,activation='softmax'))

METRICS = [ 'accuracy']#, 'precision','recall']


model.compile( optimizer= keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=METRICS)

#
model.summary()
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=2)

history = model.fit_generator(
training_set,
steps_per_epoch= steps_per_epoch,
epochs= epochs,
validation_data=validation_set,
validation_steps= 2735  // batch_size,
callbacks=[es]
)

testing_data_generator = ImageDataGenerator(rescale =1/255)

testing_set = training_data_generator.flow_from_directory('../input/handsignimages/Test', 
                                                          target_size =(28, 28), 
                                                          batch_size = 16 , 
                                                          class_mode ='categorical')
score = model.evaluate_generator(testing_set, steps= len(testing_set))
for idx, metric in enumerate(model.metrics_names):
    print(metric, score[idx])
