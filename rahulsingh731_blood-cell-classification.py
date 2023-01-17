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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import cv2,os,math
train_data = '/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN'

test_data = '/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TEST'
print(os.listdir(train_data))

print(os.listdir(test_data))
categories = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
os.listdir(train_data+'/NEUTROPHIL')
def plot_images(img_path):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = img.astype('float32')/255.0

    plt.imshow(img)

    return

plt.figure(figsize=(15,8))

plt.subplot(221)

plt.title('EOSINOPHIL');plt.axis('off');plot_images(train_data+'/EOSINOPHIL/_33_8228.jpeg')

plt.subplot(222)

plt.title('LYMPHOCYTE');plt.axis('off');plot_images(train_data+'/LYMPHOCYTE/_13_2787.jpeg')

plt.subplot(223)

plt.title('MONOCYTE');plt.axis('off');plot_images(train_data+'/MONOCYTE/_6_4213.jpeg')

plt.subplot(224)

plt.title('NEUTROPHIL');plt.axis('off');plot_images(train_data+'/NEUTROPHIL/_144_8668.jpeg')
print('Training Samples Length')

num_samples = 0

for cell in os.listdir(train_data):

    num_cells = len(os.listdir(os.path.join(train_data,cell)))

    num_samples+=num_cells

    print('cell :{0} length: {1}'.format(cell,num_cells))

print('Total Train Samples {}'.format(num_samples))

print('\n'*2)

print('Testing Samples Length')

num_samples=0

for cell in os.listdir(test_data):

    num_cells = len(os.listdir(os.path.join(test_data,cell)))

    num_samples+=num_cells

    print('cell :{0} length: {1}'.format(cell,num_cells))

print('Total Train Samples {}'.format(num_samples))
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(train_data,

                                                target_size=(64,64),

                                                batch_size=32,

                                                color_mode='rgb',

                                                shuffle=True,

                                                seed = None,

                                                class_mode='categorical'

                                                )
test_set = train_datagen.flow_from_directory(test_data,

                                                target_size=(64,64),

                                                batch_size=32,

                                                color_mode='rgb',

                                                shuffle=True,

                                                seed = None,

                                                class_mode='categorical'

                                                )
model = Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(64,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(4,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit_generator(training_set,steps_per_epoch=100,epochs=20,validation_data=test_set,validation_steps=100)
print(history.history.keys())
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')

plt.xlabel('accuracy')

plt.ylabel('epoch')



plt.legend(['train','test'],loc='upper left')

plt.show()
preds = model.predict_generator(test_set)
preds=np.argmax(preds,axis=1)
preds