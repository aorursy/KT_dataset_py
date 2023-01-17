# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
       # print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#assign image width & height
image_width = 128
image_height = 128

parasitized_img = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/')
uninfected_img = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/')
print("Number of parasitized Image:",len(parasitized_img))
print("Number of uninfected Image:", len(uninfected_img))
import random
from tensorflow.keras.preprocessing.image import load_img
list1 = ['../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'
,  '../input/cell-images-for-detecting-malaria/cell_images/Uninfected/']

fig = plt.figure(figsize=(12, 8))
#fig.set_size_inches(,13)
plt.style.use("ggplot")
j=1
for i in list1: 
    for k in range(5):

        filenames  = os.listdir(i)
        sample = random.choice(filenames)
        image = load_img(i+sample) #'../input/10-monkey-species/training/training/n1'
        plt.subplot(2,5,j)
        plt.imshow(image)
        plt.xlabel(i.split("/")[-2])
        j+=1
plt.tight_layout()
datagen = ImageDataGenerator(rescale=1/255.0,validation_split=0.2)
train_data_generator = datagen.flow_from_directory(directory = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                                   target_size=(image_width,image_height),
                                                   class_mode = 'binary',
                                                   batch_size=16,
                                                   subset='training'
                                                  )

validation_data_generator = datagen.flow_from_directory(directory = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                                   target_size=(image_width,image_height),
                                                   class_mode = 'binary',
                                                   batch_size=16,
                                                   subset='validation'
                                                  )
model = Sequential()
model.add(Conv2D(16,(3,3),input_shape=(image_width,image_height,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(generator=train_data_generator,steps_per_epoch=len(train_data_generator),
                              epochs=5,
                              validation_data=validation_data_generator,validation_steps=len(validation_data_generator),
                             verbose=1)
#accuracy
plt.plot(history.history['accuracy'],'coral')
plt.plot(history.history['val_accuracy'],'gold')
plt.title('Model Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train','Val'],loc='upper left')
plt.show()

#Loss
plt.plot(history.history['loss'],'coral')
plt.plot(history.history['val_loss'],'gold')
plt.title('Model Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train','Val'],loc='upper left')
plt.show()
from tensorflow.keras.applications.resnet50 import ResNet50
resnet = ResNet50(weights='imagenet',include_top=False,input_shape=(128,128,3))
x = resnet.output
x = Flatten()(x)
x = Dense(64,activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(1,activation='sigmoid')(x)
from tensorflow.keras.models import Model
final_model = Model(inputs=resnet.input,outputs = prediction)
final_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = final_model.fit_generator(generator=train_data_generator,steps_per_epoch=len(train_data_generator),
                              epochs=5,
                              validation_data=validation_data_generator,validation_steps=len(validation_data_generator),
                             verbose=1)
#accuracy
plt.plot(history.history['accuracy'],'b')
plt.plot(history.history['val_accuracy'],'r')
plt.title('Model Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train','Val'],loc='upper left')
plt.show()

#Loss
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
plt.title('Model Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train','Val'],loc='upper left')
plt.show()
