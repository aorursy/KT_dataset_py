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
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import keras
import PIL
from PIL import Image
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout , BatchNormalization, Activation

print(np.asarray(Image.open("/kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image.jpg")).shape)
Image.open("/kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image.jpg")
train_data = image_dataset_from_directory(directory = "/kaggle/input/brain-tumor-classification-mri/Training",labels ="inferred",label_mode = 'categorical',image_size=(224, 224))
test_data = image_dataset_from_directory(directory = "/kaggle/input/brain-tumor-classification-mri/Testing",labels ="inferred",label_mode = 'categorical',image_size=(224, 224))
# Build model

model=Sequential()
# layer 1
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))

# layer 2 
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))

#layer 3
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))

# flatten image
model.add(Flatten())

# dense layers
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))

# output layer
model.add(Dense(units=4,activation='softmax'))

# view model summary
model.summary()

# compile data
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=test_data)
# Model after first 10 epochs
model.evaluate(test_data)[1]
model.fit(train_data, epochs=30, validation_data=test_data)
model.evaluate(test_data)[1]
# Build model

model=Sequential()
# layer 1
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(BatchNormalization())

# layer 2 
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(BatchNormalization())

#layer 3
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(BatchNormalization())

# flatten image
model.add(Flatten())

# dense layers
model.add(Dense(units=512,activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.25))
# output layer
model.add(Dense(units=4,activation='softmax'))

# view model summary
model.summary()

# compile data
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data, epochs=30, validation_data=test_data)
temp1 = model.history.history
plt.plot(temp1["loss"],label ='Traing Loss')
plt.plot(temp1["val_loss"],label ='Validation Loss')
plt.legend()
plt.show()

plt.plot(temp1["accuracy"],label="Training Accuracy")
plt.plot(temp1["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()
model.evaluate(test_data)[1]
# trainig 2 
model.fit(train_data, epochs=75, validation_data=test_data, batch_size = 75)
hist1 = model.history.history
plt.plot(hist1["loss"],label ='Traing Loss')
plt.plot(hist1["val_loss"],label ='Validation Loss')
plt.legend()
plt.show()

plt.plot(hist1["accuracy"],label="Training Accuracy")
plt.plot(hist1["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()
model.evaluate(test_data)[1]
train_data = image_dataset_from_directory(directory = "/kaggle/input/brain-tumor-classification-mri/Training",labels ="inferred",label_mode = 'categorical',image_size=(299, 299))
test_data = image_dataset_from_directory(directory = "/kaggle/input/brain-tumor-classification-mri/Testing",labels ="inferred",label_mode = 'categorical',image_size=(299, 299))
from keras.applications import Xception
from keras import Model
base_model = Xception(include_top=False,input_shape=(299,299,3))
base_model.trainable = False
base_model.summary()
flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(4, activation='softmax')(class1)
# define new model
model = Model(inputs=base_model.inputs, outputs=output)
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data, epochs=30, validation_data=test_data)
hist2 = model.history.history
plt.plot(hist2["loss"],label ='Traing Loss')
plt.plot(hist2["val_loss"],label ='Validation Loss')
plt.legend()
plt.show()

plt.plot(hist2["accuracy"],label="Training Accuracy")
plt.plot(hist2["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()
model.evaluate(test_data)[1]





