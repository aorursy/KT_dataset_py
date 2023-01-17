# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

bs=16 # Increase Batch size based on your hardware capacity, if you have powerful GPU try BS=128,256...

train_path=r'../input/bird-species-classification-220-categories/Train'

test_path=r'../input/bird-species-classification-220-categories/Test'
train_gen=ImageDataGenerator(rescale=1./255)

Data_train=train_gen.flow_from_directory(train_path,target_size=(150,150),class_mode='categorical',batch_size=bs)



test_gen=ImageDataGenerator(rescale=1./255)

Data_test=test_gen.flow_from_directory(test_path,target_size=(150,150),class_mode='categorical',batch_size=bs)


model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(150,150,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(150,150,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(200,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
training_steps = Data_train.samples//Data_train.batch_size

validation_steps=Data_test.samples//Data_test.batch_size

history=model.fit_generator(Data_train,epochs=10,steps_per_epoch=training_steps,validation_data=Data_test,validation_steps=validation_steps)
pd.DataFrame(history.history).plot()
from tensorflow.keras.applications.inception_v3 import InceptionV3

import os

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import Model



# Create an instance of the inception model from the local pre-trained weights

# local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(

    input_shape=(150, 150, 3),

    include_top=False,

    weights='imagenet'

)



# Make all the layers in the pre-trained model non-trainable

for layer in pre_trained_model.layers:

  layer.trainable = False



# Print the model summary

pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output



from tensorflow.keras.optimizers import RMSprop



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2

x = layers.Dropout(.2)(x)                  

# Add a final sigmoid layer for classification

x = layers.Dense(1, activation='sigmoid')(x)           



model = Model(pre_trained_model.input, x) 



model.compile(

    optimizer=RMSprop(lr=0.0001), 

    loss='binary_crossentropy', 

    metrics=['accuracy']

)



model.summary()
training_steps = Data_train.samples//Data_train.batch_size

validation_steps=Data_test.samples//Data_test.batch_size

inceptionv3_history=model.fit_generator(Data_train,epochs=2,steps_per_epoch=training_steps,validation_data=Data_test,validation_steps=validation_steps)