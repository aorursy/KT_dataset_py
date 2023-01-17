import numpy as np
import pandas as pd 
import os
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
from keras.layers import GlobalAveragePooling2D

#create dataframe for training images
train_file = []       
train_label = []

for dirname, _, filenames in os.walk('../input/pokemon-generation-one/dataset'):
    for filename in filenames:
        train_file.append(os.path.join(dirname, filename))
        train_label.append(dirname.split('/')[-1])
        
train_image = pd.DataFrame(train_file)
train_image.columns = ['file_name']
train_image['target'] = train_label

print (train_image)




train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True, 
                                    validation_split = 0.2)

train_generator = train_datagen.flow_from_dataframe(
        train_image,
        x_col='file_name',
        y_col='target',
        target_size=(224, 224),
        batch_size=64,
        shuffle=True,
        class_mode='categorical',
        subset='training')

test_generator = train_datagen.flow_from_dataframe(
        train_image,
        x_col='file_name',
        y_col='target',
        target_size=(224, 224),
        batch_size=64,
        shuffle=True,
        class_mode='categorical',
        subset='validation')
#Transfer learning from ResNet50 with imagenet weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
X = GlobalAveragePooling2D()(model.output)
X = Dense(1000,activation = 'relu')(X)
X = Dense(500,activation='relu')(X)
output = Dense(149,activation='softmax')(X)

final_model = Model(inputs=model.input,outputs = output)
#Compile the model
final_model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
#Fit the model for training and validation dataset
batch_size=16
train_steps = train_image.shape[0]//batch_size
history = final_model.fit_generator(
    train_generator,
    
    validation_data = test_generator,
    epochs=15)
