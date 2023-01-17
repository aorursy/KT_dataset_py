import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_data_path = '../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(  
    train_data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(
    valid_data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')
for i, j in train_generator:
    print(i.shape)
    print(j.shape)
    break
plt.imshow(i[0])
plt.show()
print(j[0])
def model():
    #input: 128,128,3
    #output: 38
    pass
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator))