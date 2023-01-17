import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.preprocessing import image

from tensorflow.keras import layers

import matplotlib.pyplot as plt
traindir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"

validdir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"

testdir = "../input/new-plant-diseases-dataset/test/test"

train_gen = ImageDataGenerator(rescale = 1. / 255)

valid_gen = ImageDataGenerator(rescale = 1. / 255)
batch_size = 32

target_size=(224, 224)

epochs = 5
train_data = train_gen.flow_from_directory(directory=traindir ,batch_size=batch_size , shuffle=True,target_size=target_size ,class_mode='categorical')

valid_data = valid_gen.flow_from_directory(directory=validdir ,batch_size=batch_size , shuffle=True,target_size=target_size ,class_mode='categorical')
base_model=VGG16(include_top=False,input_shape=(224,224,3))

base_model.trainable=False
inputs = tf.keras.Input(shape=(224,224, 3))

model_layers = base_model(inputs)

model_layers = layers.Flatten()(model_layers)

outputs = layers.Dense(38, activation='softmax')(model_layers)



model = tf.keras.Model(inputs,outputs)



print(model.summary())

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=["accuracy"])



history=model.fit(train_data,steps_per_epoch=len(train_data)/epochs, epochs=epochs , validation_data=valid_data,validation_steps=len(valid_data))

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
base_model2=MobileNetV2(include_top=False,input_shape=(224,224,3))

base_model2.trainable=False
inputs2 = tf.keras.Input(shape=(224,224, 3))

model_layers2 = base_model2(inputs2)

model_layers2 = layers.Flatten()(model_layers2)

outputs2 = layers.Dense(38, activation='softmax')(model_layers2)



model2 = tf.keras.Model(inputs2,outputs2)



print(model2.summary())

model2.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=["accuracy"])



history2 = model2.fit(train_data,steps_per_epoch=len(train_data)/epochs, epochs=epochs , validation_data=valid_data,validation_steps=len(valid_data))

# list all data in history

print(history2.history.keys())

# summarize history for accuracy

plt.plot(history2.history['accuracy'])

plt.plot(history2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
base_model3=MobileNetV2(include_top=False,input_shape=(224,224,3))

base_model3.trainable=True
inputs3 = tf.keras.Input(shape=(224,224, 3))

model_layers3 = base_model3(inputs3)

model_layers3 = layers.Flatten()(model_layers3)

outputs3 = layers.Dense(38, activation='softmax')(model_layers3)



model3 = tf.keras.Model(inputs3,outputs3)



print(model3.summary())

model3.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=["accuracy"])



history3 = model3.fit(train_data,steps_per_epoch=len(train_data)/epochs, epochs=epochs , validation_data=valid_data,validation_steps=len(valid_data))

# list all data in history

print(history3.history.keys())

# summarize history for accuracy

plt.plot(history3.history['accuracy'])

plt.plot(history3.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()